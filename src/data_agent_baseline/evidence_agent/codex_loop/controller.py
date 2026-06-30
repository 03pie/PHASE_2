from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from data_agent_baseline.agents.deep_state import DeepAgentConfig, TraceCallback
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.evidence_agent.codex_loop.context import (
    SYSTEM_PROMPT,
    build_context_fragments,
    render_user_context,
)
from data_agent_baseline.evidence_agent.codex_loop.document_agent import DocumentAgent
from data_agent_baseline.evidence_agent.codex_loop.guard import guard_action
from data_agent_baseline.evidence_agent.codex_loop.inventory import build_inventory
from data_agent_baseline.evidence_agent.codex_loop.native_tools import (
    action_from_tool_call,
    bind_native_tools,
    extract_tool_calls,
    tool_output_content,
)
from data_agent_baseline.evidence_agent.codex_loop.protocol import (
    COMPUTABLE_BINDING_TYPES,
    Evidence,
    LoopState,
    ModelAction,
    ToolInvocation,
    ToolOutputEnvelope,
    TranscriptWindow,
)
from data_agent_baseline.evidence_agent.codex_loop.registry import EvidenceActionRegistry
from data_agent_baseline.evidence_agent.codex_loop.state_views import (
    answer_contract_view,
    answer_candidates,
    final_output_contract,
    selected_source_candidates,
    semantic_selection_view,
    source_coverage_map,
)
from data_agent_baseline.evidence_agent.knowledge import (
    build_knowledge_catalog,
    build_semantic_cards,
    build_source_mappings,
    expand_semantic_card_dependencies,
    match_knowledge_sections,
    match_semantic_cards,
)
from data_agent_baseline.evidence_agent.tracing import EvidenceTrace

_SQL_RELATION_PATTERN = re.compile(
    r'\b(from|join)\s+("([^"]+)"|[A-Za-z_][\w]*)',
    re.IGNORECASE,
)

_DEFAULT_TOOL_CHOICE = "required"


def _compact(value: Any, *, limit: int = 40) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > 8_000:
            return value[:7_900] + "\n...[truncated]"
        return value
    if isinstance(value, dict):
        return {str(key): _compact(item, limit=limit) for key, item in list(value.items())[:limit]}
    if isinstance(value, (list, tuple)):
        return [_compact(item, limit=limit) for item in list(value)[:limit]]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "to_dict"):
        return _compact(value.to_dict(), limit=limit)
    if hasattr(value, "item"):
        try:
            return _compact(value.item(), limit=limit)
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        return str(value)
    return value


def _semantic_norm(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value or "").casefold())


def _observed_column_for_field(field: str | None, allowed_columns: tuple[str, ...]) -> str | None:
    field_norm = _semantic_norm(field)
    if not field_norm:
        return None
    for column in allowed_columns:
        if _semantic_norm(column) == field_norm:
            return str(column)
    return None


def _field_variants(field_id: str) -> set[str]:
    text = str(field_id or "").casefold()
    variants = {text}
    if "." in text:
        variants.add(text.rsplit(".", 1)[-1])
    return {item for item in variants if item}


def _physical_columns_from_contract_value(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, dict):
        return []
    columns: list[str] = []
    for key in ("source_field", "source_column", "field", "column", "physical_field", "physical_column"):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            columns.append(raw)
    return columns


def _record_columns(records: Any) -> tuple[str, ...]:
    if not isinstance(records, list):
        return ()
    columns: list[str] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        for column in record:
            text = str(column).strip()
            if not text or text == "provenance" or text in seen:
                continue
            seen.add(text)
            columns.append(text)
    return tuple(columns)


def _merge_columns(*groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for column in group:
            text = str(column).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
    return tuple(merged)


def _sanitize_semantic_contract(
    state: LoopState,
    contract: dict[str, Any],
    *,
    source_id: str | None,
    binding_type: str,
    allowed_columns: tuple[str, ...],
) -> tuple[dict[str, Any], list[str]]:
    if not contract or not allowed_columns:
        return {}, []
    allowed_norms = {_semantic_norm(column): str(column) for column in allowed_columns}
    allowed_statuses = (
        {"unverified_document_candidate"}
        if binding_type == "document_record_set"
        else {"unverified_structured_candidate"}
        if binding_type in {"structured_source", "structured_field"}
        else set()
    )
    selected_card_ids = (
        set(state.semantic_selection.card_ids)
        if state.semantic_selection is not None
        else set()
    )
    grounded_fields = {
        f"{card.semantic_scope}.{card.semantic_slot}".casefold()
        for card in state.semantic_cards
        if card.semantic_slot
        and (not selected_card_ids or card.id in selected_card_ids)
        for mapping in state.source_mappings
        if mapping.card_id == card.id
        and mapping.source_id == source_id
        and mapping.status in allowed_statuses
    }
    physical_mapping: dict[str, Any] = {}
    rejected: list[str] = []
    raw_mapping = contract.get("physical_field_mapping")
    if isinstance(raw_mapping, dict):
        for key, value in raw_mapping.items():
            field_id = str(key).casefold()
            if field_id not in grounded_fields:
                rejected.append(field_id)
                continue
            physical_columns = _physical_columns_from_contract_value(value)
            observed = [column for column in physical_columns if _semantic_norm(column) in allowed_norms]
            if not observed:
                rejected.append(field_id)
                continue
            if isinstance(value, dict):
                physical_mapping[field_id] = {**value, "field": allowed_norms[_semantic_norm(observed[0])]}
            else:
                physical_mapping[field_id] = allowed_norms[_semantic_norm(observed[0])]
    canonical_fields: list[str] = []
    raw = contract.get("canonical_fields")
    if isinstance(raw, list):
        for item in raw:
            field_id = str(item).casefold()
            if field_id not in grounded_fields:
                rejected.append(field_id)
                continue
            if field_id in physical_mapping:
                canonical_fields.append(field_id)
                continue
            variants = _field_variants(field_id)
            observed = next((variant for variant in variants if _semantic_norm(variant) in allowed_norms), None)
            if observed:
                canonical_fields.append(field_id)
                physical_mapping.setdefault(field_id, allowed_norms[_semantic_norm(observed)])
            elif field_id not in rejected:
                rejected.append(field_id)
    if not canonical_fields and not physical_mapping:
        return {}, sorted(set(rejected))
    sanitized = {
        key: value
        for key, value in contract.items()
        if key not in {"canonical_fields", "physical_field_mapping"}
    }
    sanitized["canonical_fields"] = sorted(set(canonical_fields or physical_mapping))
    sanitized["physical_field_mapping"] = physical_mapping
    return sanitized, sorted(set(rejected))


def _infer_semantic_contract(
    state: LoopState,
    *,
    binding_type: str,
    source_id: str | None,
    table: str | None,
    allowed_columns: tuple[str, ...],
) -> dict[str, Any]:
    if not source_id:
        return {}
    column_norms = {_semantic_norm(column) for column in allowed_columns}
    table_norm = _semantic_norm(table)
    canonical_fields: list[str] = []
    physical_field_mapping: dict[str, dict[str, Any]] = {}
    selected_card_ids = (
        set(state.semantic_selection.card_ids)
        if state.semantic_selection is not None
        else set()
    )
    if not selected_card_ids:
        return {}
    for card in state.semantic_cards:
        if card.id not in selected_card_ids:
            continue
        if not card.semantic_slot:
            continue
        observed_column = _observed_column_for_field(card.semantic_slot, allowed_columns)
        if column_norms and observed_column is None:
            continue
        field_id = f"{card.semantic_scope}.{card.semantic_slot}".casefold()
        for mapping in state.source_mappings:
            if mapping.card_id != card.id or mapping.source_id != source_id:
                continue
            if mapping.status == "unverified_structured_candidate" and binding_type in {"structured_source", "structured_field"}:
                mapped_table = _semantic_norm(mapping.physical_table)
                if table_norm and mapped_table and table_norm != mapped_table:
                    continue
                canonical_fields.append(field_id)
                physical_field_mapping[field_id] = {
                    "source_id": source_id,
                    "source_path": mapping.source_path,
                    "table": table,
                    "field": observed_column or mapping.physical_field or card.semantic_slot,
                    "mapping_status": mapping.status,
                }
            elif mapping.status == "unverified_document_candidate" and binding_type == "document_record_set":
                canonical_fields.append(field_id)
                physical_field_mapping[field_id] = {
                    "source_id": source_id,
                    "source_path": mapping.source_path,
                    "field": observed_column or mapping.physical_field or card.semantic_slot,
                    "mapping_status": mapping.status,
                }
    canonical_fields = sorted(set(canonical_fields))
    if not canonical_fields:
        return {}
    return {
        "canonical_fields": canonical_fields,
        "physical_field_mapping": physical_field_mapping,
    }


def _progress_identity(evidence: Evidence) -> str:
    payload = evidence.payload or {}
    identity: dict[str, Any] = {
        "tool": evidence.tool_name,
        "ok": evidence.ok,
        "source_id": evidence.source_id,
        "data_form": evidence.data_form,
        "summary": evidence.summary,
    }
    for key in (
        "query",
        "value",
        "path",
        "source_id",
        "table",
        "compute_ref",
        "sql",
    ):
        if key in payload:
            identity[key] = payload.get(key)
    if isinstance(payload.get("columns"), list):
        identity["columns"] = payload["columns"]
    hits = payload.get("hits")
    if isinstance(hits, list):
        identity["hit_count"] = len(hits)
        identity["hit_scope"] = [
            {
                "source_id": hit.get("source_id"),
                "table": hit.get("table"),
                "matched_column": hit.get("matched_column"),
                "row_index": hit.get("row_index"),
                "line": hit.get("line"),
            }
            for hit in hits[:8]
            if isinstance(hit, dict)
        ]
    windows = payload.get("windows")
    if isinstance(windows, list):
        identity["window_count"] = len(windows)
        identity["window_scope"] = [window for window in windows[:8] if isinstance(window, dict)]
    matches = payload.get("matches")
    if isinstance(matches, list):
        identity["match_count"] = len(matches)
        identity["match_scope"] = [
            {
                "line": match.get("line"),
                "page": match.get("page"),
                "recommended_read": match.get("recommended_read"),
            }
            for match in matches[:8]
            if isinstance(match, dict)
        ]
    slice_catalog = payload.get("slice_catalog")
    if isinstance(slice_catalog, list):
        identity["slice_count"] = len(slice_catalog)
    if isinstance(payload.get("records"), list):
        identity["record_count"] = len(payload["records"])
    return json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str)


def _trace_action_name(action: ModelAction, evidence: Evidence, *, guard_allowed: bool) -> str:
    if not guard_allowed:
        return "tool_error"
    if action.kind == "tool_call":
        return action.tool_name or evidence.tool_name
    if action.kind == "compute":
        return "run_verified_compute"
    if action.kind == "final":
        return "submit_final"
    return str(action.kind)


def _model_turn_trace_action(tool_calls: list[dict[str, Any]]) -> str:
    tool_names = [
        str(call.get("name") or "").strip()
        for call in tool_calls
        if str(call.get("name") or "").strip()
    ]
    if not tool_names:
        return "llm_response"
    return "+".join(tool_names)


def _model_turn_action_input(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    if not tool_calls:
        return {}
    compact_calls = [
        {
            "tool_name": str(call.get("name") or ""),
            "tool_call_id": call.get("id"),
            "arguments": _compact(call.get("args") or {}),
        }
        for call in tool_calls
    ]
    if len(compact_calls) == 1:
        call = compact_calls[0]
        return {
            "tool_name": call["tool_name"],
            "tool_call_id": call["tool_call_id"],
            "arguments": call["arguments"],
        }
    return {"tool_calls": compact_calls}


def _model_response_thought(response: Any) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                parts.append(json.dumps(item, ensure_ascii=False, default=str))
                continue
            parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _answer_from_compute(state: LoopState, compute_ref: str) -> AnswerTable | None:
    compute = state.compute_results.get(compute_ref)
    if compute is None or not compute.ok:
        return None
    return AnswerTable(columns=list(compute.columns), rows=[list(row) for row in compute.rows])


def _answer_from_final(state: LoopState) -> AnswerTable | None:
    answer = state.final_answer or {}
    columns = answer.get("columns")
    rows = answer.get("rows")
    if isinstance(columns, list) and isinstance(rows, list):
        normalized_rows: list[list[Any]] = []
        for row in rows:
            if isinstance(row, list):
                normalized_rows.append(row)
            elif isinstance(row, tuple):
                normalized_rows.append(list(row))
            else:
                normalized_rows.append([row])
        return AnswerTable(columns=[str(column) for column in columns], rows=normalized_rows)
    compute_ref = str(answer.get("compute_ref") or "")
    if compute_ref:
        return _answer_from_compute(state, compute_ref)
    return None


def _canonicalize_sql_relation_names(state: LoopState, sql: str) -> tuple[str, bool]:
    alias_sets: dict[str, set[str]] = {}
    for binding in state.bindings.values():
        if not binding.relation_name:
            continue
        aliases = {binding.relation_name}
        if binding.table:
            aliases.add(binding.table)
        source = state.sources.get(binding.source_id or "")
        if source is not None:
            aliases.update({source.stem, source.basename})
            aliases.update(source.tables)
        for alias in aliases:
            alias_text = str(alias).strip()
            if not alias_text:
                continue
            alias_sets.setdefault(alias_text.casefold(), set()).add(binding.relation_name)
    alias_map = {
        alias: next(iter(relations))
        for alias, relations in alias_sets.items()
        if len(relations) == 1
    }

    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        keyword = match.group(1)
        raw_relation = match.group(3) or match.group(2)
        relation = raw_relation.strip('"')
        canonical = alias_map.get(relation.casefold())
        if not canonical or canonical == relation:
            return match.group(0)
        changed = True
        return f"{keyword} {canonical}"

    return _SQL_RELATION_PATTERN.sub(replace, sql), changed


def _audit_binding_refs(
    state: LoopState,
    binding_refs: list[str],
    evidence_refs: list[str],
) -> tuple[list[str], list[str], list[str]]:
    missing_requirements: list[str] = []
    weak_bindings: list[str] = []
    conflicts: list[str] = []
    for binding_ref in binding_refs:
        binding = state.bindings.get(binding_ref)
        if binding is None:
            missing_requirements.append(f"unknown_binding:{binding_ref}")
            continue
        if not binding.evidence_refs:
            weak_bindings.append(f"{binding_ref}:missing_evidence_refs")
        if (
            binding.binding_type == "document_record_set"
            and binding.metadata.get("partial_coverage")
        ):
            weak_bindings.append(f"{binding_ref}:partial_document_record_set_coverage")
        for evidence_ref in binding.evidence_refs:
            evidence = state.evidence.get(evidence_ref)
            if evidence is None:
                weak_bindings.append(f"{binding_ref}:unknown_evidence:{evidence_ref}")
            elif not evidence.ok:
                weak_bindings.append(f"{binding_ref}:failed_evidence:{evidence_ref}")
            elif evidence.data_form == "video":
                conflicts.append("video_evidence_in_final")
    for evidence_ref in evidence_refs:
        evidence = state.evidence.get(evidence_ref)
        if evidence is None:
            missing_requirements.append(f"unknown_evidence:{evidence_ref}")
        elif not evidence.ok:
            weak_bindings.append(f"failed_final_evidence:{evidence_ref}")
        elif evidence.data_form == "video":
            conflicts.append("video_evidence_in_final")
    return missing_requirements, weak_bindings, conflicts


def _audit_final(state: LoopState) -> dict[str, Any]:
    answer = state.final_answer or {}
    compute_ref = str(answer.get("compute_ref") or "")
    compute = state.compute_results.get(compute_ref)
    issues: list[str] = []
    warnings: list[str] = []
    missing_requirements: list[str] = []
    unsupported_operations: list[str] = []
    weak_bindings: list[str] = []
    conflicts: list[str] = []
    binding_refs: list[str] = []
    evidence_refs: list[str] = []
    final_mode = "compute" if compute_ref else "direct"
    output_contract = final_output_contract(state)
    if compute_ref:
        if compute is None:
            missing_requirements.append("unknown_compute_ref")
        elif not compute.ok:
            unsupported_operations.append("failed_compute_ref")
        else:
            if not compute.binding_refs:
                missing_requirements.append("missing_binding_lineage")
            if not compute.columns:
                missing_requirements.append("missing_output_columns")
            if not compute.rows:
                missing_requirements.append("missing_output_rows")
            binding_refs = list(compute.binding_refs)
            evidence_refs = list(compute.evidence_refs)
    else:
        columns = answer.get("columns")
        rows = answer.get("rows")
        binding_refs = [str(item) for item in answer.get("binding_refs", [])]
        evidence_refs = [str(item) for item in answer.get("evidence_refs", [])]
        if not isinstance(columns, list) or not columns:
            missing_requirements.append("missing_output_columns")
        if not isinstance(rows, list):
            missing_requirements.append("missing_output_rows")
        if not binding_refs:
            missing_requirements.append("missing_binding_lineage")
        if not evidence_refs:
            missing_requirements.append("missing_evidence_lineage")

    more_missing, more_weak, more_conflicts = _audit_binding_refs(
        state, binding_refs, evidence_refs
    )
    missing_requirements.extend(more_missing)
    weak_bindings.extend(more_weak)
    conflicts.extend(dict.fromkeys(more_conflicts))
    warnings.extend(f"output_contract:{warning}" for warning in output_contract["warnings"])
    if state.final_answer and not output_contract["passed"]:
        unsupported_operations.extend(
            f"output_contract:{issue}" for issue in output_contract["issues"]
        )
    issues.extend(missing_requirements)
    issues.extend(unsupported_operations)
    issues.extend(weak_bindings)
    issues.extend(conflicts)

    return {
        "answer_present": bool(state.final_answer),
        "final_mode": final_mode,
        "compute_ref": compute_ref or None,
        "binding_refs": binding_refs,
        "evidence_refs": evidence_refs,
        "missing_requirements": missing_requirements,
        "unsupported_operations": unsupported_operations,
        "weak_bindings": weak_bindings,
        "conflicts": conflicts,
        "audit_warnings": warnings,
        "final_output_contract": output_contract,
        "issues": issues,
        "passed": bool(state.final_answer) and not issues,
    }


class CodexEvidenceController:
    """Model-driven controlled evidence loop.

    The controller mirrors the Codex turn pattern: build bounded context, ask
    the model for one typed action, guard it, dispatch through a typed registry,
    append the observation to the ledger, then repeat.
    """

    def __init__(self, *, model: Any, config: DeepAgentConfig) -> None:
        self.model = model
        self.default_tool_choice = _DEFAULT_TOOL_CHOICE
        self.tool_model = bind_native_tools(model, tool_choice=self.default_tool_choice)
        self.config = config
        self.document_agent = DocumentAgent(model=model)
        self.registry = EvidenceActionRegistry(document_agent=self.document_agent)

    def _emit(
        self,
        *,
        task_id: str,
        trace: EvidenceTrace,
        callback: TraceCallback | None,
        status: str,
        answer: AnswerTable | None = None,
        failure_reason: str | None = None,
    ) -> None:
        if callback is None:
            return
        callback(
            AgentRunResult(
                task_id=task_id,
                answer=answer,
                steps=trace.snapshot(),
                failure_reason=failure_reason,
            ),
            status,
        )

    def _bootstrap(self, task: PublicTask, state: LoopState, trace: EvidenceTrace) -> None:
        state.sources = build_inventory(task.context_dir)
        state.source_by_path = {}
        for source in state.sources.values():
            state.source_by_path[source.id] = source.id
            state.source_by_path[source.virtual_path] = source.id
            state.source_by_path[source.path.as_posix()] = source.id
        trace.add(
            action="codex_bootstrap_inventory",
            thought="Observe the context inventory before any model-selected action.",
            observation={
                "source_count": len(state.sources),
                "sources": [
                    {
                        "id": source.id,
                        "path": source.virtual_path,
                        "data_form": source.data_form,
                        "tables": list(source.tables),
                        "columns": list(source.columns[:20]),
                        "metadata": source.metadata,
                    }
                    for source in state.sources.values()
                ],
            },
        )

        sections, lookup, schema_json, content_hash = build_knowledge_catalog(task.context_dir)
        state.knowledge_sections = sections
        state.knowledge_lookup = lookup
        state.matched_sections = match_knowledge_sections(task.question, sections)
        state.semantic_cards = build_semantic_cards(sections)
        state.matched_semantic_cards = expand_semantic_card_dependencies(
            match_semantic_cards(task.question, state.semantic_cards),
            state.semantic_cards,
        )
        state.source_mappings = build_source_mappings(
            state.semantic_cards,
            tuple(state.sources.values()),
        )
        forbidden_terms = ("profile", "physical_schema", "table_schema", "field_schema")
        trace.add(
            action="codex_bootstrap_knowledge",
            thought="Compile knowledge.md into pure semantic cards; knowledge guides meaning but does not define physical data format.",
            observation={
                "content_hash": content_hash,
                "section_count": len(sections),
                "lookup_count": len(lookup),
                "semantic_card_count": len(state.semantic_cards),
                "catalog": [
                    {
                        "id": section.id,
                        "heading_path": section.heading_path,
                        "line_start": section.line_start,
                        "line_end": section.line_end,
                        "mentions": list(section.mentions[:12]),
                    }
                    for section in sections
                ],
                "lookup_tokens": [
                    {
                        "token": entry.token,
                        "section_refs": list(entry.section_refs),
                    }
                    for entry in list(lookup.values())[:120]
                ],
                "matched_sections": [
                    {
                        "id": section.id,
                        "heading_path": section.heading_path,
                        "line_start": section.line_start,
                        "line_end": section.line_end,
                        "text": section.text,
                    }
                    for section in state.matched_sections[:8]
                ],
                "matched_semantic_cards": [
                    card.to_dict()
                    for card in state.matched_semantic_cards[:12]
                ],
                "schema_contract_ok": all(term not in schema_json for term in forbidden_terms),
            },
        )
        trace.add(
            action="codex_bootstrap_source_resolution",
            thought="Prepare source candidate ledger, but do not expose candidates until the LLM selects semantic cards.",
            observation={
                "source_mapping_count": len(state.source_mappings),
                "selected_source_mapping_count": 0,
                "selection_required": True,
                "note": (
                    "Full source mappings are internal until select_semantic_cards records "
                    "the task-specific business semantic selection."
                ),
            },
        )

        document_indexes = self.document_agent.ensure_indexes(state)
        trace.add(
            action="codex_bootstrap_document_agent",
            thought="Pre-index PDF/MD sources as record slices for the isolated DocumentAgent loop.",
            observation={
                "document_count": len(document_indexes),
                "total_slice_count": sum(index.slice_count for index in document_indexes.values()),
                "documents": [
                    {
                        "source_id": index.source_id,
                        "path": index.path,
                        "data_form": index.data_form,
                        "slice_count": index.slice_count,
                        "page_count": index.page_count,
                    }
                    for index in document_indexes.values()
                ],
            },
        )

    def _prompt_messages(
        self,
        state: LoopState,
        *,
        transcript: TranscriptWindow | None = None,
        last_error: str | None = None,
        recovery_hint: dict[str, Any] | None = None,
        extra_instruction: str | None = None,
    ) -> tuple[list[Any], list[str], dict[str, bool]]:
        fragments = build_context_fragments(
            state,
            last_error=last_error,
            recovery_hint=recovery_hint,
        )
        context = render_user_context(fragments)
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)]
        if transcript is not None:
            messages.extend(transcript.messages())
        if extra_instruction:
            messages.append(HumanMessage(content=extra_instruction))
        return (
            messages,
            [fragment.id for fragment in fragments],
            {fragment.id: fragment.truncated for fragment in fragments if fragment.truncated},
        )

    def _initial_messages(
        self,
        state: LoopState,
        *,
        last_error: str | None = None,
        extra_instruction: str | None = None,
    ) -> tuple[list[Any], list[str], dict[str, bool]]:
        return self._prompt_messages(
            state,
            transcript=None,
            last_error=last_error,
            extra_instruction=extra_instruction,
        )

    def _call_model(
        self,
        messages: list[Any],
        *,
        tool_choice: str | dict[str, Any] | None = None,
        tool_names: tuple[str, ...] | None = None,
    ) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
        last_exc: Exception | None = None
        effective_tool_choice = tool_choice if tool_choice is not None else self.default_tool_choice
        tool_model = (
            bind_native_tools(
                self.model,
                tool_choice=effective_tool_choice,
                tool_names=tool_names,
            )
            if tool_choice is not None or tool_names is not None
            else self.tool_model
        )
        for attempt in range(3):
            try:
                response = tool_model.invoke(messages)
                break
            except Exception as exc:  # noqa: BLE001 - model transport errors are retriable
                last_exc = exc
                if attempt == 0 and isinstance(effective_tool_choice, dict):
                    effective_tool_choice = self.default_tool_choice
                    tool_model = bind_native_tools(
                        self.model,
                        tool_choice=effective_tool_choice,
                        tool_names=tool_names,
                    )
                    continue
                if attempt >= 2:
                    raise
                time.sleep(min(2.0, 0.5 * (attempt + 1)))
        else:  # pragma: no cover - kept for type checkers
            raise last_exc or RuntimeError("model invocation failed")
        if self.config.model_call_interval_seconds:
            time.sleep(self.config.model_call_interval_seconds)
        tool_calls = extract_tool_calls(response)
        raw = {
            "content": getattr(response, "content", ""),
            "tool_calls": tool_calls,
            "invalid_tool_calls": _compact(getattr(response, "invalid_tool_calls", None) or []),
            "response_metadata": getattr(response, "response_metadata", {}),
            "tool_choice": effective_tool_choice,
            "tool_names": list(tool_names) if tool_names else None,
        }
        return response, tool_calls, raw

    def _apply_bind(self, state: LoopState, action: ModelAction) -> Evidence:
        evidence_items = [state.evidence[ref] for ref in action.evidence_refs]
        arguments = action.arguments
        source_id = action.source_ref or evidence_items[0].source_id
        table = str(arguments.get("table") or evidence_items[0].payload.get("table") or "").strip() or None
        field = str(arguments.get("field") or "").strip() or None
        allowed_columns_raw = arguments.get("allowed_columns")
        if isinstance(allowed_columns_raw, list):
            allowed_columns = tuple(str(column) for column in allowed_columns_raw if str(column).strip())
        else:
            allowed_columns = ()
        if not allowed_columns:
            for item in evidence_items:
                columns = item.payload.get("columns")
                if isinstance(columns, list):
                    allowed_columns = tuple(str(column) for column in columns)
                    break
        metadata: dict[str, Any] = {}
        if isinstance(arguments.get("answer"), dict):
            metadata["answer"] = arguments["answer"]
        raw_semantic_mappings = arguments.get("semantic_mappings")
        if isinstance(raw_semantic_mappings, list):
            semantic_mappings = [
                dict(item)
                for item in raw_semantic_mappings
                if isinstance(item, dict)
            ]
            if semantic_mappings:
                metadata["semantic_mappings"] = semantic_mappings
        explicit_contract: dict[str, Any] = {}
        if isinstance(arguments.get("semantic_contract"), dict):
            explicit_contract.update(arguments["semantic_contract"])
        canonical_fields_arg = arguments.get("canonical_fields")
        if isinstance(canonical_fields_arg, list):
            explicit_contract["canonical_fields"] = [
                str(item).casefold()
                for item in canonical_fields_arg
                if str(item).strip()
            ]
        physical_mapping_arg = arguments.get("physical_field_mapping")
        if isinstance(physical_mapping_arg, dict):
            explicit_contract["physical_field_mapping"] = physical_mapping_arg
        semantic_card_ids_arg = arguments.get("semantic_card_ids")
        if isinstance(semantic_card_ids_arg, list):
            explicit_contract["semantic_card_ids"] = [
                str(item)
                for item in semantic_card_ids_arg
                if str(item).strip()
            ]
        if action.binding_type == "document_record_set":
            for item in evidence_items:
                records = item.payload.get("records")
                if isinstance(records, list):
                    record_columns = _record_columns(records)
                    allowed_columns = _merge_columns(allowed_columns, record_columns)
                    metadata["records"] = records
                    metadata["record_count"] = len(records)
                    metadata["coverage"] = item.payload.get("coverage_summary") or {}
                    metadata["partial_coverage"] = bool(item.payload.get("partial_coverage"))
                    metadata["include_missing_records"] = bool(item.payload.get("include_missing_records"))
                    break
        if action.binding_type == "document_window":
            metadata["window_evidence_refs"] = list(action.evidence_refs)
            metadata["windows"] = [
                {
                    "evidence_ref": item.id,
                    "source_id": item.source_id,
                    "data_form": item.data_form,
                    "processed_slice_ids": item.payload.get("processed_slice_ids") or [],
                    "coverage_summary": item.payload.get("coverage_summary") or {},
                }
                for item in evidence_items
            ]
        alignment = str(arguments.get("alignment") or action.reason or "")
        binding_type = action.binding_type or "structured_source"
        inferred_contract = _infer_semantic_contract(
            state,
            binding_type=str(binding_type),
            source_id=source_id,
            table=table,
            allowed_columns=allowed_columns,
        )
        semantic_contract = {**inferred_contract, **explicit_contract}
        semantic_contract, rejected_contract_fields = _sanitize_semantic_contract(
            state,
            semantic_contract,
            source_id=source_id,
            binding_type=str(binding_type),
            allowed_columns=allowed_columns,
        )
        if semantic_contract:
            metadata["semantic_contract"] = semantic_contract
        if rejected_contract_fields:
            metadata["rejected_semantic_contract_fields"] = rejected_contract_fields
        binding = state.add_binding(
            binding_type=binding_type,
            evidence_refs=action.evidence_refs,
            source_id=source_id,
            table=table,
            field=field,
            allowed_columns=allowed_columns,
            alignment=alignment,
            metadata=metadata,
        )
        if str(binding_type) in COMPUTABLE_BINDING_TYPES:
            allowed_next_tools = ("inspect_relation", "run_verified_compute", "bind")
            recommended_items: list[dict[str, Any]] = [
                {
                    "tool_name": "inspect_relation",
                    "arguments": {"binding_ref": binding.id},
                    "reason": "Inspect verified relation schema before compute.",
                },
            ]
            recommended_next_actions = tuple(recommended_items[:8])
            summary = f"Created verified binding {binding.id} as relation {binding.relation_name}."
        else:
            allowed_next_tools = ("submit_final", "bind", "run_document_agent")
            recommended_next_actions = (
                {
                    "tool_name": "submit_final",
                    "arguments": {
                        "binding_refs": [binding.id],
                        "evidence_refs": list(action.evidence_refs),
                        "answer": {},
                    },
                    "reason": "If this verified evidence fully supports the answer, submit a direct final answer with columns and rows.",
                },
            )
            summary = f"Created verified non-relation binding {binding.id}."
        return state.add_evidence(
            tool_name="bind",
            ok=True,
            summary=summary,
            payload={"binding": binding.to_dict()},
            source_id=source_id,
            allowed_next_tools=allowed_next_tools,
            recommended_next_actions=recommended_next_actions,
        )

    def _dispatch_action(self, state: LoopState, action: ModelAction) -> Evidence:
        if action.kind == "bind":
            return self._apply_bind(state, action)
        if action.kind in {"tool_call", "compute", "final"}:
            return self.registry.dispatch(state, action)
        if action.kind == "blocked":
            state.blocked_reason = action.reason or "blocked"
            return state.add_evidence(
                tool_name="blocked",
                ok=False,
                summary=state.blocked_reason,
                payload={
                    "reason": state.blocked_reason,
                    "evidence_refs": list(action.evidence_refs),
                },
            )
        return state.add_evidence(
            tool_name="invalid_action",
            ok=False,
            summary=f"Unknown action kind: {action.kind}",
            payload={"action": action.to_dict()},
        )

    def _progress_key(self, state: LoopState) -> str:
        effective_evidence_keys = {
            _progress_identity(evidence)
            for evidence in state.evidence.values()
            if evidence.ok
        }
        return (
            f"effective_ev={len(effective_evidence_keys)};"
            f"neg={len(state.negative_scopes)};"
            f"cand={len(state.candidates)};"
            f"bind={len(state.bindings)};"
            f"contract={bool(state.answer_contract)};"
            f"semantic_selection={bool(state.semantic_selection)};"
            f"okcomp={sum(1 for item in state.compute_results.values() if item.ok)};"
            f"final={bool(state.final_answer)}"
        )

    def _progress_reason(self, evidence: Evidence, *, progressed: bool) -> str:
        if not progressed:
            return "duplicate_or_no_effective_state_change"
        if evidence.tool_name == "bind" and evidence.ok:
            return "new_binding"
        if evidence.tool_name == "run_verified_compute" and evidence.ok:
            return "new_compute_result"
        if evidence.tool_name == "submit_final" and evidence.ok:
            return "final_answer_submitted"
        if evidence.tool_name == "declare_answer_contract" and evidence.ok:
            return "answer_contract_declared"
        if evidence.tool_name == "select_semantic_cards" and evidence.ok:
            return "semantic_cards_selected"
        if evidence.negative_scope is not None:
            return "new_negative_scope"
        if evidence.ok and evidence.tool_name not in {
            "declare_answer_contract",
            "select_semantic_cards",
        }:
            return "new_observation"
        return "new_ledger_state"

    def _canonicalize_action(self, state: LoopState, action: ModelAction) -> ModelAction:
        if action.kind != "compute" or not action.sql:
            return action
        sql, changed = _canonicalize_sql_relation_names(state, action.sql)
        if not changed:
            return action
        arguments = dict(action.arguments)
        arguments["original_sql"] = action.sql
        arguments["sql"] = sql
        return ModelAction(
            kind=action.kind,
            reason=action.reason,
            tool_name=action.tool_name,
            arguments=arguments,
            binding_type=action.binding_type,
            evidence_refs=action.evidence_refs,
            source_ref=action.source_ref,
            binding_refs=action.binding_refs,
            sql=sql,
            compute_ref=action.compute_ref,
            answer=action.answer,
            raw=action.raw,
        )

    def _state_summary(self, state: LoopState) -> dict[str, Any]:
        remaining_steps = max(0, self.config.max_steps - state.step_index)
        source_coverage = source_coverage_map(state)
        budget_pressure: dict[str, Any] = {
            "step_index": state.step_index,
            "max_steps": self.config.max_steps,
            "remaining_steps": remaining_steps,
        }
        if remaining_steps <= 3:
            budget_pressure["instruction"] = (
                "The evidence loop is near its step limit. Submit a verified answer, "
                "make one new evidence-producing tool call, or call blocked with cited evidence."
            )
        return {
            "step_budget": budget_pressure,
            "latest_evidence_ids": list(state.evidence)[-8:],
            "negative_scopes": state.negative_scopes[-12:],
            "failed_action_feedback": state.guard_feedback[-8:],
            "candidate_count": len(state.candidates),
            "source_coverage": {
                "source_count": source_coverage["source_count"],
                "status_counts": source_coverage["status_counts"],
                "recent_sources": source_coverage["sources"][-12:],
            },
            "bindings": [binding.to_dict() for binding in state.bindings.values()],
            "answer_contract": answer_contract_view(state),
            "semantic_selection": semantic_selection_view(state),
            "selected_source_candidates": selected_source_candidates(state),
            "direct_final_bindings": [
                binding.to_dict()
                for binding in state.bindings.values()
                if binding.binding_type in {"document_window", "value", "operation", "answer_candidate"}
            ],
            "compute_results": [
                {
                    "id": result.id,
                    "ok": result.ok,
                    "columns": list(result.columns),
                    "row_count": len(result.rows),
                    "binding_refs": list(result.binding_refs),
                    "error": result.error,
                }
                for result in state.compute_results.values()
            ],
            "answer_candidates": answer_candidates(state)[-8:],
            "final_output_contract": final_output_contract(state),
            "final_answer_present": bool(state.final_answer),
            "blocked_reason": state.blocked_reason,
        }

    def run(
        self,
        task: PublicTask,
        *,
        trace_callback: TraceCallback | None = None,
    ) -> AgentRunResult:
        state = LoopState(question=task.question, context_dir=task.context_dir)
        trace = EvidenceTrace()
        answer: AnswerTable | None = None
        failure_reason: str | None = None

        try:
            self._bootstrap(task, state, trace)
            transcript = TranscriptWindow(max_groups=8)
            recovery_hint: dict[str, Any] | None = None
            last_error: str | None = None
            self._emit(task_id=task.task_id, trace=trace, callback=trace_callback, status="running")

            for turn in range(1, self.config.max_steps + 1):
                state.step_index = turn
                messages, context_fragment_ids, context_truncated = self._prompt_messages(
                    state,
                    transcript=transcript,
                    last_error=last_error,
                    recovery_hint=recovery_hint,
                )
                response, tool_calls, raw = self._call_model(messages)
                transcript.add_model_response(response)
                raw["context_fragment_ids"] = context_fragment_ids
                raw["context_truncated"] = context_truncated
                trace.add(
                    action=_model_turn_trace_action(tool_calls),
                    action_input=_model_turn_action_input(tool_calls),
                    thought=_model_response_thought(response),
                    raw_response=raw,
                    observation={
                        "turn": turn,
                        "prompt_message_count": len(messages),
                        "transcript_group_count": len(transcript.groups),
                        "tool_call_count": len(tool_calls),
                        "tool_calls": _compact(tool_calls),
                        "state_summary": self._state_summary(state),
                    },
                    ok=bool(tool_calls),
                )

                if not tool_calls:
                    content = str(getattr(response, "content", "") or "").strip()
                    reason = (
                        "model_returned_text_without_tool_call"
                        if content
                        else "model_returned_no_tool_call"
                    )
                    state.guard_feedback.append(
                        {
                            "turn": turn,
                            "signature": reason,
                            "reason": content[:500] or reason,
                        }
                    )
                    state.repeated_no_progress += 1
                    trace.add(
                        action="codex_no_tool_diagnostic",
                        thought="Model response did not contain native tool calls; stop without post-action recovery.",
                        observation={
                            "turn": turn,
                            "reason": reason,
                            "content_preview": content[:800],
                        },
                        ok=False,
                    )
                    failure_reason = (
                        "Model did not use native tool calling; no post-action recovery was attempted."
                    )
                    break

                recovery_hint = None
                last_error = None

                turn_progressed = False
                terminal_action: ModelAction | None = None
                terminal_evidence: Evidence | None = None
                for call_index, call in enumerate(tool_calls, start=1):
                    before_key = self._progress_key(state)
                    tool_call_id = str(call.get("id") or f"turn_{turn:04d}_call_{call_index:02d}")
                    action = self._canonicalize_action(state, action_from_tool_call(call))
                    invocation = ToolInvocation(
                        tool_name=str(call.get("name") or action.tool_name or action.kind),
                        call_id=tool_call_id,
                        arguments=action.arguments,
                        action=action.to_dict(),
                    )
                    guard = guard_action(state, action, self.registry)
                    if guard.allowed:
                        evidence = self._dispatch_action(state, action)
                    else:
                        state.guard_feedback.append(
                            {
                                "turn": turn,
                                "signature": action.signature(),
                                "action": action.to_dict(),
                                "reason": guard.reason,
                            }
                        )
                        evidence = state.add_evidence(
                            tool_name="guard",
                            ok=False,
                            summary=guard.reason,
                            payload={"action": action.to_dict(), "tool_call": call},
                            negative_scope={
                                "kind": "invalid_tool_action",
                                "signature": action.signature(),
                                "reason": guard.reason,
                            },
                            allowed_next_tools=guard.allowed_next_tools,
                            recommended_next_actions=guard.recommended_next_actions,
                        )
                    if guard.allowed and not evidence.ok:
                        state.guard_feedback.append(
                            {
                                "turn": turn,
                                "signature": action.signature(),
                                "action": action.to_dict(),
                                "reason": evidence.summary,
                            }
                        )

                    after_key = self._progress_key(state)
                    progressed = before_key != after_key
                    progress_reason = self._progress_reason(evidence, progressed=progressed)
                    turn_progressed = turn_progressed or progressed
                    progress = {
                        "progressed": progressed,
                        "progress_reason": progress_reason,
                        "progress_key_before": before_key,
                        "progress_key_after": after_key,
                    }
                    envelope = ToolOutputEnvelope.from_evidence(
                        evidence,
                        guard=guard,
                        progress=progress,
                    )
                    tool_trace_name = _trace_action_name(
                        action, evidence, guard_allowed=guard.allowed
                    )
                    trace.add(
                        action=tool_trace_name,
                        action_input=action.to_dict(),
                        thought="Native tool call was guarded, dispatched, appended to the ledger, and returned as a model-visible ToolMessage.",
                        tool_call_id=tool_call_id,
                        observation={
                            "turn": turn,
                            "tool_call_id": tool_call_id,
                            "tool_invocation": invocation.to_dict(),
                            "native_tool_call": _compact(call),
                            "tool_output_envelope": _compact(envelope),
                            "guard": guard.to_dict(),
                            "evidence_id": evidence.id,
                            "tool_name": evidence.tool_name,
                            "ok": evidence.ok,
                            "summary": evidence.summary,
                            "source_id": evidence.source_id,
                            "candidate_id": evidence.candidate_id,
                            "data_form": evidence.data_form,
                            "payload": _compact(evidence.payload),
                            "progressed": progressed,
                            "progress_reason": progress_reason,
                            "progress_key_before": before_key,
                            "progress_key_after": after_key,
                        },
                        ok=guard.allowed and evidence.ok,
                    )
                    transcript.add_tool_output(
                        ToolMessage(
                            content=tool_output_content(
                                envelope,
                                state_summary=self._state_summary(state),
                            ),
                            tool_call_id=tool_call_id,
                            name=tool_trace_name,
                            status="success" if guard.allowed and evidence.ok else "error",
                        )
                    )
                    if action.kind == "final" and evidence.ok and state.final_answer:
                        terminal_action = action
                        terminal_evidence = evidence
                        break
                    elif action.kind == "blocked":
                        terminal_action = action
                        terminal_evidence = evidence
                        break

                if turn_progressed:
                    state.repeated_no_progress = 0
                else:
                    state.repeated_no_progress += 1
                self._emit(
                    task_id=task.task_id,
                    trace=trace,
                    callback=trace_callback,
                    status="running",
                )

                if (
                    terminal_action is not None
                    and terminal_action.kind == "final"
                    and terminal_evidence is not None
                    and terminal_evidence.ok
                    and state.final_answer
                ):
                    answer = _answer_from_final(state)
                    break
                if terminal_action is not None and terminal_action.kind == "blocked":
                    failure_reason = state.blocked_reason or "Evidence loop blocked."
                    break
                if state.repeated_no_progress >= 2:
                    failure_reason = "Evidence loop stopped after two no-progress turns."
                    break
            else:
                failure_reason = "Evidence loop reached max_steps."

            audit = _audit_final(state)
            if answer is None and state.final_answer:
                answer = _answer_from_final(state)
            if answer is None and failure_reason is None:
                failure_reason = "No final answer was submitted."

            trace.add(
                action="codex_final_audit",
                thought="Record final audit diagnostics without post-hoc blocking.",
                observation={
                    "audit": audit,
                    "answer": answer.to_dict() if answer is not None else None,
                    "failure_reason": failure_reason,
                    "blocked_reason": state.blocked_reason,
                },
                ok=answer is not None and failure_reason is None,
            )
            result = AgentRunResult(
                task_id=task.task_id,
                answer=answer,
                steps=trace.snapshot(),
                failure_reason=failure_reason,
            )
            self._emit(
                task_id=task.task_id,
                trace=trace,
                callback=trace_callback,
                status="completed" if result.succeeded else "failed",
                answer=answer,
                failure_reason=failure_reason,
            )
            return result

        except Exception as exc:  # noqa: BLE001 - preserve benchmark trace on unexpected failures
            trace.add(
                action="codex_agent_error",
                observation={"error": str(exc), "type": type(exc).__name__},
                ok=False,
            )
            result = AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=trace.snapshot(),
                failure_reason=f"Codex evidence loop failed with {type(exc).__name__}: {exc}",
            )
            self._emit(
                task_id=task.task_id,
                trace=trace,
                callback=trace_callback,
                status="failed",
                failure_reason=result.failure_reason,
            )
            return result
