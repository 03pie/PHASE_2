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
    RecoveryHint,
    ToolInvocation,
    ToolOutputEnvelope,
    TranscriptWindow,
)
from data_agent_baseline.evidence_agent.codex_loop.registry import EvidenceActionRegistry
from data_agent_baseline.evidence_agent.codex_loop.state_views import (
    answer_candidates,
    blocked_audit,
    final_output_contract,
    recovery_hints,
    requirement_coverage,
    source_coverage_map,
    verifier_decisions,
)
from data_agent_baseline.evidence_agent.knowledge import (
    build_knowledge_sections,
    match_knowledge_sections,
)
from data_agent_baseline.evidence_agent.tracing import EvidenceTrace

_SQL_RELATION_PATTERN = re.compile(
    r'\b(from|join)\s+("([^"]+)"|[A-Za-z_][\w]*)',
    re.IGNORECASE,
)


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
        "start_line",
        "end_line",
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
        identity["window_scope"] = [
            {
                "start_line": window.get("start_line"),
                "end_line": window.get("end_line"),
                "page_start": window.get("page_start"),
                "page_end": window.get("page_end"),
            }
            for window in windows[:8]
            if isinstance(window, dict)
        ]
    if isinstance(payload.get("records"), list):
        identity["record_count"] = len(payload["records"])
    return json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str)


def _trace_action_name(action: ModelAction, evidence: Evidence, *, guard_allowed: bool) -> str:
    if not guard_allowed:
        return "guard_block"
    if action.kind == "tool_call":
        return action.tool_name or evidence.tool_name
    if action.kind == "compute":
        return "run_verified_compute"
    if action.kind == "final":
        return "submit_final"
    return str(action.kind)


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
    requirements = requirement_coverage(state)
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
        direct_types = {"document_window", "value", "operation", "answer_candidate"}
        if binding_refs and not any(
            (state.bindings.get(ref) is not None and state.bindings[ref].binding_type in direct_types)
            for ref in binding_refs
        ):
            unsupported_operations.append("direct_final_without_direct_binding")

    more_missing, more_weak, more_conflicts = _audit_binding_refs(
        state, binding_refs, evidence_refs
    )
    missing_requirements.extend(more_missing)
    if requirements["declared_count"]:
        for item in requirements["pending_requirements"]:
            warnings.append(f"pending_requirement:{item['id']}")
        for item in requirements["blocked_requirements"]:
            warnings.append(f"blocked_requirement:{item['id']}")
        for item in requirements["weak_satisfied_requirements"]:
            weak_bindings.append(f"weak_requirement_lineage:{item['id']}")
        for item in requirements["conflict_requirements"]:
            conflicts.append(f"conflict_requirement:{item['id']}")
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
        "requirement_coverage": requirements,
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
        self.tool_model = bind_native_tools(model)
        self.config = config
        self.registry = EvidenceActionRegistry()

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

        sections, schema_json, content_hash = build_knowledge_sections(task.context_dir)
        state.knowledge_sections = sections
        state.matched_sections = match_knowledge_sections(task.question, sections)
        forbidden_terms = ("profile", "physical_schema", "table_schema", "field_schema")
        trace.add(
            action="codex_bootstrap_knowledge",
            thought="Knowledge remains document-only; it can guide semantics but cannot bind data.",
            observation={
                "content_hash": content_hash,
                "section_count": len(sections),
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
                "schema_contract_ok": all(term not in schema_json for term in forbidden_terms),
            },
        )

    def _prompt_messages(
        self,
        state: LoopState,
        *,
        transcript: TranscriptWindow | None = None,
        last_error: str | None = None,
        recovery_hint: RecoveryHint | None = None,
        extra_instruction: str | None = None,
    ) -> tuple[list[Any], list[str], dict[str, bool]]:
        fragments = build_context_fragments(
            state,
            last_error=last_error,
            recovery_hint=recovery_hint.to_dict() if recovery_hint else None,
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
        tool_model = (
            bind_native_tools(
                self.model,
                tool_choice=tool_choice,
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
                if attempt == 0 and isinstance(tool_choice, dict):
                    tool_choice = "required"
                    tool_model = bind_native_tools(
                        self.model,
                        tool_choice=tool_choice,
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
            "response_metadata": getattr(response, "response_metadata", {}),
            "tool_choice": tool_choice,
            "tool_names": list(tool_names) if tool_names else None,
        }
        return response, tool_calls, raw

    def _forced_recovery_policy(
        self,
        state: LoopState,
        *,
        attempt: int,
    ) -> tuple[str | dict[str, Any], tuple[str, ...] | None, RecoveryHint | None]:
        del attempt
        hints = recovery_hints(state, limit=1)
        hint = hints[0] if hints else None
        if hint is None:
            return "required", None, None
        tool_names = tuple(dict.fromkeys((hint.tool_name, "blocked")))
        tool_choice: str | dict[str, Any]
        if hint.tool_name and hint.tool_name != "blocked":
            tool_choice = {
                "type": "function",
                "function": {"name": hint.tool_name},
            }
        else:
            tool_choice = "required"
        return tool_choice, tool_names, hint

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
        if action.binding_type == "document_record_set":
            for item in evidence_items:
                records = item.payload.get("records")
                if isinstance(records, list):
                    metadata["records"] = records
                    metadata["record_count"] = len(records)
                    metadata["coverage"] = item.payload.get("coverage") or []
                    metadata["partial_coverage"] = bool(item.payload.get("partial_coverage"))
                    if not allowed_columns and records and isinstance(records[0], dict):
                        allowed_columns = tuple(str(column) for column in records[0] if column != "provenance")
                    break
        if action.binding_type == "document_window":
            metadata["window_evidence_refs"] = list(action.evidence_refs)
            metadata["windows"] = [
                {
                    "evidence_ref": item.id,
                    "source_id": item.source_id,
                    "data_form": item.data_form,
                    "start_line": item.payload.get("start_line"),
                    "end_line": item.payload.get("end_line"),
                    "query": item.payload.get("query"),
                }
                for item in evidence_items
            ]
        alignment = str(arguments.get("alignment") or action.reason or "")
        binding_type = action.binding_type or "structured_source"
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
            if (
                action.binding_type == "document_record_set"
                and metadata.get("partial_coverage")
            ):
                allowed_next_tools = (
                    "search_document",
                    "read_document_window",
                    "extract_records",
                    "bind",
                    "inspect_relation",
                )
                for coverage in metadata.get("coverage") or []:
                    for window in coverage.get("unreturned_matches") or []:
                        if isinstance(window, dict) and isinstance(window.get("recommended_window"), dict):
                            recommended_items.insert(
                                0,
                                {
                                    "tool_name": "read_document_window",
                                    "arguments": window["recommended_window"],
                                    "reason": "This document record-set was extracted from partial search coverage; read the next uncovered matching window before treating it as complete.",
                                },
                            )
                            break
                    if recommended_items[0]["tool_name"] == "read_document_window":
                        break
            recommended_next_actions = tuple(recommended_items[:8])
            summary = f"Created verified binding {binding.id} as relation {binding.relation_name}."
        else:
            allowed_next_tools = ("submit_final", "bind", "search_document", "read_document_window")
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
        if evidence.negative_scope is not None:
            return "new_negative_scope"
        if evidence.ok and evidence.tool_name not in {"verify_alignment", "track_requirements"}:
            return "new_observation"
        if evidence.ok and evidence.tool_name == "verify_alignment":
            return "new_verifier_decision"
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
        hints = recovery_hints(state, limit=1)
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
            "requirements": requirement_coverage(state),
            "verifier_decisions": verifier_decisions(state)[-8:],
            "recovery_hint": hints[0].to_dict() if hints else None,
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

    def _blocked_needs_repair(self, state: LoopState, audit: dict[str, Any]) -> bool:
        if audit["passed"]:
            return False
        signature = self._blocked_repair_signature(audit)
        if signature is None:
            return False
        previous = sum(
            1 for item in state.guard_feedback if item.get("signature") == signature
        )
        return previous < 1

    def _blocked_repair_signature(self, audit: dict[str, Any]) -> str | None:
        for item in audit.get("recovery_hints") or []:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool_name") or "")
            if not tool_name or tool_name == "blocked":
                continue
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            return "blocked_audit_repair:" + json.dumps(
                {"tool_name": tool_name, "arguments": arguments},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        return None

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
            recovery_hint: RecoveryHint | None = None
            last_error: str | None = None
            self._emit(task_id=task.task_id, trace=trace, callback=trace_callback, status="running")

            forced_tool_choice: str | dict[str, Any] | None = None
            forced_tool_names: tuple[str, ...] | None = None
            forced_recovery_attempts = 0
            for turn in range(1, self.config.max_steps + 1):
                state.step_index = turn
                messages, context_fragment_ids, context_truncated = self._prompt_messages(
                    state,
                    transcript=transcript,
                    last_error=last_error,
                    recovery_hint=recovery_hint,
                )
                response, tool_calls, raw = self._call_model(
                    messages,
                    tool_choice=forced_tool_choice,
                    tool_names=forced_tool_names,
                )
                transcript.add_model_response(response)
                raw["context_fragment_ids"] = context_fragment_ids
                raw["context_truncated"] = context_truncated
                trace.add(
                    action="codex_turn",
                    thought="Model sampling returned native tool calls.",
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
                    forced_recovery_attempts += 1
                    forced_tool_choice, forced_tool_names, recovery_hint = self._forced_recovery_policy(
                        state,
                        attempt=forced_recovery_attempts,
                    )
                    hint_dict = recovery_hint.to_dict() if recovery_hint else None
                    state.guard_feedback.append(
                        {
                            "turn": turn,
                            "signature": reason,
                            "reason": content[:500] or reason,
                            "recovery_hint": hint_dict,
                            "forced_tool_choice_next": forced_tool_choice,
                            "forced_tool_names_next": list(forced_tool_names or []),
                        }
                    )
                    state.repeated_no_progress += 1
                    trace.add(
                        action="codex_no_tool_repair",
                        thought="Model response did not contain native tool calls; feed back a bounded repair instruction.",
                        observation={
                            "turn": turn,
                            "reason": reason,
                            "content_preview": content[:800],
                            "recovery_hint": hint_dict,
                            "forced_recovery_attempts": forced_recovery_attempts,
                            "forced_tool_choice_next": forced_tool_choice,
                            "forced_tool_names_next": list(forced_tool_names or []),
                        },
                        ok=False,
                    )
                    if forced_recovery_attempts >= 4:
                        failure_reason = (
                            "Model did not use native tool calling after forced recovery attempts."
                        )
                        break
                    last_error = reason
                    transcript.add_repair_message(
                        HumanMessage(
                            content=(
                                "Your previous response was ignored because it did not contain a native tool call. "
                                "The next request is forced to use native tool calling. "
                                "Return exactly one native tool call. Do not answer in text. "
                                "Use observed evidence and compact state, or call blocked with cited evidence_refs."
                            )
                        )
                    )
                    continue

                forced_tool_choice = None
                forced_tool_names = None
                forced_recovery_attempts = 0
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
                                "kind": "guard_block",
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
                        audit_now = _audit_final(state)
                        if audit_now["passed"]:
                            terminal_action = action
                            terminal_evidence = evidence
                            break
                        state.guard_feedback.append(
                            {
                                "turn": turn,
                                "signature": "final_audit",
                                "action": action.to_dict(),
                                "reason": "Final audit failed before termination: "
                                + ", ".join(audit_now["issues"]),
                                "audit": audit_now,
                            }
                        )
                        trace.add(
                            action="codex_final_audit_repair",
                            thought="Final audit failed; keep the candidate final in state and feed audit gaps back for repair.",
                            observation={"audit": audit_now},
                            ok=False,
                        )
                        transcript.add_repair_message(
                            HumanMessage(
                                content=(
                                    "Final audit failed, but the candidate final answer remains available in state. "
                                    "Continue with native tool calls only. Repair projection, lineage, requirements, "
                                    "or submit a corrected final; call blocked only if no valid repair remains:\n"
                                    + str(_compact(audit_now, limit=20))
                                )
                            )
                        )
                        last_error = "final_audit_failed"
                    elif action.kind == "blocked":
                        audit_now = blocked_audit(
                            state,
                            action.reason or evidence.summary,
                            cited_evidence_refs=action.evidence_refs,
                        )
                        if self._blocked_needs_repair(state, audit_now):
                            signature = self._blocked_repair_signature(audit_now)
                            state.guard_feedback.append(
                                {
                                    "turn": turn,
                                    "signature": signature,
                                    "action": action.to_dict(),
                                    "reason": "Blocked audit found unresolved answer candidates or positive evidence.",
                                    "audit": audit_now,
                                }
                            )
                            state.blocked_reason = None
                            trace.add(
                                action="codex_blocked_audit_repair",
                                thought="Blocked was rejected once because unresolved answer candidates or positive evidence remain.",
                                observation={"audit": audit_now},
                                ok=False,
                            )
                            transcript.add_repair_message(
                                HumanMessage(
                                    content=(
                                        "Blocked audit found unresolved candidates/evidence. "
                                        "Continue with native tool calls only. Submit a sufficient candidate, "
                                        "bind/use the positive evidence, gather specific missing evidence, "
                                        "or call blocked again with a reason that addresses this audit:\n"
                                        + str(_compact(audit_now, limit=20))
                                    )
                                )
                            )
                            last_error = "blocked_audit_failed"
                        else:
                            terminal_action = action
                            terminal_evidence = evidence
                            break

                if turn_progressed:
                    state.repeated_no_progress = 0
                else:
                    state.repeated_no_progress += 1
                if terminal_action is None:
                    trace.add(
                        action="codex_context_refresh",
                        thought="Refresh compact state fragments while preserving the recent AI/tool transcript window.",
                        observation={
                            "turn": turn,
                            "transcript_group_count": len(transcript.groups),
                            "state_summary": self._state_summary(state),
                        },
                    )
                    if state.repeated_no_progress == 1:
                        forced_tool_choice, forced_tool_names, recovery_hint = self._forced_recovery_policy(
                            state,
                            attempt=1,
                        )
                        hint_dict = recovery_hint.to_dict() if recovery_hint else None
                        state.guard_feedback.append(
                            {
                                "turn": turn,
                                "signature": "no_progress_repair",
                                "reason": "Last native tool turn did not add effective progress.",
                                "recovery_hint": hint_dict,
                                "forced_tool_choice_next": forced_tool_choice,
                                "forced_tool_names_next": list(forced_tool_names or []),
                            }
                        )
                        trace.add(
                            action="codex_no_progress_repair",
                            thought="A native tool turn made no effective progress; return a recoverable tool-loop repair to the model.",
                            observation={
                                "turn": turn,
                                "recovery_hint": hint_dict,
                                "forced_tool_choice_next": forced_tool_choice,
                                "forced_tool_names_next": list(forced_tool_names or []),
                            },
                            ok=False,
                        )
                        transcript.add_repair_message(
                            HumanMessage(
                                content=(
                                    "The previous native tool turn made no effective progress. "
                                    "Return exactly one native tool call that uses new evidence, submits a verified answer, or calls blocked."
                                )
                            )
                        )
                        last_error = "no_progress_repair"
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
            if answer is not None and not audit["passed"]:
                failure_reason = "Final audit failed: " + ", ".join(audit["issues"])
                answer = None

            trace.add(
                action="codex_final_audit",
                thought="Final answer must come from verified compute or direct evidence lineage.",
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
