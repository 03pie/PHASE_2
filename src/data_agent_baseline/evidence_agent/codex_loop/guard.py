from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.evidence_agent.codex_loop.protocol import (
    COMPUTABLE_BINDING_TYPES,
    GuardDecision,
    LoopState,
    ModelAction,
)
from data_agent_baseline.evidence_agent.codex_loop.registry import EvidenceActionRegistry
from data_agent_baseline.evidence_agent.codex_loop.lineage import (
    DIRECT_FINAL_BINDING_TYPES,
    META_EVIDENCE_TOOLS,
)

_STRUCTURED_FORMS = {"sqlite_database", "csv_records", "json_records"}
_DOCUMENT_FORMS = {"pdf_document", "markdown_document"}
_BINDING_TYPES = COMPUTABLE_BINDING_TYPES | set(DIRECT_FINAL_BINDING_TYPES)
_RELATION_PATTERN = re.compile(r'\b(?:from|join)\s+("([^"]+)"|[A-Za-z_][\w]*)', re.IGNORECASE)
_QUOTED_IDENTIFIER_PATTERN = re.compile(r'"([^"]+)"')


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _source_exists(state: LoopState, source_ref: str | None) -> bool:
    return bool(source_ref and source_ref in state.sources)


def _evidence_exists(state: LoopState, evidence_refs: tuple[str, ...]) -> bool:
    return bool(evidence_refs) and all(ref in state.evidence for ref in evidence_refs)


def _path_is_observed(state: LoopState, path: str | None) -> bool:
    if not path:
        return True
    return path in state.source_by_path


def _relation_by_name(state: LoopState) -> dict[str, object]:
    return {
        binding.relation_name: binding
        for binding in state.bindings.values()
        if binding.relation_name
    }


def _inspect_relation_recommendation() -> tuple[dict[str, object], ...]:
    return (
        {
            "tool_name": "inspect_relation",
            "arguments": {},
            "reason": "Inspect verified relation names, columns, and samples before retrying compute.",
        },
    )


def _document_focus_rerun_recommendation(state: LoopState, evidence_items: list[Any]) -> tuple[dict[str, object], ...]:
    for evidence in evidence_items:
        if evidence.tool_name != "run_document_agent":
            continue
        payload = evidence.payload if isinstance(evidence.payload, dict) else {}
        uncertain = payload.get("uncertain_slices") if isinstance(payload.get("uncertain_slices"), list) else []
        slice_ids = [
            str(item.get("slice_id") or "").strip()
            for item in uncertain
            if isinstance(item, dict) and str(item.get("slice_id") or "").strip()
        ][:24]
        doc_task = payload.get("doc_task") if isinstance(payload.get("doc_task"), dict) else {}
        coverage_policy = dict(doc_task.get("coverage_policy") or {})
        if slice_ids:
            coverage_policy["focus_slice_ids"] = slice_ids
            coverage_policy["scan_batch_size"] = min(max(len(slice_ids), 1), 24)
        source_candidates = (
            doc_task.get("source_candidates")
            if isinstance(doc_task.get("source_candidates"), list)
            else payload.get("source_refs")
        )
        target_fields = doc_task.get("target_fields") if isinstance(doc_task.get("target_fields"), list) else []
        semantic_cards = doc_task.get("semantic_cards") if isinstance(doc_task.get("semantic_cards"), list) else []
        arguments = {
            "question": doc_task.get("question") or state.question,
            "source_candidates": source_candidates or ([evidence.source_id] if evidence.source_id else []),
            "target_fields": target_fields,
            "semantic_cards": semantic_cards,
            "required_record_grain": doc_task.get("required_record_grain") or "",
            "coverage_policy": coverage_policy,
        }
        return (
            {
                "tool_name": "run_document_agent",
                "arguments": arguments,
                "reason": (
                    "Resolve the unresolved document slices before binding; use focus_slice_ids "
                    "so DocumentAgent reads only the recorded uncertain slices."
                ),
            },
        )
    return (
        {
            "tool_name": "run_document_agent",
            "arguments": {"question": state.question},
            "reason": "Rerun DocumentAgent before binding partial document evidence.",
        },
    )


def _semantic_field_variants(field_id: str) -> set[str]:
    text = str(field_id or "").casefold()
    variants = {text}
    if "." in text:
        variants.add(text.rsplit(".", 1)[-1])
    return {item for item in variants if item}


def _physical_columns_from_mapping(value: Any) -> list[str]:
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


def _binding_contract_physical_mismatches(binding: Any) -> list[str]:
    metadata = getattr(binding, "metadata", {}) or {}
    contract = metadata.get("semantic_contract")
    if not isinstance(contract, dict):
        return []
    allowed_columns = {_normalize(column) for column in getattr(binding, "allowed_columns", ())}
    if not allowed_columns:
        return []
    mismatches: list[str] = []
    raw_mapping = contract.get("physical_field_mapping")
    mapped_fields: set[str] = set()
    if isinstance(raw_mapping, dict):
        for key, value in raw_mapping.items():
            field_id = str(key).casefold()
            mapped_fields.add(field_id)
            physical_columns = _physical_columns_from_mapping(value)
            if not physical_columns or not any(_normalize(column) in allowed_columns for column in physical_columns):
                mismatches.append(field_id)
    raw = contract.get("canonical_fields")
    if isinstance(raw, list):
        for item in raw:
            field_id = str(item).casefold()
            if field_id in mapped_fields:
                continue
            variants = _semantic_field_variants(field_id)
            if not any(_normalize(variant) in allowed_columns for variant in variants):
                mismatches.append(field_id)
    return sorted(set(mismatches))


def _binding_contract_mismatches(state: LoopState, binding_refs: tuple[str, ...]) -> list[str]:
    problems: list[str] = []
    for ref in binding_refs:
        binding = state.bindings.get(ref)
        if binding is None:
            continue
        for field_id in _binding_contract_physical_mismatches(binding):
            problems.append(f"{ref}:{field_id}")
    return sorted(set(problems))


def _compute_final_answer_columns(action: ModelAction) -> list[str]:
    answer = action.answer if isinstance(action.answer, dict) else action.arguments.get("answer")
    if not isinstance(answer, dict):
        return []
    columns = answer.get("columns")
    if not isinstance(columns, list):
        return []
    return [str(column) for column in columns if str(column).strip()]


def _payload_table(payload: dict[str, Any]) -> str:
    return str(payload.get("table") or "").strip()


def _action_source_id(state: LoopState, action: ModelAction) -> str:
    path = action.arguments.get("path")
    source_ref = action.arguments.get("source_ref") or action.arguments.get("source_id")
    return str(source_ref or state.source_by_path.get(str(path)) or "").strip()


def _successful_observation(
    state: LoopState,
    *,
    tool_name: str,
    source_id: str | None = None,
    table: str | None = None,
    binding_ref: str | None = None,
    relation_name: str | None = None,
    query: str | None = None,
) -> object | None:
    for evidence in reversed(list(state.evidence.values())):
        if not evidence.ok or evidence.tool_name != tool_name:
            continue
        payload = evidence.payload or {}
        if source_id is not None and evidence.source_id != source_id:
            continue
        if table is not None and _payload_table(payload) != table:
            continue
        if binding_ref is not None and str(payload.get("binding_ref") or "") != binding_ref:
            continue
        if relation_name is not None and str(payload.get("relation_name") or "") != relation_name:
            continue
        if query is not None and str(payload.get("query") or "").casefold() != query.casefold():
            continue
        return evidence
    return None


def _noop_observation_guard(
    state: LoopState,
    *,
    reason: str,
    allowed_next_tools: tuple[str, ...],
    preferred_tools: tuple[str, ...] = (),
    fallback_actions: tuple[dict[str, object], ...] = (),
) -> GuardDecision:
    del state
    recommendations = tuple(
        dict.fromkeys(
            [repr(item) for item in fallback_actions]
        )
    )
    recommendation_items: list[dict[str, object]] = []
    for item_repr in recommendations:
        for item in fallback_actions:
            if repr(item) == item_repr:
                recommendation_items.append(item)
                break
    return GuardDecision(
        False,
        reason,
        allowed_next_tools=tuple(dict.fromkeys((*preferred_tools, *allowed_next_tools, "blocked"))),
        recommended_next_actions=tuple(recommendation_items),
    )


def _guard_repeated_observation(state: LoopState, action: ModelAction) -> GuardDecision | None:
    tool_name = str(action.tool_name or "")
    source_id = _action_source_id(state, action)
    table = str(action.arguments.get("table") or "").strip()

    if tool_name == "inspect_source" and source_id:
        existing = _successful_observation(
            state,
            tool_name="inspect_source",
            source_id=source_id,
            table=table,
        )
        if existing is not None:
            source = state.sources.get(source_id)
            if source is not None and source.data_form in _DOCUMENT_FORMS:
                preferred = ("run_document_agent", "bind")
                allowed = ("run_document_agent", "bind")
                fallback = (
                    {
                        "tool_name": "run_document_agent",
                        "arguments": {"question": state.question, "source_candidates": [source_id]},
                        "reason": "The source form is already observed; delegate document evidence work to DocumentAgent.",
                    },
                )
            else:
                preferred = ("bind", "sample_records", "search_values", "inspect_relation", "run_verified_compute")
                allowed = ("bind", "sample_records", "search_values", "inspect_relation", "run_verified_compute")
                fallback = (
                    {
                        "tool_name": "bind",
                        "arguments": {"binding_type": "structured_source", "evidence_refs": [existing.id]},
                        "reason": "Consume the existing schema observation as a binding if it supports the task.",
                    },
                )
            return _noop_observation_guard(
                state,
                reason="This source/table schema has already been observed successfully; repeat inspection would not add evidence.",
                allowed_next_tools=allowed,
                preferred_tools=preferred,
                fallback_actions=fallback,
            )

    if tool_name == "sample_records" and source_id:
        existing = _successful_observation(
            state,
            tool_name="sample_records",
            source_id=source_id,
            table=table,
        )
        if existing is not None:
            return _noop_observation_guard(
                state,
                reason="This source/table sample has already been observed successfully; repeat sampling would not add evidence.",
                allowed_next_tools=("bind", "search_values", "inspect_relation", "run_verified_compute"),
                preferred_tools=("bind", "search_values", "inspect_relation", "run_verified_compute"),
                fallback_actions=(
                    {
                        "tool_name": "bind",
                        "arguments": {"binding_type": "structured_source", "evidence_refs": [existing.id]},
                        "reason": "Bind or reject the existing sample/schema evidence instead of sampling again.",
                    },
                ),
            )

    if tool_name == "inspect_relation":
        binding_ref = str(action.arguments.get("binding_ref") or "").strip()
        relation_name = str(action.arguments.get("relation_name") or "").strip()
        existing = _successful_observation(
            state,
            tool_name="inspect_relation",
            binding_ref=binding_ref or None,
            relation_name=relation_name or None,
        )
        if existing is not None:
            target_binding = binding_ref or str((existing.payload or {}).get("binding_ref") or "")
            target_relation = relation_name or str((existing.payload or {}).get("relation_name") or "")
            fallback = (
                {
                    "tool_name": "run_verified_compute",
                    "arguments": {"binding_refs": [target_binding]},
                    "reason": (
                        "Use the already inspected relation for a targeted verified compute. "
                        f"Write SQL over {target_relation} using the inspected columns."
                    ),
                },
            ) if target_binding and target_relation else _inspect_relation_recommendation()
            return _noop_observation_guard(
                state,
                reason="This verified relation has already been inspected successfully; repeat relation inspection would not add evidence.",
                allowed_next_tools=("run_verified_compute", "submit_final", "bind", "blocked"),
                preferred_tools=("run_verified_compute", "submit_final", "bind"),
                fallback_actions=fallback,
            )

    return None


def _equivalent_structured_binding(
    state: LoopState,
    *,
    source_id: str | None,
    table: str | None,
) -> object | None:
    if not source_id:
        return None
    source = state.sources.get(source_id)
    table_name = str(table or "").strip()
    for binding in state.bindings.values():
        if binding.binding_type not in {"structured_source", "structured_field"}:
            continue
        if binding.source_id != source_id:
            continue
        if source is not None and source.data_form in {"csv_records", "json_records"}:
            return binding
        bound_table = str(binding.table or "").strip()
        if table_name and bound_table == table_name:
            return binding
        if not table_name and not bound_table:
            return binding
    return None


def _check_sql_preflight(state: LoopState, action: ModelAction) -> GuardDecision | None:
    sql = action.sql or ""
    if "`" in sql:
        return GuardDecision(
            False,
            "DuckDB SQL must use double quotes for quoted identifiers; backticks are not allowed.",
            allowed_next_tools=("inspect_relation",),
            recommended_next_actions=_inspect_relation_recommendation(),
        )
    relations = _relation_by_name(state)
    relation_names = {name for name in relations if name}
    used_relations: set[str] = set()
    for match in _RELATION_PATTERN.finditer(sql):
        relation = match.group(2) or match.group(1)
        relation = relation.strip('"')
        used_relations.add(relation)
    if not used_relations:
        return GuardDecision(
            False,
            "Compute SQL must reference at least one verified relation in FROM/JOIN.",
            allowed_next_tools=("inspect_relation",),
            recommended_next_actions=_inspect_relation_recommendation(),
        )
    unknown_relations = sorted(used_relations - relation_names)
    if unknown_relations:
        return GuardDecision(
            False,
            "Compute SQL references unverified relation/table names: "
            + ", ".join(unknown_relations),
            allowed_next_tools=("inspect_relation",),
            recommended_next_actions=_inspect_relation_recommendation(),
        )
    available_columns: set[str] = set()
    for relation in used_relations:
        binding = relations.get(relation)
        if binding is not None:
            available_columns.update(str(column) for column in binding.allowed_columns)
    quoted = {identifier for identifier in _QUOTED_IDENTIFIER_PATTERN.findall(sql)}
    quoted_columns = quoted - relation_names
    if available_columns:
        unknown_columns = sorted(
            column for column in quoted_columns if column not in available_columns
        )
        if unknown_columns:
            return GuardDecision(
                False,
                "Compute SQL references columns not present in verified relation schemas: "
                + ", ".join(unknown_columns),
                allowed_next_tools=("inspect_relation",),
                recommended_next_actions=_inspect_relation_recommendation(),
            )
    return None


def guard_action(
    state: LoopState,
    action: ModelAction,
    registry: EvidenceActionRegistry,
) -> GuardDecision:
    if action.kind not in {"tool_call", "bind", "compute", "final", "blocked"}:
        return GuardDecision(False, f"Unknown action kind: {action.kind}")

    if action.kind == "blocked":
        return GuardDecision(True, "Model reported blocked/conflict.")

    if action.kind == "tool_call":
        if not action.tool_name or registry.spec(action.tool_name) is None:
            return GuardDecision(False, f"Unknown tool: {action.tool_name}")
        if action.tool_name in {"run_verified_compute", "submit_final"}:
            return GuardDecision(False, f"Use action kind for {action.tool_name}, not tool_call.")
        path = action.arguments.get("path")
        source_ref = action.arguments.get("source_ref") or action.arguments.get("source_id")
        source_ref_valid = _source_exists(state, str(source_ref)) if source_ref else False
        if path and not source_ref_valid and not _path_is_observed(state, str(path)):
            return GuardDecision(False, "Tool references a path that is not in observed inventory.")
        if source_ref and not source_ref_valid:
            return GuardDecision(False, "Tool references an unknown source_ref.")
        if action.tool_name == "inspect_relation":
            relation_name = str(action.arguments.get("relation_name") or "").strip()
            binding_ref = str(action.arguments.get("binding_ref") or "").strip()
            relations = _relation_by_name(state)
            if relation_name and relation_name not in relations:
                return GuardDecision(False, f"Unknown verified relation: {relation_name}")
            if binding_ref and binding_ref not in state.bindings:
                return GuardDecision(False, f"Unknown binding_ref: {binding_ref}")
        return GuardDecision(True, "Tool call allowed.")

    if action.kind == "bind":
        if str(action.binding_type or "") not in _BINDING_TYPES:
            return GuardDecision(False, f"Unknown binding_type: {action.binding_type}")
        if not _evidence_exists(state, action.evidence_refs):
            return GuardDecision(False, "Binding must cite existing evidence_refs.")
        if action.source_ref and action.source_ref not in state.sources:
            return GuardDecision(False, "Binding references an unknown source_ref.")
        evidence_items = [state.evidence[ref] for ref in action.evidence_refs]
        if not all(item.ok for item in evidence_items):
            return GuardDecision(False, "Binding cannot cite failed evidence.")
        if not any(item.tool_name not in META_EVIDENCE_TOOLS for item in evidence_items):
            return GuardDecision(
                False,
                "Binding requires at least one successful observed evidence item, not only ledger evidence.",
                allowed_next_tools=(
                    "inspect_source",
                    "sample_records",
                    "search_values",
                    "run_document_agent",
                ),
            )
        if action.binding_type in {"structured_source", "structured_field"}:
            source_id = action.source_ref or evidence_items[0].source_id
            source = state.sources.get(source_id or "")
            if source is None:
                return GuardDecision(False, "Structured binding requires an observed source.")
            if source.data_form not in _STRUCTURED_FORMS:
                return GuardDecision(False, f"{source.data_form} cannot be bound as structured data.")
            if action.binding_type == "structured_field" and not (
                action.arguments.get("field") or action.arguments.get("allowed_columns")
            ):
                return GuardDecision(False, "Structured field binding requires a field or allowed_columns.")
        if action.binding_type == "document_record_set":
            has_records = any(isinstance(item.payload.get("records"), list) for item in evidence_items)
            if not has_records:
                return GuardDecision(False, "Document record-set binding requires extracted record evidence.")
        if action.binding_type == "document_window":
            has_document_package = any(item.tool_name == "run_document_agent" for item in evidence_items)
            if not has_document_package:
                return GuardDecision(
                    False,
                    "Document-window binding requires successful DocumentAgent evidence.",
                    allowed_next_tools=("run_document_agent",),
                )
        if any(item.data_form == "video" for item in evidence_items):
            return GuardDecision(False, "V1 video evidence cannot become a final binding.")
        return GuardDecision(True, "Binding allowed from observed evidence.")

    if action.kind == "compute":
        if not action.sql:
            return GuardDecision(False, "Compute requires SQL.")
        refs = action.binding_refs
        if not refs:
            return GuardDecision(False, "Compute requires verified bindings.")
        if not all(ref in state.bindings for ref in refs):
            return GuardDecision(False, "Compute references unknown binding_refs.")
        non_relational = [ref for ref in refs if not state.bindings[ref].relation_name]
        if non_relational:
            return GuardDecision(
                False,
                "Compute can only use bindings with verified relation names: " + ", ".join(non_relational),
                allowed_next_tools=("inspect_relation", "run_verified_compute", "submit_final", "blocked"),
                recommended_next_actions=(
                    {
                        "tool_name": "submit_final",
                        "arguments": {
                            "binding_refs": non_relational,
                            "evidence_refs": [],
                            "answer": {},
                        },
                        "reason": "If direct evidence fully supports the answer, submit a direct final with answer, binding_refs, and evidence_refs instead of SQL.",
                    },
                ),
            )
        contract_mismatches = _binding_contract_mismatches(state, refs)
        if contract_mismatches:
            return GuardDecision(
                False,
                "Binding semantic_contract physical_field_mapping references fields not present in observed binding columns: "
                + ", ".join(contract_mismatches),
            )
        preflight = _check_sql_preflight(state, action)
        if preflight is not None:
            return preflight
        return GuardDecision(True, "Compute allowed over verified bindings.")

    if action.kind == "final":
        if action.compute_ref:
            if action.compute_ref not in state.compute_results:
                return GuardDecision(False, "Final references an unknown compute_ref.")
            compute = state.compute_results[action.compute_ref]
            if not compute.ok:
                return GuardDecision(False, "Final cannot use a failed compute result.")
            if not compute.binding_refs:
                return GuardDecision(False, "Final compute result has no binding lineage.")
            if not compute.rows:
                return GuardDecision(False, "Final compute result has no output rows.")
            final_columns = _compute_final_answer_columns(action)
            if not final_columns:
                return GuardDecision(False, "Compute-backed final requires an explicit answer.columns projection.")
            unknown_final_columns = [
                column for column in final_columns
                if column not in {str(item) for item in compute.columns}
            ]
            if unknown_final_columns:
                return GuardDecision(
                    False,
                    "Final answer.columns must be a subset of the compute result columns: "
                    + ", ".join(unknown_final_columns),
                )
            return GuardDecision(True, "Final allowed from verified and explicitly projected compute result.")

        answer = action.answer or action.arguments.get("answer")
        if not isinstance(answer, dict) or not answer:
            return GuardDecision(False, "Direct final requires an answer object or a compute_ref.")
        if not action.binding_refs:
            return GuardDecision(False, "Direct final requires verified binding_refs.")
        if not action.evidence_refs:
            return GuardDecision(False, "Direct final requires successful evidence_refs.")
        if not all(ref in state.bindings for ref in action.binding_refs):
            return GuardDecision(False, "Direct final references unknown binding_refs.")
        if not all(ref in state.evidence for ref in action.evidence_refs):
            return GuardDecision(False, "Direct final references unknown evidence_refs.")
        evidence_items = [state.evidence[ref] for ref in action.evidence_refs]
        if not all(item.ok for item in evidence_items):
            return GuardDecision(False, "Direct final cannot cite failed evidence.")
        if any(item.data_form == "video" for item in evidence_items):
            return GuardDecision(False, "V1 video evidence cannot support final answer.")
        return GuardDecision(True, "Direct final allowed from verified binding and evidence lineage.")

    return GuardDecision(False, "Unhandled action.")

