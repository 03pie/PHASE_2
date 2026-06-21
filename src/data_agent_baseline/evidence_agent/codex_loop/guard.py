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

_STRUCTURED_FORMS = {"sqlite_database", "csv_records", "json_records"}
_DOCUMENT_FORMS = {"pdf_document", "markdown_document"}
_DIRECT_FINAL_BINDING_TYPES = {"document_window", "value", "operation", "answer_candidate"}
_BINDING_TYPES = COMPUTABLE_BINDING_TYPES | _DIRECT_FINAL_BINDING_TYPES
_META_EVIDENCE_TOOLS = {"verify_alignment", "track_requirements", "bind", "submit_final"}
_RELATION_PATTERN = re.compile(r'\b(?:from|join)\s+("([^"]+)"|[A-Za-z_][\w]*)', re.IGNORECASE)
_QUOTED_IDENTIFIER_PATTERN = re.compile(r'"([^"]+)"')


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


def _payload_table(payload: dict[str, Any]) -> str:
    return str(payload.get("table") or "").strip()


def _action_source_id(state: LoopState, action: ModelAction) -> str:
    path = action.arguments.get("path")
    source_ref = action.arguments.get("source_ref") or action.arguments.get("source_id")
    return str(source_ref or state.source_by_path.get(str(path)) or "").strip()


def _scheduled_recommendations(
    state: LoopState,
    *,
    preferred_tools: tuple[str, ...] = (),
    limit: int = 4,
) -> tuple[dict[str, object], ...]:
    from data_agent_baseline.evidence_agent.codex_loop.state_views import recovery_hints  # noqa: PLC0415

    schedule = [hint.to_dict() for hint in recovery_hints(state, limit=limit * 3)]
    if preferred_tools:
        preferred = [
            item for item in schedule
            if str(item.get("tool_name") or "") in preferred_tools
        ]
        others = [
            item for item in schedule
            if str(item.get("tool_name") or "") not in preferred_tools
        ]
        schedule = preferred + others
    return tuple(
        {
            "tool_name": item.get("tool_name"),
            "arguments": item.get("arguments") or {},
            "reason": item.get("reason") or "Use the next ledger-backed action instead of repeating a no-op observation.",
        }
        for item in schedule[:limit]
    )


def _successful_observation(
    state: LoopState,
    *,
    tool_name: str,
    source_id: str | None = None,
    table: str | None = None,
    binding_ref: str | None = None,
    relation_name: str | None = None,
    query: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
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
        if start_line is not None and int(payload.get("line_start") or payload.get("start_line") or -1) != start_line:
            continue
        if end_line is not None and int(payload.get("line_end") or payload.get("end_line") or -1) != end_line:
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
    scheduled = _scheduled_recommendations(
        state,
        preferred_tools=preferred_tools,
    )
    recommendations = tuple(
        dict.fromkeys(
            [repr(item) for item in (*fallback_actions, *scheduled)]
        )
    )
    recommendation_items: list[dict[str, object]] = []
    for item_repr in recommendations:
        for item in (*fallback_actions, *scheduled):
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
                preferred = ("profile_document", "search_document", "read_document_window", "bind")
                allowed = ("profile_document", "search_document", "read_document_window", "bind")
                fallback = (
                    {
                        "tool_name": "profile_document",
                        "arguments": {"source_ref": source_id},
                        "reason": "The source form is already observed; profile the document instead of re-inspecting it.",
                    },
                    {
                        "tool_name": "search_document",
                        "arguments": {"source_ref": source_id, "query": state.question},
                        "reason": "Search bounded document windows after observing the document source.",
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

    if tool_name == "profile_document" and source_id:
        existing = _successful_observation(
            state,
            tool_name="profile_document",
            source_id=source_id,
        )
        if existing is not None:
            return _noop_observation_guard(
                state,
                reason="This document profile has already been observed successfully; repeat profiling would not add evidence.",
                allowed_next_tools=("search_document", "read_document_window", "extract_records", "bind"),
                preferred_tools=("search_document", "read_document_window", "extract_records", "bind"),
                fallback_actions=(
                    {
                        "tool_name": "search_document",
                        "arguments": {"source_ref": source_id, "query": state.question},
                        "reason": "Search bounded windows after profiling the document.",
                    },
                ),
            )

    if tool_name == "search_document" and source_id:
        query = str(action.arguments.get("query") or state.question or "").strip()
        existing = _successful_observation(
            state,
            tool_name="search_document",
            source_id=source_id,
            query=query,
        )
        if existing is not None:
            return _noop_observation_guard(
                state,
                reason="This document query has already returned a successful bounded-window observation; repeat search would not add evidence.",
                allowed_next_tools=("read_document_window", "extract_records", "bind", "search_document"),
                preferred_tools=("read_document_window", "extract_records", "bind", "search_document"),
                fallback_actions=(
                    {
                        "tool_name": "extract_records",
                        "arguments": {"evidence_refs": [existing.id], "spec": {}},
                        "reason": "If the window contains repeated records, provide an extraction spec over the existing evidence.",
                    },
                ),
            )

    if tool_name == "read_document_window" and source_id:
        start = action.arguments.get("line_start") or action.arguments.get("start_line")
        end = action.arguments.get("line_end") or action.arguments.get("end_line")
        try:
            start_int = int(start) if start is not None else None
            end_int = int(end) if end is not None else None
        except (TypeError, ValueError):
            start_int = None
            end_int = None
        if start_int is not None and end_int is not None:
            existing = _successful_observation(
                state,
                tool_name="read_document_window",
                source_id=source_id,
                start_line=start_int,
                end_line=end_int,
            )
            if existing is not None:
                return _noop_observation_guard(
                    state,
                    reason="This exact document window has already been read successfully; repeat reading would not add evidence.",
                    allowed_next_tools=("extract_records", "bind", "search_document", "read_document_window"),
                    preferred_tools=("extract_records", "bind", "search_document", "read_document_window"),
                    fallback_actions=(
                        {
                            "tool_name": "bind",
                            "arguments": {"binding_type": "document_window", "evidence_refs": [existing.id]},
                            "reason": "Bind the existing document window if it directly supports the answer.",
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

    signature = action.signature()
    repeated = sum(1 for feedback in state.guard_feedback[-6:] if feedback.get("signature") == signature)
    if repeated >= 2:
        from data_agent_baseline.evidence_agent.codex_loop.state_views import recovery_hints  # noqa: PLC0415

        recovery_actions = [hint.to_dict() for hint in recovery_hints(state, limit=4)]
        recovery_tools = tuple(
            dict.fromkeys(
                str(item.get("tool_name"))
                for item in recovery_actions
                if str(item.get("tool_name") or "").strip()
            )
        )
        return GuardDecision(
            False,
            "Repeated identical action without effective progress.",
            allowed_next_tools=tuple(dict.fromkeys((*recovery_tools, "blocked")))
            or (
                "locate_sources",
                "inspect_relation",
                "profile_document",
                "search_document",
                "blocked",
            ),
            recommended_next_actions=tuple(recovery_actions)
            or (
                {
                    "tool_name": "locate_sources",
                    "arguments": {"query": state.question},
                    "reason": "Switch to alternative observed source candidates.",
                },
                {
                    "tool_name": "blocked",
                    "arguments": {"reason": "Repeated action failed without new evidence."},
                    "reason": "Stop instead of repeating the same failed action.",
                },
            ),
        )

    if action.kind == "blocked":
        return GuardDecision(True, "Model reported blocked/conflict.")

    if action.kind == "tool_call":
        if not action.tool_name or registry.spec(action.tool_name) is None:
            return GuardDecision(
                False,
                f"Unknown tool: {action.tool_name}",
                allowed_next_tools=registry.tool_names,
            )
        if action.tool_name in {"run_verified_compute", "submit_final"}:
            return GuardDecision(False, f"Use action kind for {action.tool_name}, not tool_call.")
        path = action.arguments.get("path")
        source_ref = action.arguments.get("source_ref") or action.arguments.get("source_id")
        source_ref_valid = _source_exists(state, str(source_ref)) if source_ref else False
        if path and not source_ref_valid and not _path_is_observed(state, str(path)):
            return GuardDecision(False, "Tool references a path that is not in observed inventory.")
        if source_ref and not source_ref_valid:
            return GuardDecision(False, "Tool references an unknown source_ref.")
        if action.tool_name in {"inspect_source", "sample_records"}:
            source_id = str(source_ref or state.source_by_path.get(str(path)) or "")
            source = state.sources.get(source_id)
            if source is not None and source.data_form in _DOCUMENT_FORMS | {"video"}:
                return GuardDecision(
                    False,
                    f"{source.data_form} cannot be inspected as a structured table; use a document/video tool.",
                    allowed_next_tools=(
                        "profile_document",
                        "search_document",
                        "read_document_window",
                        "inspect_video",
                        "blocked",
                    ),
                    recommended_next_actions=(
                        {
                            "tool_name": "profile_document",
                            "arguments": {"source_ref": source.id},
                            "reason": "Profile the document before deciding whether windows or extraction can support the task.",
                        },
                        {
                            "tool_name": "search_document",
                            "arguments": {"source_ref": source.id, "query": state.question},
                            "reason": "Search bounded document windows instead of treating the document as a table.",
                        },
                    ),
                )
        if action.tool_name == "read_document_window":
            source_id = str(source_ref or state.source_by_path.get(str(path)) or "")
            source = state.sources.get(source_id)
            if source is not None and source.data_form not in _DOCUMENT_FORMS:
                return GuardDecision(False, f"{source.data_form} is not a PDF/MD document source.")
        if action.tool_name in {"profile_document", "search_document"}:
            source_id = str(source_ref or state.source_by_path.get(str(path)) or "")
            source = state.sources.get(source_id)
            if source is not None and source.data_form not in _DOCUMENT_FORMS:
                return GuardDecision(
                    False,
                    f"{source.data_form} is not a PDF/MD document source.",
                    allowed_next_tools=("inspect_source", "sample_records", "search_values"),
                )
        if action.tool_name == "inspect_relation":
            relation_name = str(action.arguments.get("relation_name") or "").strip()
            binding_ref = str(action.arguments.get("binding_ref") or "").strip()
            relations = _relation_by_name(state)
            if relation_name and relation_name not in relations:
                return GuardDecision(False, f"Unknown verified relation: {relation_name}")
            if binding_ref and binding_ref not in state.bindings:
                return GuardDecision(False, f"Unknown binding_ref: {binding_ref}")
        repeated_observation = _guard_repeated_observation(state, action)
        if repeated_observation is not None:
            return repeated_observation
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
        if not any(item.tool_name not in _META_EVIDENCE_TOOLS for item in evidence_items):
            return GuardDecision(
                False,
                "Binding requires at least one successful observed evidence item, not only verifier or ledger evidence.",
                allowed_next_tools=(
                    "inspect_source",
                    "sample_records",
                    "search_values",
                    "read_document_window",
                    "search_document",
                    "extract_records",
                ),
            )
        if action.binding_type in {"structured_source", "structured_field"}:
            source_id = action.source_ref or evidence_items[0].source_id
            source = state.sources.get(source_id or "")
            if source is None:
                return GuardDecision(False, "Structured binding requires an observed source.")
            if source.data_form not in _STRUCTURED_FORMS:
                return GuardDecision(False, f"{source.data_form} cannot be bound as structured data.")
            table = str(action.arguments.get("table") or "").strip() or None
            existing = _equivalent_structured_binding(state, source_id=source_id, table=table)
            if existing is not None:
                relation_name = str(getattr(existing, "relation_name", "") or "")
                binding_ref = str(getattr(existing, "id", "") or "")
                return GuardDecision(
                    False,
                    "Equivalent structured source is already bound; do not create duplicate relation bindings.",
                    allowed_next_tools=("inspect_relation", "run_verified_compute", "blocked"),
                    recommended_next_actions=(
                        {
                            "tool_name": "inspect_relation",
                            "arguments": {"binding_ref": binding_ref},
                            "reason": "Inspect the existing verified relation instead of binding the same source again.",
                        },
                        {
                            "tool_name": "run_verified_compute",
                            "arguments": {"binding_refs": [binding_ref]},
                            "reason": "Use the existing verified relation for targeted SQL written from the question and inspected columns.",
                        },
                    )
                    if binding_ref and relation_name
                    else _inspect_relation_recommendation(),
                )
            if action.binding_type == "structured_field" and not (
                action.arguments.get("field") or action.arguments.get("allowed_columns")
            ):
                return GuardDecision(False, "Structured field binding requires a field or allowed_columns.")
        if action.binding_type == "document_record_set":
            has_records = any(isinstance(item.payload.get("records"), list) for item in evidence_items)
            if not has_records:
                return GuardDecision(False, "Document record-set binding requires extracted record evidence.")
        if action.binding_type == "document_window":
            has_window_text = any(
                item.data_form in _DOCUMENT_FORMS
                and (
                    isinstance(item.payload.get("text"), str)
                    or isinstance(item.payload.get("windows"), list)
                )
                for item in evidence_items
            )
            if not has_window_text:
                return GuardDecision(
                    False,
                    "Document-window binding requires successful PDF/MD window or search evidence.",
                    allowed_next_tools=("profile_document", "search_document", "read_document_window"),
                )
        if any(item.data_form == "video" for item in evidence_items):
            return GuardDecision(False, "V1 video evidence cannot become a final binding.")
        return GuardDecision(True, "Binding allowed from observed evidence.")

    if action.kind == "compute":
        if not action.sql:
            return GuardDecision(False, "Compute requires SQL.")
        refs = action.binding_refs or tuple(state.bindings)
        ok_computes = [result for result in state.compute_results.values() if result.ok]
        previously_computed_refs = {
            binding_ref
            for result in ok_computes
            for binding_ref in result.binding_refs
        }
        if len(ok_computes) >= 10 and set(refs).issubset(previously_computed_refs):
            return GuardDecision(
                False,
                "Multiple successful compute results already exist; choose a final result, gather non-compute evidence for a specific gap, or block instead of continuing exploratory compute.",
                allowed_next_tools=("submit_final", "inspect_relation", "bind", "blocked"),
                recommended_next_actions=tuple(
                    {
                        "tool_name": "submit_final",
                        "arguments": {"compute_ref": result.id},
                        "reason": "Submit this successful compute if it answers the question.",
                    }
                    for result in ok_computes[-3:]
                )
                + (
                    {
                        "tool_name": "blocked",
                        "arguments": {
                            "reason": "Successful computes do not satisfy the requested evidence requirements."
                        },
                        "reason": "Stop only if no successful compute satisfies the question.",
                    },
                ),
            )
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
                return GuardDecision(
                    False,
                    "Final compute result has no output rows; use more evidence, a different compute, or blocked.",
                    allowed_next_tools=("run_verified_compute", "inspect_relation", "blocked"),
                    recommended_next_actions=(
                        {
                            "tool_name": "blocked",
                            "arguments": {
                                "reason": "Verified compute returned no output rows for the requested answer."
                            },
                            "reason": "Use blocked if the empty result proves the requested data is absent.",
                        },
                    ),
                )
            return GuardDecision(True, "Final allowed from verified compute result.")

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
        direct_bindings = [state.bindings[ref] for ref in action.binding_refs]
        if not any(binding.binding_type in _DIRECT_FINAL_BINDING_TYPES for binding in direct_bindings):
            return GuardDecision(
                False,
                "Direct final requires a value/document/operation/answer_candidate binding; use compute_ref for relation outputs.",
                allowed_next_tools=("run_verified_compute", "bind", "inspect_relation"),
            )
        evidence_items = [state.evidence[ref] for ref in action.evidence_refs]
        if not all(item.ok for item in evidence_items):
            return GuardDecision(False, "Direct final cannot cite failed evidence.")
        if any(item.data_form == "video" for item in evidence_items):
            return GuardDecision(False, "V1 video evidence cannot support final answer.")
        return GuardDecision(True, "Direct final allowed from verified binding and evidence lineage.")

    return GuardDecision(False, "Unhandled action.")
