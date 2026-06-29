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


def _compute_has_candidate_answer_verification(state: LoopState, compute_ref: str) -> bool:
    for evidence in state.evidence.values():
        if evidence.tool_name != "verify_alignment" or not evidence.ok:
            continue
        payload = evidence.payload or {}
        if payload.get("decision") != "candidate_answer":
            continue
        if payload.get("target_kind") not in {"compute_result", "final_answer"}:
            continue
        refs = {
            str(ref)
            for ref in [
                *(payload.get("compute_refs") or []),
                *(payload.get("target_refs") or []),
            ]
        }
        if compute_ref in refs:
            return True
    return False


def _semantic_field_id(card: Any) -> str | None:
    table = str(getattr(card, "canonical_table", "") or "").strip()
    field = str(getattr(card, "canonical_field", "") or "").strip()
    if not table or not field:
        return None
    return f"{table}.{field}".casefold()


def _semantic_refs_from_definition(definition: str) -> set[str]:
    refs = {f"{left}.{right}".casefold() for left, right in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b", definition)}
    alias_to_table: dict[str, str] = {}
    relation_pattern = re.compile(
        r"\b(?:from|join)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?(?:\s+(?:as\s+)?`?([A-Za-z_][A-Za-z0-9_]*)`?)?",
        re.IGNORECASE,
    )
    for match in relation_pattern.finditer(definition):
        table = match.group(1)
        alias = match.group(2)
        if alias:
            alias_to_table[alias.casefold()] = table.casefold()
    for left, field in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b", definition):
        table = alias_to_table.get(left.casefold())
        if table:
            refs.add(f"{table}.{field.casefold()}")
    return refs


def _active_required_semantic_cards(state: LoopState) -> list[Any]:
    if not state.matched_semantic_cards:
        return []
    top_card = state.matched_semantic_cards[0]
    if getattr(top_card, "canonical_field", None):
        candidates = [top_card]
    else:
        refs = _semantic_refs_from_definition(str(getattr(top_card, "definition", "") or ""))
        candidates = [
            card for card in state.semantic_cards
            if _semantic_field_id(card) in refs and getattr(card, "kind", "") in {"field", "metric"}
        ]
    mapped_cards: list[Any] = []
    for card in candidates:
        if any(
            mapping.card_id == card.id
            and mapping.source_id
            and mapping.status in {"exact_structured_source", "document_source"}
            for mapping in state.source_mappings
        ):
            mapped_cards.append(card)
    return mapped_cards


def _contract_covers_field(binding: Any, field_id: str) -> bool:
    metadata = getattr(binding, "metadata", {}) or {}
    contract = metadata.get("semantic_contract")
    canonical_fields: list[str] = []
    if isinstance(contract, dict):
        raw = contract.get("canonical_fields")
        if isinstance(raw, list):
            canonical_fields.extend(str(item).casefold() for item in raw)
        raw_mapping = contract.get("physical_field_mapping")
        if isinstance(raw_mapping, dict):
            canonical_fields.extend(str(key).casefold() for key in raw_mapping)
    raw = metadata.get("canonical_fields")
    if isinstance(raw, list):
        canonical_fields.extend(str(item).casefold() for item in raw)
    return field_id.casefold() in set(canonical_fields)


def _binding_covers_semantic_card(state: LoopState, binding: Any, card: Any) -> bool:
    field_id = _semantic_field_id(card)
    if not field_id:
        return False
    if _contract_covers_field(binding, field_id):
        return True
    field_norm = _normalize(str(getattr(card, "canonical_field", "") or ""))
    table_norm = _normalize(str(getattr(card, "canonical_table", "") or ""))
    allowed_columns = {_normalize(column) for column in getattr(binding, "allowed_columns", ())}
    binding_table = _normalize(str(getattr(binding, "table", "") or ""))
    for mapping in state.source_mappings:
        if mapping.card_id != card.id or mapping.source_id != getattr(binding, "source_id", None):
            continue
        if mapping.status == "exact_structured_source" and getattr(binding, "binding_type", "") in {"structured_source", "structured_field"}:
            if field_norm and field_norm not in allowed_columns:
                continue
            mapped_table = _normalize(str(mapping.matched_table or ""))
            if binding_table and mapped_table and binding_table != mapped_table:
                continue
            if table_norm and binding_table and table_norm != binding_table:
                continue
            return True
        if mapping.status == "document_source" and getattr(binding, "binding_type", "") == "document_record_set":
            return not field_norm or field_norm in allowed_columns
    return False


def _missing_semantic_compute_requirements(state: LoopState, binding_refs: tuple[str, ...]) -> list[str]:
    if not binding_refs:
        return []
    bindings = [state.bindings[ref] for ref in binding_refs if ref in state.bindings]
    missing: list[str] = []
    for card in _active_required_semantic_cards(state):
        field_id = _semantic_field_id(card)
        if not field_id:
            continue
        if not any(_binding_covers_semantic_card(state, binding, card) for binding in bindings):
            missing.append(field_id)
    return sorted(set(missing))


def _semantic_requirement_recommendations(state: LoopState, missing_fields: list[str]) -> tuple[dict[str, object], ...]:
    recommendations: list[dict[str, object]] = []
    cards_by_field = {
        _semantic_field_id(card): card
        for card in state.semantic_cards
        if _semantic_field_id(card)
    }
    for field_id in missing_fields[:4]:
        card = cards_by_field.get(field_id)
        if card is None:
            continue
        mappings = [mapping for mapping in state.source_mappings if mapping.card_id == card.id]
        document_sources = [mapping.source_id for mapping in mappings if mapping.status == "document_source" and mapping.source_id]
        if document_sources:
            recommendations.append(
                {
                    "tool_name": "run_document_agent",
                    "arguments": {
                        "question": state.question,
                        "target_fields": [str(getattr(card, "canonical_field", "") or "")],
                        "semantic_cards": [card.to_dict()],
                        "source_candidates": document_sources[:4],
                        "required_record_grain": str(getattr(card, "record_grain", "") or ""),
                    },
                    "reason": f"Collect verified document records for canonical field {field_id}.",
                }
            )
            continue
        structured_sources = [mapping.source_id for mapping in mappings if mapping.status == "exact_structured_source" and mapping.source_id]
        if structured_sources:
            recommendations.append(
                {
                    "tool_name": "inspect_source",
                    "arguments": {"source_ref": structured_sources[0]},
                    "reason": f"Observe the mapped source for canonical field {field_id} before compute.",
                }
            )
    return tuple(recommendations[:6])


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

    signature = action.signature()
    repeated = sum(1 for feedback in state.guard_feedback[-6:] if feedback.get("signature") == signature)
    if repeated >= 2:
        return GuardDecision(
            False,
            "Repeated identical action without effective progress.",
            allowed_next_tools=(
                "list_inventory",
                "retrieve_knowledge",
                "locate_sources",
                "inspect_source",
                "sample_records",
                "search_values",
                "run_document_agent",
                "inspect_relation",
                "discover_join_paths",
                "run_verified_compute",
                "submit_final",
                "blocked",
            ),
            recommended_next_actions=(),
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
                allowed = ("inspect_video", "blocked") if source.data_form == "video" else ("run_document_agent", "blocked")
                recommended = (
                    {
                        "tool_name": "inspect_video",
                        "arguments": {"source_ref": source.id},
                        "reason": "Inspect unsupported video metadata instead of treating it as a table.",
                    },
                ) if source.data_form == "video" else (
                    {
                        "tool_name": "run_document_agent",
                        "arguments": {"question": state.question, "source_candidates": [source.id]},
                        "reason": "Delegate PDF/MD search, record reading, extraction, and coverage to DocumentAgent.",
                    },
                )
                return GuardDecision(
                    False,
                    f"{source.data_form} cannot be inspected as a structured table; use a document/video tool.",
                    allowed_next_tools=allowed,
                    recommended_next_actions=recommended,
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
        partial_document_sets = [
            ref for ref in refs
            if (
                state.bindings[ref].binding_type == "document_record_set"
                and state.bindings[ref].metadata.get("partial_coverage")
            )
        ]
        if partial_document_sets:
            return GuardDecision(
                False,
                "Compute cannot use partial document record-set bindings; rerun DocumentAgent for a complete record set first: "
                + ", ".join(partial_document_sets),
                allowed_next_tools=("run_document_agent", "bind", "blocked"),
                recommended_next_actions=(
                    {
                        "tool_name": "run_document_agent",
                        "arguments": {"question": state.question},
                        "reason": "Let DocumentAgent search/read/extract additional record slices before recomputing.",
                    },
                ),
            )
        # Canonical semantic source selection is surfaced before the model chooses
        # sources (source_resolution.source_plan and locate_sources). The compute
        # guard stays focused on executable safety instead of acting as a strict
        # post-hoc semantic interceptor.
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
            if not _compute_has_candidate_answer_verification(state, action.compute_ref):
                return GuardDecision(
                    False,
                    "Compute-backed final requires verify_alignment(decision='candidate_answer', target_kind='compute_result') for this compute_ref before submit_final.",
                    allowed_next_tools=("verify_alignment", "run_verified_compute", "inspect_relation", "blocked"),
                    recommended_next_actions=(
                        {
                            "tool_name": "verify_alignment",
                            "arguments": {
                                "decision": "candidate_answer",
                                "target_kind": "compute_result",
                                "compute_refs": [action.compute_ref],
                                "evidence_refs": list(compute.evidence_refs),
                                "binding_refs": list(compute.binding_refs),
                                "alignment": "<explain why this exact compute result satisfies the question and knowledge>",
                            },
                            "reason": "Classify the compute result before final submission, or compute a different result.",
                        },
                    ),
                )
            final_columns = _compute_final_answer_columns(action)
            if not final_columns:
                return GuardDecision(
                    False,
                    "Compute-backed final requires an explicit answer.columns projection; do not submit all compute columns by default.",
                    allowed_next_tools=("submit_final", "inspect_relation", "run_verified_compute", "blocked"),
                    recommended_next_actions=(
                        {
                            "tool_name": "submit_final",
                            "arguments": {
                                "compute_ref": action.compute_ref,
                                "answer": {
                                    "columns": list(compute.columns),
                                },
                            },
                            "reason": "Choose the final output columns explicitly from this compute result, removing helper columns unless they are requested.",
                        },
                    ),
                )
            unknown_final_columns = [
                column for column in final_columns
                if column not in {str(item) for item in compute.columns}
            ]
            if unknown_final_columns:
                return GuardDecision(
                    False,
                    "Final answer.columns must be a subset of the compute result columns: "
                    + ", ".join(unknown_final_columns),
                    allowed_next_tools=("submit_final", "run_verified_compute", "inspect_relation", "blocked"),
                    recommended_next_actions=(
                        {
                            "tool_name": "inspect_relation",
                            "arguments": {},
                            "reason": "Inspect verified relation columns before computing or projecting final columns.",
                        },
                    ),
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

