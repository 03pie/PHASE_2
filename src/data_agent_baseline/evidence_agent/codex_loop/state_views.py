from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.evidence_agent.codex_loop.lineage import DIRECT_FINAL_BINDING_TYPES
from data_agent_baseline.evidence_agent.codex_loop.protocol import LoopState


def _normalize(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value or "").casefold())


def answer_contract_view(state: LoopState) -> dict[str, Any]:
    contract = state.answer_contract
    if contract is None:
        return {
            "present": False,
            "instruction": (
                "No LLM-declared question contract yet. The next useful action is "
                "declare_answer_contract with intent_summary, answer_grain, final_outputs, "
                "constraints, operations, helper_fields, field_roles, row_shape, null_policy, "
                "transform_intent, document_policy, and unresolved_terms. This is semantic "
                "intent only, not a physical schema or source choice."
            ),
        }
    return {
        "present": True,
        "contract": contract.to_dict(),
        "instruction": (
            "Use this as the per-turn semantic answer contract. It does not prove any "
            "physical field/source; inspect, extract, bind, and compute from observed evidence."
        ),
    }


def semantic_selection_view(state: LoopState) -> dict[str, Any]:
    if state.semantic_selection is None:
        return {
            "present": False,
            "candidate_cards": [
                {
                    "id": card.id,
                    "kind": card.kind,
                    "name": card.name,
                    "semantic_scope": card.semantic_scope,
                    "semantic_slot": card.semantic_slot,
                    "definition_preview": str(card.definition or "")[:240],
                    "unit": card.unit,
                    "record_grain": card.record_grain,
                }
                for card in state.matched_semantic_cards[:12]
            ],
            "instruction": (
                "No LLM semantic card selection yet. Select only knowledge cards that the "
                "LLM judges relevant to this question; put unmatched intents in unmapped_intents."
            ),
            "errors": list(state.semantic_selection_errors[-5:]),
        }
    return {
        "present": True,
        "selection": state.semantic_selection.to_dict(),
        "selected_source_mapping_count": len(state.selected_source_mappings),
        "instruction": (
            "Only selected cards are eligible for automatic source candidate expansion. "
            "Use unmapped_intents with inventory/search tools."
        ),
    }


def selected_source_candidates(state: LoopState) -> dict[str, Any]:
    mappings = state.selected_source_mappings if state.semantic_selection is not None else []
    candidates: list[dict[str, Any]] = []
    for mapping in mappings:
        source_status = _source_observation_status(state, mapping.source_id)
        candidate_kind = str(mapping.status or "")
        candidates.append(
            {
                "card_id": mapping.card_id,
                "semantic_scope": mapping.semantic_scope,
                "semantic_slot": mapping.semantic_slot,
                "source_id": mapping.source_id,
                "source_path": mapping.source_path,
                "data_form": mapping.data_form,
                "physical_table": mapping.physical_table,
                "physical_field": mapping.physical_field,
                "candidate_kind": candidate_kind,
                "recommended_tool": (
                    "run_document_agent"
                    if candidate_kind == "unverified_document_candidate"
                    else "inspect_source"
                    if candidate_kind == "unverified_structured_candidate"
                    else "locate_sources"
                ),
                **source_status,
            }
        )
    priority = {
        "unverified_structured_candidate": 0,
        "unverified_document_candidate": 1,
        "unverified_lexical_candidate": 2,
        "unresolved_grounding": 3,
    }
    candidates.sort(
        key=lambda item: (
            item.get("coverage_status") != "unseen",
            priority.get(str(item.get("candidate_kind") or ""), 9),
            str(item.get("source_path") or ""),
            str(item.get("physical_table") or ""),
            str(item.get("physical_field") or ""),
        )
    )
    return {
        "instruction": (
            "Selected-card source candidates only. They are grounding candidates, not proof. "
            "Observe them with inspect_source/sample_records or run_document_agent before bind."
        ),
        "candidates": candidates,
    }


def _source_observation_status(state: LoopState, source_id: str | None) -> dict[str, Any]:
    if not source_id:
        return {"coverage_status": "missing_source", "evidence_refs": [], "binding_refs": []}
    evidence_refs = [
        evidence.id
        for evidence in state.evidence.values()
        if evidence.source_id == source_id and evidence.ok
    ]
    binding_refs = [
        binding.id
        for binding in state.bindings.values()
        if binding.source_id == source_id
    ]
    if binding_refs:
        status = "bound"
    elif evidence_refs:
        status = "observed"
    else:
        status = "unseen"
    return {"coverage_status": status, "evidence_refs": evidence_refs, "binding_refs": binding_refs}


def primary_next_action(state: LoopState) -> dict[str, Any]:
    if state.answer_contract is None:
        return {
            "tool_name": "declare_answer_contract",
            "arguments": {
                "intent_summary": "<restate what the question asks in semantic terms>",
                "answer_grain": "<what one answer row represents>",
                "final_outputs": [],
                "constraints": [
                    {
                        "semantic_field": "<semantic field used by a question condition>",
                        "operator": "<semantic operator from the question>",
                        "value": "<question value, if any>",
                        "reason": "<why this condition is required>",
                    }
                ],
                "operations": {
                    "row_shape": "preserve_rows",
                    "sort_by": [],
                    "group_by": [],
                    "aggregate": [],
                    "top_n": None,
                    "reason": "<semantic row operation intent>",
                },
                "helper_fields": {
                    "filter_fields": [],
                    "sort_fields": [],
                    "join_keys": [],
                    "row_selection_fields": [],
                    "evidence_anchor_fields": [],
                },
                "field_roles": [],
                "row_shape": "preserve_rows",
                "null_policy": "preserve",
                "transform_intent": (
                    "<state semantic filters/sort/aggregation/join intent; do not put helper "
                    "fields in final_outputs>"
                ),
                "document_policy": {
                    "include_missing_records": True,
                    "required_fields": [],
                },
                "unresolved_terms": [],
            },
            "reason": (
                "Declare the question understanding contract before choosing knowledge cards or physical sources. "
                "This is not a physical schema binding."
            ),
        }
    if state.semantic_selection is None:
        if state.semantic_cards:
            return {
                "tool_name": "select_semantic_cards",
                "arguments": {
                    "card_ids": [],
                    "rationale": "<choose relevant knowledge semantic cards for this question>",
                    "unmapped_intents": [],
                },
                "reason": (
                    "Select the business semantic cards before expanding source candidates. "
                    "Do not choose physical sources from the full knowledge catalog."
                ),
            }
        return {
            "tool_name": "locate_sources",
            "arguments": {"query": state.question},
            "reason": "No semantic knowledge cards are available; use observed inventory search.",
        }
    candidates = selected_source_candidates(state).get("candidates", [])
    for candidate in candidates:
        if candidate.get("coverage_status") != "unseen" or not candidate.get("source_id"):
            continue
        tool_name = str(candidate.get("recommended_tool") or "inspect_source")
        if tool_name == "run_document_agent":
            return {
                "tool_name": "run_document_agent",
                "arguments": {
                    "question": state.question,
                    "source_candidates": [candidate.get("source_id")],
                    "target_fields": [
                        candidate.get("semantic_slot") or candidate.get("semantic_scope") or ""
                    ],
                    "semantic_cards": [
                        card
                        for card in (
                            state.semantic_selection.selected_cards
                            if state.semantic_selection is not None
                            else ()
                        )
                        if card.get("id") == candidate.get("card_id")
                    ],
                },
                "reason": "Observe the next selected document candidate before binding.",
            }
        if tool_name == "inspect_source":
            args: dict[str, Any] = {"source_ref": candidate.get("source_id")}
            if candidate.get("physical_table"):
                args["table"] = candidate.get("physical_table")
            return {
                "tool_name": "inspect_source",
                "arguments": args,
                "reason": "Inspect the next selected structured source candidate before binding.",
            }
    if state.bindings and not any(result.ok and result.rows for result in state.compute_results.values()):
        relation_bindings = [
            binding for binding in state.bindings.values() if binding.relation_name
        ]
        return {
            "tool_name": "run_verified_compute" if relation_bindings else "submit_final",
            "arguments": (
                {
                    "binding_refs": [binding.id for binding in relation_bindings],
                    "sql": "<write SQL over relation names from verified bindings>",
                }
                if relation_bindings
                else {
                    "binding_refs": list(state.bindings),
                    "evidence_refs": [
                        ref
                        for binding in state.bindings.values()
                        for ref in binding.evidence_refs
                    ],
                    "answer": {"columns": [], "rows": []},
                }
            ),
            "reason": "Verified bindings exist; compute or direct-final from those bindings.",
        }
    successful = [result for result in state.compute_results.values() if result.ok and result.rows]
    if successful:
        latest = successful[-1]
        return {
            "tool_name": "submit_final",
            "arguments": {
                "compute_ref": latest.id,
                "answer": {"columns": []},
            },
            "reason": "A successful compute exists; submit an explicit projection or recompute if it is not the answer.",
        }
    return {
        "tool_name": "locate_sources",
        "arguments": {"query": state.question},
        "reason": "No selected candidate remains unobserved; use inventory search for unmapped intents or block with evidence.",
    }


def _slot_from_field_ref(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    normalized = _normalize(text)
    return normalized or None


def _binding_semantic_slots(binding: Any) -> set[str]:
    slots: set[str] = set()
    metadata = getattr(binding, "metadata", {}) or {}
    contract = metadata.get("semantic_contract") if isinstance(metadata, dict) else None
    if isinstance(contract, dict):
        raw_fields = contract.get("canonical_fields")
        if isinstance(raw_fields, list):
            slots.update(slot for item in raw_fields if (slot := _slot_from_field_ref(item)))
        raw_mapping = contract.get("physical_field_mapping")
        if isinstance(raw_mapping, dict):
            slots.update(slot for key in raw_mapping if (slot := _slot_from_field_ref(key)))
        raw_field_mappings = contract.get("field_mappings")
        if isinstance(raw_field_mappings, list):
            for item in raw_field_mappings:
                if isinstance(item, dict):
                    slots.update(slot for key in ("canonical", "semantic", "field") if (slot := _slot_from_field_ref(item.get(key))))
    slots.update(slot for column in getattr(binding, "allowed_columns", ()) if (slot := _slot_from_field_ref(column)))
    return slots


def source_coverage_map(state: LoopState) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for source in state.sources.values():
        evidence_items = [
            evidence for evidence in state.evidence.values()
            if evidence.source_id == source.id
        ]
        binding_refs = [
            binding.id for binding in state.bindings.values()
            if binding.source_id == source.id
        ]
        positive_refs = [evidence.id for evidence in evidence_items if evidence.ok]
        negative_refs = [evidence.id for evidence in evidence_items if not evidence.ok]
        tools = list(dict.fromkeys(evidence.tool_name for evidence in evidence_items))
        if binding_refs:
            status = "bound"
        elif positive_refs:
            status = "observed"
        elif negative_refs:
            status = "negative_only"
        else:
            status = "unseen"
        status_counts[status] = status_counts.get(status, 0) + 1
        rows.append(
            {
                "source_id": source.id,
                "path": source.virtual_path,
                "data_form": source.data_form,
                "coverage_status": status,
                "observed_tools": tools,
                "positive_evidence_refs": positive_refs,
                "negative_evidence_refs": negative_refs,
                "binding_refs": binding_refs,
            }
        )
    return {
        "source_count": len(rows),
        "status_counts": status_counts,
        "sources": rows,
    }


def final_output_contract(state: LoopState) -> dict[str, Any]:
    answer = state.final_answer or {}
    issues: list[str] = []
    warnings: list[str] = []
    compute_ref = str(answer.get("compute_ref") or "")
    source = "compute" if compute_ref else "direct"
    columns: list[Any] | None = None
    rows: list[Any] | None = None

    if compute_ref:
        compute = state.compute_results.get(compute_ref)
        if compute is None:
            issues.append("unknown_compute_ref")
        elif not compute.ok:
            issues.append("failed_compute_ref")
        else:
            raw_columns = answer.get("columns")
            raw_rows = answer.get("rows")
            columns = raw_columns if isinstance(raw_columns, list) else list(compute.columns)
            rows = raw_rows if isinstance(raw_rows, list) else [list(row) for row in compute.rows]
    else:
        raw_columns = answer.get("columns")
        raw_rows = answer.get("rows")
        if isinstance(raw_columns, list):
            columns = raw_columns
        else:
            issues.append("columns_not_list")
        if isinstance(raw_rows, list):
            rows = raw_rows
        else:
            issues.append("rows_not_list")

    normalized_columns = [str(column) for column in columns] if columns is not None else []
    if columns is not None:
        if not normalized_columns:
            issues.append("empty_columns")
        if any(not column.strip() for column in normalized_columns):
            issues.append("blank_column_name")
        duplicate_columns = sorted(
            column for column in set(normalized_columns)
            if normalized_columns.count(column) > 1
        )
        if duplicate_columns:
            issues.append("duplicate_columns:" + ",".join(duplicate_columns))

    if rows is not None:
        for index, row in enumerate(rows):
            if isinstance(row, tuple):
                row_values = list(row)
            elif isinstance(row, list):
                row_values = row
            else:
                issues.append(f"row_not_sequence:{index}")
                continue
            if normalized_columns and len(row_values) != len(normalized_columns):
                issues.append(f"row_width_mismatch:{index}")
                break
    if rows == []:
        issues.append("empty_rows")

    return {
        "answer_present": bool(state.final_answer),
        "source": source,
        "compute_ref": compute_ref or None,
        "columns": normalized_columns,
        "row_count": len(rows) if isinstance(rows, list) else None,
        "issues": issues,
        "warnings": warnings,
        "passed": bool(state.final_answer) and not issues,
    }


def answer_candidates(state: LoopState) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for result in state.compute_results.values():
        if not result.ok or not result.rows:
            continue
        candidates.append(
            {
                "kind": "compute_result",
                "compute_ref": result.id,
                "columns": list(result.columns),
                "row_count": len(result.rows),
                "preview_rows": [list(row) for row in result.rows[:5]],
                "binding_refs": list(result.binding_refs),
                "evidence_refs": list(result.evidence_refs),
                "sql": result.sql,
            }
        )
    for binding in state.bindings.values():
        if binding.binding_type not in DIRECT_FINAL_BINDING_TYPES:
            continue
        candidates.append(
            {
                "kind": "direct_evidence_binding",
                "binding_ref": binding.id,
                "binding_type": binding.binding_type,
                "source_id": binding.source_id,
                "evidence_refs": list(binding.evidence_refs),
                "alignment": binding.alignment,
                "metadata": binding.metadata,
            }
        )
    return candidates


def exhaustion_status(state: LoopState) -> dict[str, Any]:
    observed_sources: dict[str, dict[str, Any]] = {}
    for evidence in state.evidence.values():
        if not evidence.source_id:
            continue
        source = state.sources.get(evidence.source_id)
        observed_sources.setdefault(
            evidence.source_id,
            {
                "source_id": evidence.source_id,
                "path": source.virtual_path if source else None,
                "data_form": evidence.data_form or (source.data_form if source else None),
                "tools": [],
                "ok_evidence": 0,
                "failed_evidence": 0,
            },
        )
        item = observed_sources[evidence.source_id]
        if evidence.tool_name not in item["tools"]:
            item["tools"].append(evidence.tool_name)
        if evidence.ok:
            item["ok_evidence"] += 1
        else:
            item["failed_evidence"] += 1

    negative_by_kind: dict[str, int] = {}
    for scope in state.negative_scopes:
        kind = str(scope.get("kind") or "unknown")
        negative_by_kind[kind] = negative_by_kind.get(kind, 0) + 1

    return {
        "observed_source_count": len(observed_sources),
        "observed_sources": list(observed_sources.values())[-20:],
        "negative_scope_count": len(state.negative_scopes),
        "negative_by_kind": negative_by_kind,
        "recent_negative_scopes": state.negative_scopes[-12:],
    }


