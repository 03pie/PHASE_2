from __future__ import annotations

import json
import re
from typing import Any

from data_agent_baseline.evidence_agent.codex_loop.protocol import LoopState, RecoveryHint

DIRECT_FINAL_BINDING_TYPES = {"document_window", "value", "operation", "answer_candidate"}
META_EVIDENCE_TOOLS = {"verify_alignment", "track_requirements", "bind", "submit_final"}
STRUCTURED_FORMS = {"sqlite_database", "csv_records", "json_records"}
DOCUMENT_FORMS = {"pdf_document", "markdown_document"}
NON_ANSWER_DECISIONS = {
    "intermediate",
    "not_applicable",
    "needs_more_evidence",
    "conflict",
}
ACTIONABLE_RECOVERY_SOURCES = {
    "bootstrap_source_discovery",
    "bootstrap_knowledge",
    "ready_compute_final",
    "observed_structured_source_binding",
    "observed_document_window_binding",
    "observed_document_record_set_binding",
    "verified_relation_inspection",
    "verified_relation_compute",
    "unclassified_compute",
    "answer_candidate",
    "positive_unbound_evidence",
    "co_occurring_value_candidate",
    "unexplored_observed_source",
}
_SHAPE_CHANGING_SQL_PATTERNS = (
    ("order_by", re.compile(r"\border\s+by\b", re.IGNORECASE)),
    ("limit", re.compile(r"\blimit\b", re.IGNORECASE)),
    ("offset", re.compile(r"\boffset\b", re.IGNORECASE)),
)


def _terms(value: Any) -> set[str]:
    return {
        part
        for part in re.split(r"[^0-9A-Za-z_]+", str(value).casefold())
        if len(part) >= 2
    }


def _compute_shape_operations(sql: str) -> list[str]:
    return [
        name for name, pattern in _SHAPE_CHANGING_SQL_PATTERNS
        if pattern.search(sql)
    ]


def verifier_decisions(state: LoopState) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for evidence in state.evidence.values():
        if evidence.tool_name != "verify_alignment" or not evidence.ok:
            continue
        payload = evidence.payload or {}
        decisions.append(
            {
                "evidence_ref": evidence.id,
                "decision": payload.get("decision"),
                "target_kind": payload.get("target_kind"),
                "target_refs": payload.get("target_refs") or [],
                "requirement_refs": payload.get("requirement_refs") or [],
                "evidence_refs": payload.get("evidence_refs") or [],
                "binding_refs": payload.get("binding_refs") or [],
                "compute_refs": payload.get("compute_refs") or [],
                "alignment": payload.get("alignment") or "",
                "limitations": payload.get("limitations") or "",
            }
        )
    return decisions


def compute_result_classifications(state: LoopState) -> dict[str, dict[str, Any]]:
    classifications: dict[str, dict[str, Any]] = {}
    for decision in verifier_decisions(state):
        if decision["target_kind"] != "compute_result":
            continue
        refs = [str(ref) for ref in decision.get("compute_refs") or []]
        refs.extend(
            str(ref)
            for ref in decision.get("target_refs") or []
            if str(ref).startswith("comp_")
        )
        for compute_ref in refs:
            classifications[compute_ref] = decision
    return classifications


def requirement_coverage(state: LoopState) -> dict[str, Any]:
    requirements = [requirement.to_dict() for requirement in state.requirements.values()]
    pending = [
        item for item in requirements
        if item.get("status") in {"pending", "", None}
    ]
    satisfied = [
        item for item in requirements
        if item.get("status") == "satisfied"
    ]
    conflicts = [
        item for item in requirements
        if item.get("status") == "conflict"
    ]
    blocked = [
        item for item in requirements
        if item.get("status") == "blocked"
    ]
    weak_satisfied = [
        item for item in satisfied
        if not (item.get("evidence_refs") or item.get("binding_refs") or item.get("compute_refs"))
    ]
    return {
        "declared_count": len(requirements),
        "satisfied_count": len(satisfied),
        "pending_count": len(pending),
        "conflict_count": len(conflicts),
        "blocked_count": len(blocked),
        "requirements": requirements,
        "pending_requirements": pending,
        "conflict_requirements": conflicts,
        "blocked_requirements": blocked,
        "weak_satisfied_requirements": weak_satisfied,
        "passed": not pending and not conflicts and not blocked and not weak_satisfied,
    }


def source_coverage_map(state: LoopState) -> dict[str, Any]:
    """Summarize per-source observation status without semantic binding.

    This mirrors Codex's tool-output ledger discipline: the model should be
    able to see what has actually been observed, what is still only a
    candidate, and which sources have produced scoped negative evidence.
    """
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for source in state.sources.values():
        evidence_items = [
            evidence for evidence in state.evidence.values()
            if evidence.source_id == source.id
        ]
        candidate_refs = [
            candidate.id for candidate in state.candidates.values()
            if candidate.source_id == source.id
        ]
        binding_refs = [
            binding.id for binding in state.bindings.values()
            if binding.source_id == source.id
        ]
        tools: list[str] = []
        positive_refs: list[str] = []
        negative_refs: list[str] = []
        observed_tables: set[str] = set()
        observed_fields: set[str] = set()
        for evidence in evidence_items:
            if evidence.tool_name not in tools:
                tools.append(evidence.tool_name)
            if evidence.ok:
                positive_refs.append(evidence.id)
            else:
                negative_refs.append(evidence.id)
            payload = evidence.payload or {}
            table = payload.get("table")
            if table:
                observed_tables.add(str(table))
            columns = payload.get("columns")
            if isinstance(columns, list):
                observed_fields.update(str(column) for column in columns)
            sample = payload.get("sample")
            if isinstance(sample, list):
                for row in sample[:5]:
                    if isinstance(row, dict):
                        observed_fields.update(str(column) for column in row)

        inspected = any(
            evidence.tool_name == "inspect_source" and evidence.ok
            for evidence in evidence_items
        )
        sampled = any(
            evidence.tool_name == "sample_records" and evidence.ok
            for evidence in evidence_items
        ) or any(
            evidence.tool_name == "inspect_source"
            and evidence.ok
            and isinstance((evidence.payload or {}).get("sample"), list)
            and bool((evidence.payload or {}).get("sample"))
            for evidence in evidence_items
        )
        profiled = any(
            evidence.tool_name == "profile_document" and evidence.ok
            for evidence in evidence_items
        )
        window_observed = any(
            evidence.tool_name in {"search_document", "read_document_window"}
            and evidence.ok
            and (
                isinstance((evidence.payload or {}).get("windows"), list)
                or isinstance((evidence.payload or {}).get("text"), str)
            )
            for evidence in evidence_items
        )
        record_set_observed = any(
            evidence.tool_name == "extract_records"
            and evidence.ok
            and isinstance((evidence.payload or {}).get("records"), list)
            and bool((evidence.payload or {}).get("records"))
            for evidence in evidence_items
        )
        unsupported = any(
            evidence.data_form == "video"
            and (not evidence.ok or (evidence.payload or {}).get("unsupported"))
            for evidence in evidence_items
        )

        if binding_refs:
            coverage_status = "bound"
        elif unsupported:
            coverage_status = "unsupported"
        elif record_set_observed:
            coverage_status = "record_set_observed"
        elif window_observed:
            coverage_status = "document_window_observed"
        elif profiled:
            coverage_status = "document_profiled"
        elif sampled:
            coverage_status = "sample_observed"
        elif inspected:
            coverage_status = "schema_observed"
        elif evidence_items and not positive_refs:
            coverage_status = "negative_only"
        elif candidate_refs:
            coverage_status = "candidate_only"
        else:
            coverage_status = "unseen"

        status_counts[coverage_status] = status_counts.get(coverage_status, 0) + 1
        rows.append(
            {
                "source_id": source.id,
                "path": source.virtual_path,
                "data_form": source.data_form,
                "coverage_status": coverage_status,
                "observed_tools": tools,
                "candidate_refs": candidate_refs,
                "positive_evidence_refs": positive_refs,
                "negative_evidence_refs": negative_refs,
                "binding_refs": binding_refs,
                "observed_tables": sorted(observed_tables),
                "observed_fields_preview": sorted(observed_fields)[:30],
                "flags": {
                    "inspected": inspected,
                    "sampled": sampled,
                    "profiled": profiled,
                    "document_window_observed": window_observed,
                    "record_set_observed": record_set_observed,
                    "unsupported": unsupported,
                    "bound": bool(binding_refs),
                },
            }
        )
    return {
        "source_count": len(rows),
        "status_counts": status_counts,
        "sources": rows,
    }


def auto_requirement_skeleton(state: LoopState) -> dict[str, Any]:
    """Return a generic checklist prompt when the model has not declared one.

    The checklist deliberately avoids parsing task/domain content. It is a
    context nudge for the model to maintain coverage; it is not a binding and
    does not make final audit pass by itself.
    """
    if state.requirements:
        return {
            "active": False,
            "reason": "model_declared_requirements_present",
            "declared_requirement_ids": list(state.requirements),
            "suggested_requirements": [],
        }
    suggested = [
        {
            "kind": "source_evidence",
            "text": "Identify every real source needed for the answer and observe its data form before binding.",
        },
        {
            "kind": "data_element_evidence",
            "text": "Verify every physical field, value, document window, or record set used by the answer with cited evidence.",
        },
        {
            "kind": "operation_evidence",
            "text": "Verify each computation, join, filter, aggregation, ordering, unit conversion, or direct extraction before final.",
        },
        {
            "kind": "output_contract",
            "text": "Verify that final columns, rows, and lineage match the requested answer shape.",
        },
    ]
    if state.matched_sections:
        suggested.insert(
            1,
            {
                "kind": "knowledge_alignment",
                "text": "Use matched knowledge text for semantic alignment while keeping physical bindings grounded in observed evidence.",
            },
        )
    if any(source.data_form in DOCUMENT_FORMS for source in state.sources.values()):
        suggested.append(
            {
                "kind": "document_boundary",
                "text": "For PDF/MD evidence, use bounded windows or extracted records with provenance; do not treat documents as tables.",
            }
        )
    return {
        "active": True,
        "reason": "no_model_declared_requirements",
        "instruction": (
            "Call track_requirements when the task has multiple constraints, "
            "then update each item with evidence_refs, binding_refs, or compute_refs."
        ),
        "suggested_requirements": suggested,
    }


def final_output_contract(state: LoopState) -> dict[str, Any]:
    """Validate final answer shape and verifier consistency generically."""
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
            columns = list(compute.columns)
            rows = [list(row) for row in compute.rows]
            classification = compute_result_classifications(state).get(compute_ref)
            shape_operations = _compute_shape_operations(compute.sql)
            if classification is None:
                if shape_operations:
                    issues.append(
                        "shape_changing_compute_unverified:"
                        + ",".join(shape_operations)
                    )
                else:
                    warnings.append("compute_result_unverified_as_final_candidate")
            else:
                decision = str(classification.get("decision") or "")
                if decision in NON_ANSWER_DECISIONS:
                    issues.append(f"compute_result_classified_non_answer:{decision}")
                elif decision != "candidate_answer":
                    warnings.append(f"compute_result_not_final_candidate:{decision}")
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

    if columns is not None:
        normalized_columns = [str(column) for column in columns]
        if not normalized_columns:
            issues.append("empty_columns")
        blank_columns = [index for index, column in enumerate(normalized_columns) if not column.strip()]
        if blank_columns:
            issues.append("blank_column_name")
        duplicate_columns = sorted(
            column for column in set(normalized_columns)
            if normalized_columns.count(column) > 1
        )
        if duplicate_columns:
            issues.append("duplicate_columns:" + ",".join(duplicate_columns))
    else:
        normalized_columns = []

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
    classifications = compute_result_classifications(state)
    for result in state.compute_results.values():
        if not result.ok or not result.rows:
            continue
        classification = classifications.get(result.id)
        decision = str((classification or {}).get("decision") or "")
        is_final_candidate = decision == "candidate_answer"
        shape_operations = _compute_shape_operations(result.sql)
        if classification is None:
            kind = (
                "shape_changing_compute_needs_verification"
                if shape_operations
                else "ready_compute_final"
            )
        elif is_final_candidate:
            kind = "successful_compute"
        else:
            kind = "classified_non_answer_compute"
        candidates.append(
            {
                "kind": kind,
                "compute_ref": result.id,
                "columns": list(result.columns),
                "row_count": len(result.rows),
                "preview_rows": [list(row) for row in result.rows[:5]],
                "binding_refs": list(result.binding_refs),
                "evidence_refs": list(result.evidence_refs),
                "sql": result.sql,
                "shape_operations": shape_operations,
                "classification": classification,
                "verification_status": (
                    "requires_model_verification_for_shape_operations"
                    if classification is None and shape_operations
                    else (
                        "unverified_by_model"
                        if classification is None
                        else "classified_by_model"
                    )
                ),
                "allowed_next_action": {
                    "tool_name": "submit_final",
                    "arguments": {"compute_ref": result.id},
                } if (classification is None and not shape_operations) or is_final_candidate else {
                    "tool_name": "verify_alignment",
                    "arguments": {
                        "target_kind": "compute_result",
                        "compute_refs": [result.id],
                    },
                },
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
                "allowed_next_action": {
                    "tool_name": "submit_final",
                    "arguments": {
                        "answer": {},
                        "binding_refs": [binding.id],
                        "evidence_refs": list(binding.evidence_refs),
                    },
                },
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


def _positive_unbound_evidence(state: LoopState) -> list[dict[str, Any]]:
    positive_unbound: list[dict[str, Any]] = []
    bound_evidence = {
        evidence_ref
        for binding in state.bindings.values()
        for evidence_ref in binding.evidence_refs
    }
    for evidence in list(state.evidence.values())[-12:]:
        if evidence.tool_name in META_EVIDENCE_TOOLS:
            continue
        if not evidence.ok or evidence.id in bound_evidence:
            continue
        payload = evidence.payload or {}
        if evidence.negative_scope is not None:
            continue
        if evidence.tool_name == "run_verified_compute" and isinstance(payload.get("rows"), list):
            if not payload["rows"]:
                continue
        if evidence.tool_name == "search_values" and isinstance(payload.get("hits"), list):
            if not payload["hits"]:
                continue
        if evidence.tool_name == "search_document" and isinstance(payload.get("windows"), list):
            if not payload["windows"]:
                continue
        if evidence.tool_name == "extract_records" and isinstance(payload.get("records"), list):
            if not payload["records"]:
                continue
        has_positive_payload = (
            isinstance(payload.get("sample"), list)
            and bool(payload.get("sample"))
        ) or (
            isinstance(payload.get("hits"), list)
            and bool(payload.get("hits"))
        ) or (
            isinstance(payload.get("windows"), list)
            and bool(payload.get("windows"))
        ) or (
            isinstance(payload.get("records"), list)
            and bool(payload.get("records"))
        ) or (
            isinstance(payload.get("text"), str)
            and bool(payload.get("text").strip())
        ) or (
            isinstance(payload.get("columns"), list)
            and bool(payload.get("columns"))
        )
        if has_positive_payload:
            positive_unbound.append(
                {
                    "evidence_ref": evidence.id,
                    "tool_name": evidence.tool_name,
                    "summary": evidence.summary,
                    "source_id": evidence.source_id,
                    "data_form": evidence.data_form,
                }
            )
    return positive_unbound


def _binding_action_for_evidence(
    state: LoopState,
    evidence_ref: str,
    *,
    priority: int,
) -> dict[str, Any] | None:
    evidence = state.evidence.get(evidence_ref)
    if evidence is None or not evidence.ok:
        return None
    if any(evidence_ref in binding.evidence_refs for binding in state.bindings.values()):
        return None
    payload = evidence.payload or {}
    if evidence.data_form in STRUCTURED_FORMS and evidence.tool_name in {"inspect_source", "sample_records"}:
        table = str(payload.get("table") or "").strip() or None
        if _structured_binding_exists(state, evidence.source_id, table=table):
            return None
        columns = payload.get("columns")
        if not isinstance(columns, list) or not columns:
            sample = payload.get("sample")
            if isinstance(sample, list) and sample and isinstance(sample[0], dict):
                columns = list(sample[0])
        arguments: dict[str, Any] = {
            "binding_type": "structured_source",
            "source_ref": evidence.source_id,
            "evidence_refs": [evidence_ref],
            "alignment": (
                "Observed structured source can be registered as a verified relation; "
                "semantic use still depends on later compute/final lineage."
            ),
        }
        if table:
            arguments["table"] = table
        if isinstance(columns, list) and columns:
            arguments["allowed_columns"] = [str(column) for column in columns]
        return {
            "priority": priority,
            "reason": "A successful structured source observation is not bound; register it as a verified relation before compute.",
            "source": "observed_structured_source_binding",
            "evidence_ref": evidence_ref,
            "source_ref": evidence.source_id,
            "data_form": evidence.data_form,
            "tool_name": "bind",
            "arguments": arguments,
        }
    if evidence.data_form in DOCUMENT_FORMS and evidence.tool_name in {"search_document", "read_document_window"}:
        has_window = (
            isinstance(payload.get("text"), str)
            and bool(payload.get("text").strip())
        ) or (
            isinstance(payload.get("windows"), list)
            and bool(payload.get("windows"))
        )
        if has_window:
            return {
                "priority": priority,
                "reason": "A successful document window observation is not bound; bind it only as document evidence.",
                "source": "observed_document_window_binding",
                "evidence_ref": evidence_ref,
                "source_ref": evidence.source_id,
                "data_form": evidence.data_form,
                "tool_name": "bind",
                "arguments": {
                    "binding_type": "document_window",
                    "source_ref": evidence.source_id,
                    "evidence_refs": [evidence_ref],
                    "alignment": "Observed bounded document window may support direct evidence or extraction with provenance.",
                },
            }
    if evidence.tool_name == "extract_records" and isinstance(payload.get("records"), list) and payload["records"]:
        return {
            "priority": priority,
            "reason": "Extracted provenance-backed records are not bound; bind them as a document record set before compute.",
            "source": "observed_document_record_set_binding",
            "evidence_ref": evidence_ref,
            "source_ref": evidence.source_id,
            "data_form": evidence.data_form,
            "tool_name": "bind",
            "arguments": {
                "binding_type": "document_record_set",
                "source_ref": evidence.source_id,
                "evidence_refs": [evidence_ref],
                "alignment": "Observed extracted records include provenance and can be registered as a verified relation.",
            },
        }
    return None


def _verified_compute_refs(state: LoopState) -> set[str]:
    refs: set[str] = set()
    for decision in verifier_decisions(state):
        if decision.get("target_kind") != "compute_result":
            continue
        if decision.get("decision") not in {"candidate_answer", "intermediate", "not_applicable", "conflict"}:
            continue
        refs.update(str(ref) for ref in decision.get("compute_refs") or [])
        refs.update(
            str(ref)
            for ref in decision.get("target_refs") or []
            if str(ref).startswith("comp_")
        )
    return refs


def _recent_co_occurring_value_actions(state: LoopState, *, limit: int) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    searched_values = {
        str(evidence.payload.get("value") or "")
        for evidence in state.evidence.values()
        if evidence.tool_name == "search_values"
    }
    for evidence in reversed(list(state.evidence.values())):
        if evidence.tool_name != "search_values" or not evidence.ok:
            continue
        candidates = evidence.payload.get("co_occurring_value_candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            value = str(candidate.get("value") or "").strip()
            if not value or value in searched_values:
                continue
            key = (
                str(candidate.get("source_id") or ""),
                str(candidate.get("table") or ""),
                str(candidate.get("field") or ""),
                value,
            )
            if key in seen:
                continue
            seen.add(key)
            actions.append(
                {
                    "priority": 50 + min(int(candidate.get("support_count") or 1), 20),
                    "reason": "Verify a co-occurring scalar value observed in a real hit record before using it as an entity/key/value.",
                    "source": "co_occurring_value_candidate",
                    "candidate": {
                        "source_id": candidate.get("source_id"),
                        "path": candidate.get("path"),
                        "table": candidate.get("table"),
                        "field": candidate.get("field"),
                        "value": value,
                        "support_count": candidate.get("support_count"),
                        "requires_verification": True,
                    },
                    "tool_name": "search_values",
                    "arguments": {"value": value, "limit": 50},
                }
            )
            if len(actions) >= limit:
                return actions
    return actions


def _source_has_tool_evidence(
    state: LoopState,
    source_id: str,
    tool_names: set[str],
    *,
    table: str | None = None,
) -> bool:
    for evidence in state.evidence.values():
        if evidence.source_id != source_id or evidence.tool_name not in tool_names:
            continue
        if table and str(evidence.payload.get("table") or "") != table:
            continue
        return True
    return False


def _structured_binding_exists(
    state: LoopState,
    source_id: str | None,
    *,
    table: str | None = None,
) -> bool:
    if not source_id:
        return False
    source = state.sources.get(source_id)
    table_name = str(table or "").strip()
    for binding in state.bindings.values():
        if binding.binding_type not in {"structured_source", "structured_field"}:
            continue
        if binding.source_id != source_id:
            continue
        if source is not None and source.data_form in {"csv_records", "json_records"}:
            return True
        bound_table = str(binding.table or "").strip()
        if table_name:
            if bound_table == table_name:
                return True
            continue
        if not bound_table:
            return True
    return False


def _relation_inspection_actions(state: LoopState) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    inspected = {
        str(evidence.payload.get("binding_ref") or "")
        for evidence in state.evidence.values()
        if evidence.ok and evidence.tool_name == "inspect_relation"
    }
    for binding in state.bindings.values():
        if not binding.relation_name or binding.id in inspected:
            continue
        actions.append(
            {
                "priority": 86,
                "reason": "A verified relation exists but its executable relation shape has not been inspected in the ledger.",
                "source": "verified_relation_inspection",
                "binding_ref": binding.id,
                "relation_name": binding.relation_name,
                "tool_name": "inspect_relation",
                "arguments": {"binding_ref": binding.id},
            }
        )
    return actions


def _relation_compute_actions(state: LoopState) -> list[dict[str, Any]]:
    inspected: dict[str, dict[str, Any]] = {}
    for evidence in state.evidence.values():
        if not evidence.ok or evidence.tool_name != "inspect_relation":
            continue
        binding_ref = str(evidence.payload.get("binding_ref") or "")
        if binding_ref:
            inspected[binding_ref] = evidence.payload
    computed_bindings = {
        binding_ref
        for result in state.compute_results.values()
        if result.ok
        for binding_ref in result.binding_refs
    }
    actions: list[dict[str, Any]] = []
    for binding in state.bindings.values():
        if (
            not binding.relation_name
            or binding.id not in inspected
            or binding.id in computed_bindings
        ):
            continue
        payload = inspected[binding.id]
        actions.append(
            {
                "priority": 85,
                "reason": (
                    "A verified relation has been inspected and has not produced a compute result yet. "
                    "Write task-specific SQL over the verified relation name and columns."
                ),
                "source": "verified_relation_compute",
                "binding_ref": binding.id,
                "relation_name": binding.relation_name,
                "columns": list(payload.get("columns") or binding.allowed_columns),
                "tool_name": "run_verified_compute",
                "arguments": {"binding_refs": [binding.id]},
            }
        )
    return actions


def _source_action_for(
    state: LoopState,
    *,
    source_id: str,
    table: str | None = None,
    reason: str,
    priority: int,
    source_kind: str,
) -> dict[str, Any] | None:
    source = state.sources.get(source_id)
    if source is None:
        return None
    if source.data_form in STRUCTURED_FORMS:
        if _structured_binding_exists(state, source_id, table=table):
            return None
        if table and not _source_has_tool_evidence(
            state, source_id, {"inspect_source", "sample_records"}, table=table
        ):
            return {
                "priority": priority,
                "reason": reason,
                "source": source_kind,
                "source_ref": source_id,
                "data_form": source.data_form,
                "path": source.virtual_path,
                "table": table,
                "tool_name": "inspect_source",
                "arguments": {"source_ref": source_id, "table": table},
            }
        if not _source_has_tool_evidence(state, source_id, {"inspect_source", "sample_records"}):
            return {
                "priority": priority,
                "reason": reason,
                "source": source_kind,
                "source_ref": source_id,
                "data_form": source.data_form,
                "path": source.virtual_path,
                "tool_name": "inspect_source",
                "arguments": {"source_ref": source_id},
            }
        if not _source_has_tool_evidence(state, source_id, {"sample_records"}):
            args: dict[str, Any] = {"source_ref": source_id, "limit": 20}
            if table:
                args["table"] = table
            return {
                "priority": max(priority - 5, 1),
                "reason": "Source schema was observed; sample records to verify values, units, and row shape before binding.",
                "source": source_kind,
                "source_ref": source_id,
                "data_form": source.data_form,
                "path": source.virtual_path,
                "table": table,
                "tool_name": "sample_records",
                "arguments": args,
            }
        return None
    if source.data_form in DOCUMENT_FORMS:
        if not _source_has_tool_evidence(state, source_id, {"profile_document"}):
            return {
                "priority": priority,
                "reason": reason,
                "source": source_kind,
                "source_ref": source_id,
                "data_form": source.data_form,
                "path": source.virtual_path,
                "tool_name": "profile_document",
                "arguments": {"source_ref": source_id},
            }
        if not _source_has_tool_evidence(state, source_id, {"search_document", "read_document_window"}):
            return {
                "priority": max(priority - 5, 1),
                "reason": "Document source was profiled; search bounded windows before deciding whether evidence exists.",
                "source": source_kind,
                "source_ref": source_id,
                "data_form": source.data_form,
                "path": source.virtual_path,
                "tool_name": "search_document",
                "arguments": {"source_ref": source_id, "query": state.question, "limit": 10},
            }
        return None
    if source.data_form == "video":
        if not _source_has_tool_evidence(state, source_id, {"inspect_video"}):
            return {
                "priority": max(priority - 20, 1),
                "reason": "Video sources are unsupported in v1, but metadata can be inspected without binding final evidence.",
                "source": source_kind,
                "source_ref": source_id,
                "data_form": source.data_form,
                "path": source.virtual_path,
                "tool_name": "inspect_video",
                "arguments": {"source_ref": source_id},
            }
    return None


def _source_discovery_actions(state: LoopState, *, limit: int) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, str | None, str]] = set()
    for candidate in reversed(list(state.candidates.values())):
        if not candidate.source_id:
            continue
        key = (candidate.source_id, candidate.table, candidate.kind)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        action = _source_action_for(
            state,
            source_id=candidate.source_id,
            table=candidate.table,
            reason=(
                "A located source/table/field candidate has not been fully observed. "
                "Inspect or sample the real source before rejecting or binding it."
            ),
            priority=78,
            source_kind=f"unverified_{candidate.kind}",
        )
        if action is not None:
            action["candidate_ref"] = candidate.id
            actions.append(action)
            if len(actions) >= limit:
                return actions

    query_terms = _terms(state.question)
    for section in state.matched_sections[:6]:
        query_terms.update(_terms(section.heading_path))
        for mention in section.mentions:
            query_terms.update(_terms(mention))
    scored_sources: list[tuple[int, str]] = []
    for source in state.sources.values():
        if _source_has_tool_evidence(
            state,
            source.id,
            {
                "inspect_source",
                "sample_records",
                "profile_document",
                "search_document",
                "read_document_window",
                "inspect_video",
            },
        ):
            continue
        haystack = " ".join(
            [source.virtual_path, source.basename, source.stem, *source.tables, *source.columns]
        )
        overlap = len(query_terms & _terms(haystack))
        if overlap <= 0 and state.evidence:
            continue
        scored_sources.append((overlap, source.id))
    for overlap, source_id in sorted(scored_sources, key=lambda item: (-item[0], item[1])):
        action = _source_action_for(
            state,
            source_id=source_id,
            reason=(
                "Observed inventory source is relevant by question/knowledge token overlap "
                "but has not been inspected yet."
            ),
            priority=64 + min(overlap, 10),
            source_kind="unexplored_observed_source",
        )
        if action is not None:
            action["token_overlap"] = overlap
            actions.append(action)
            if len(actions) >= limit:
                return actions
    return actions


def next_action_schedule(state: LoopState, *, limit: int = 12) -> list[dict[str, Any]]:
    """Return bounded, generic next-action pressure derived from the ledger.

    This is not a planner and it does not encode task semantics. It mirrors the
    Codex pattern of feeding typed tool outcomes back to the model as compact,
    actionable state.
    """
    scheduled: list[dict[str, Any]] = []
    if not state.evidence:
        scheduled.extend(
            [
                {
                    "priority": 95,
                    "reason": "Start the evidence loop by locating sources for the question.",
                    "source": "bootstrap_source_discovery",
                    "tool_name": "locate_sources",
                    "arguments": {"query": state.question},
                },
                {
                    "priority": 90,
                    "reason": "Retrieve relevant knowledge sections before physical binding.",
                    "source": "bootstrap_knowledge",
                    "tool_name": "retrieve_knowledge",
                    "arguments": {"query": state.question},
                },
            ]
        )
    classified_computes = _verified_compute_refs(state)
    for result in state.compute_results.values():
        if not result.ok or not result.rows or result.id in classified_computes:
            continue
        shape_operations = _compute_shape_operations(result.sql)
        if shape_operations:
            scheduled.append(
                {
                    "priority": 88,
                    "reason": "A successful compute uses shape-changing SQL operations; verify that the ordering/limit/output shape is requested before final.",
                    "source": "unclassified_compute",
                    "compute_ref": result.id,
                    "columns": list(result.columns),
                    "row_count": len(result.rows),
                    "shape_operations": shape_operations,
                    "tool_name": "verify_alignment",
                    "arguments": {
                        "decision": "candidate_answer",
                        "target_kind": "compute_result",
                        "compute_refs": [result.id],
                        "binding_refs": list(result.binding_refs),
                        "evidence_refs": list(result.evidence_refs),
                        "alignment": "Classify whether this shape-changing compute result fully matches the requested final output.",
                    },
                }
            )
            continue
        scheduled.append(
            {
                "priority": 88,
                "reason": "A successful compute has verified binding lineage and non-empty rows; it can be submitted if it answers the question. Alignment verification is optional.",
                "source": "ready_compute_final",
                "compute_ref": result.id,
                "columns": list(result.columns),
                "row_count": len(result.rows),
                "binding_refs": list(result.binding_refs),
                "evidence_refs": list(result.evidence_refs),
                "tool_name": "submit_final",
                "arguments": {"compute_ref": result.id},
            }
        )
        scheduled.append(
            {
                "priority": 55,
                "reason": "Optionally classify this successful compute before final if the model is unsure.",
                "source": "unclassified_compute",
                "compute_ref": result.id,
                "columns": list(result.columns),
                "row_count": len(result.rows),
                "tool_name": "verify_alignment",
                "arguments": {
                    "decision": "candidate_answer",
                    "target_kind": "compute_result",
                    "compute_refs": [result.id],
                    "binding_refs": list(result.binding_refs),
                    "evidence_refs": list(result.evidence_refs),
                    "alignment": "Classify whether this compute result fully answers the user question.",
                },
            }
        )

    for evidence in _positive_unbound_evidence(state)[-8:]:
        bind_action = _binding_action_for_evidence(
            state,
            evidence["evidence_ref"],
            priority=92,
        )
        if bind_action is not None:
            scheduled.append(bind_action)

    scheduled.extend(_relation_inspection_actions(state))
    scheduled.extend(_relation_compute_actions(state))
    scheduled.extend(_source_discovery_actions(state, limit=limit))

    for candidate in answer_candidates(state)[-5:]:
        if candidate["kind"] not in {"successful_compute", "ready_compute_final"}:
            continue
        scheduled.append(
            {
                "priority": 90 if candidate["kind"] == "successful_compute" else 83,
                "reason": "A compute answer candidate exists; submit it if it satisfies the question, otherwise gather a specific missing evidence item.",
                "source": "answer_candidate" if candidate["kind"] == "successful_compute" else "ready_compute_final",
                "candidate": candidate,
                "tool_name": "submit_final",
                "arguments": {"compute_ref": candidate["compute_ref"]},
            }
        )

    for evidence in _positive_unbound_evidence(state)[-6:]:
        scheduled.append(
            {
                "priority": 58,
                "reason": "Positive observed evidence is not bound or rejected; verify whether it supports a source, field, value, record set, operation, or direct answer.",
                "source": "positive_unbound_evidence",
                "evidence": evidence,
                "tool_name": "verify_alignment",
                "arguments": {
                    "decision": "bindable",
                    "target_kind": "direct_evidence",
                    "evidence_refs": [evidence["evidence_ref"]],
                    "alignment": "Decide whether this observed evidence should be bound or rejected.",
                },
            }
        )

    scheduled.extend(_recent_co_occurring_value_actions(state, limit=limit))

    coverage = requirement_coverage(state)
    for requirement in coverage["pending_requirements"][:5]:
        scheduled.append(
            {
                "priority": 35,
                "reason": "A declared requirement is still pending. Treat this as context pressure; do not let it prevent source discovery or a lineage-backed compute final.",
                "source": "pending_requirement",
                "requirement": requirement,
                "tool_name": "verify_alignment",
                "arguments": {
                    "target_kind": "requirement",
                    "requirement_refs": [requirement["id"]],
                    "decision": "needs_more_evidence",
                    "alignment": requirement.get("note") or requirement.get("text") or "",
                },
            }
        )

    for evidence in reversed(list(state.evidence.values())):
        if len(scheduled) >= limit * 2:
            break
        for action in evidence.recommended_next_actions:
            if not isinstance(action, dict):
                continue
            scheduled.append(
                {
                    "priority": 40,
                    "reason": action.get("reason") or "Tool returned a recommended next action.",
                    "source": f"tool_recommendation:{evidence.id}",
                    "tool_name": action.get("tool_name"),
                    "arguments": action.get("arguments") or {},
                }
            )

    unique: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for item in sorted(scheduled, key=lambda entry: -int(entry.get("priority") or 0)):
        signature = json.dumps(
            {
                "tool_name": item.get("tool_name"),
                "arguments": item.get("arguments") or {},
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def _is_recovery_schedule_item(item: dict[str, Any]) -> bool:
    tool_name = str(item.get("tool_name") or "")
    if not tool_name or tool_name == "blocked":
        return False
    source = str(item.get("source") or "")
    return (
        source in ACTIONABLE_RECOVERY_SOURCES
        or source.startswith("unverified_")
        or source.startswith("tool_recommendation:")
    )


def actionable_recovery_schedule(state: LoopState, *, limit: int = 8) -> list[dict[str, Any]]:
    """Actions that should be tried before declaring no-progress blocked.

    Pending requirements are deliberately excluded here. They are useful
    pressure in context, but they are not a source frontier, positive observed
    evidence, or a ready compute result.
    """
    actions = [
        item for item in next_action_schedule(state, limit=limit * 2)
        if _is_recovery_schedule_item(item)
    ]
    return actions[:limit]


def recovery_hints(state: LoopState, *, limit: int = 1) -> list[RecoveryHint]:
    """Return the small recovery surface used by the production loop.

    This is intentionally narrower than ``next_action_schedule``. It is only
    used after a failed/no-progress turn, mirroring Codex's recoverable tool
    output path: the next model call sees one concrete repair direction, while
    normal turns remain model-driven.
    """
    actions = actionable_recovery_schedule(state, limit=max(limit * 4, 4))
    hints: list[RecoveryHint] = []
    for item in actions:
        tool_name = str(item.get("tool_name") or "").strip()
        if not tool_name:
            continue
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        hints.append(
            RecoveryHint(
                tool_name=tool_name,
                arguments=dict(arguments),
                reason=str(item.get("reason") or "Use this ledger-backed recovery action."),
                source=str(item.get("source") or "recovery"),
                priority=int(item.get("priority") or 0),
            )
        )
        if len(hints) >= limit:
            break
    if hints:
        return hints
    return [
        RecoveryHint(
            tool_name="blocked",
            arguments={
                "reason": (
                    "No verified answer, actionable source frontier, positive unbound evidence, "
                    "or repairable compute remains."
                )
            },
            reason="Stop only when the ledger has no remaining recovery hint.",
            source="recovery_exhausted",
            priority=0,
        )
    ][:limit]


def _is_negative_like_evidence(state: LoopState, evidence_ref: str) -> bool:
    evidence = state.evidence.get(evidence_ref)
    if evidence is None:
        return False
    if not evidence.ok or evidence.negative_scope is not None:
        return True
    payload = evidence.payload or {}
    if evidence.tool_name == "search_values" and isinstance(payload.get("hits"), list):
        return len(payload["hits"]) == 0
    if evidence.tool_name == "search_document" and isinstance(payload.get("windows"), list):
        return len(payload["windows"]) == 0
    if evidence.tool_name == "extract_records" and isinstance(payload.get("records"), list):
        return len(payload["records"]) == 0
    if evidence.tool_name == "run_verified_compute" and isinstance(payload.get("rows"), (list, tuple)):
        return len(payload["rows"]) == 0
    if evidence.tool_name == "verify_alignment":
        return payload.get("decision") in {"not_applicable", "conflict", "blocked_ok"}
    return False


def blocked_audit(
    state: LoopState,
    reason: str,
    *,
    cited_evidence_refs: tuple[str, ...] = (),
) -> dict[str, Any]:
    candidates = answer_candidates(state)
    hints = [hint.to_dict() for hint in recovery_hints(state, limit=8)]
    issues: list[str] = []
    recommendations: list[dict[str, Any]] = []
    cited = {ref for ref in cited_evidence_refs if ref in state.evidence}
    unknown_cited = [ref for ref in cited_evidence_refs if ref not in state.evidence]
    cited_negative = sorted(ref for ref in cited if _is_negative_like_evidence(state, ref))
    has_negative_context = bool(cited_negative or state.negative_scopes)

    compute_candidates = [
        item for item in candidates
        if item["kind"] in {"successful_compute", "ready_compute_final"}
    ]
    direct_candidates = [item for item in candidates if item["kind"] == "direct_evidence_binding"]
    if compute_candidates:
        issues.append("successful_compute_available")
        recommendations.extend(
            {
                "tool_name": "submit_final",
                "arguments": {"compute_ref": item["compute_ref"]},
                "reason": "Submit this compute_ref if it answers the question; otherwise gather specific missing evidence before blocking.",
            }
            for item in compute_candidates[-3:]
        )
    if direct_candidates:
        issues.append("direct_evidence_binding_available")
        recommendations.extend(
            {
                "tool_name": "submit_final",
                "arguments": item["allowed_next_action"]["arguments"],
                "reason": "Submit a direct answer if the verified evidence binding fully supports it.",
            }
            for item in direct_candidates[-3:]
        )

    decisive_schedule = [
        item for item in hints
        if item.get("tool_name") not in {None, "blocked"}
        and (
            item.get("source") in {
                "ready_compute_final",
                "unclassified_compute",
                "answer_candidate",
                "positive_unbound_evidence",
                "co_occurring_value_candidate",
                "unexplored_observed_source",
            }
            or str(item.get("source") or "").startswith("unverified_")
        )
    ]
    cited_blocked_ok = any(
        state.evidence[ref].tool_name == "verify_alignment"
        and state.evidence[ref].payload.get("decision") == "blocked_ok"
        for ref in cited
        if ref in state.evidence
    )
    if decisive_schedule and not cited_blocked_ok:
        issues.append("actionable_schedule_available")
        recommendations.extend(decisive_schedule[:3])

    positive_unbound = _positive_unbound_evidence(state)
    uncited_positive = [
        item for item in positive_unbound if item["evidence_ref"] not in cited
    ]
    if uncited_positive and not candidates and not has_negative_context:
        issues.append("positive_unbound_evidence_available")
        recommendations.append(
            {
                "source": "positive_unbound_evidence",
                "tool_name": "bind",
                "arguments": {"evidence_refs": [uncited_positive[-1]["evidence_ref"]]},
                "reason": "Bind positive observed evidence if it supports a source, field, value, operation, document window, or answer candidate.",
            }
        )

    if unknown_cited:
        issues.append("blocked_cites_unknown_evidence")
    if not cited and state.evidence:
        issues.append("blocked_without_cited_evidence")
        recommendations.append(
            {
                "source": "blocked_repair",
                "tool_name": "blocked",
                "arguments": {"reason": reason, "evidence_refs": []},
                "reason": "If truly blocked, call blocked again with evidence_refs supporting the absence or conflict.",
            }
        )
    if cited and not has_negative_context and not candidates:
        issues.append("blocked_without_negative_evidence")
        recommendations.append(
            {
                "source": "negative_evidence_repair",
                "tool_name": "search_values",
                "arguments": {"query": state.question},
                "reason": "Produce scoped negative evidence before declaring the requested data absent.",
            }
        )
    if not cited and not state.negative_scopes and not candidates and not positive_unbound:
        issues.append("blocked_without_evidence_or_negative_scope")
        recommendations.append(
            {
                "source": "source_discovery_repair",
                "tool_name": "locate_sources",
                "arguments": {"query": state.question},
                "reason": "Find candidate sources before declaring blocked.",
            }
        )

    audit_key = json.dumps(
        {
            "issues": issues,
            "candidate_refs": [
                item.get("compute_ref") or item.get("binding_ref") for item in candidates[-8:]
            ],
            "positive_unbound": [item["evidence_ref"] for item in uncited_positive[-8:]],
            "cited_evidence_refs": sorted(cited),
            "cited_negative_evidence_refs": cited_negative,
            "unknown_cited": unknown_cited,
            "recovery_hints": [
                item.get("source") for item in decisive_schedule[:8]
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "passed": not issues,
        "reason": reason,
        "issues": issues,
        "cited_evidence_refs": sorted(cited),
        "cited_negative_evidence_refs": cited_negative,
        "unknown_cited_evidence_refs": unknown_cited,
        "answer_candidates": candidates[-10:],
        "positive_unbound_evidence": positive_unbound[-10:],
        "uncited_positive_unbound_evidence": uncited_positive[-10:],
        "exhaustion_status": exhaustion_status(state),
        "recovery_hints": hints,
        "recommended_next_actions": recommendations[:8],
        "audit_key": audit_key,
    }
