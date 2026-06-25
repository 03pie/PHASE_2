from __future__ import annotations

from typing import Any

from data_agent_baseline.evidence_agent.codex_loop.protocol import LoopState, RecoveryHint

DIRECT_FINAL_BINDING_TYPES = {"document_window", "value", "operation", "answer_candidate"}
META_EVIDENCE_TOOLS = {"verify_alignment", "track_requirements", "bind", "submit_final"}


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
        if decision.get("target_kind") != "compute_result":
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
    pending = [item for item in requirements if item.get("status") in {"pending", "", None}]
    satisfied = [item for item in requirements if item.get("status") == "satisfied"]
    conflicts = [item for item in requirements if item.get("status") == "conflict"]
    blocked = [item for item in requirements if item.get("status") == "blocked"]
    weak_satisfied = [
        item
        for item in satisfied
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
            classification = compute_result_classifications(state).get(compute_ref)
            if classification is None:
                issues.append("compute_result_unverified_as_final_candidate")
            else:
                decision = str(classification.get("decision") or "")
                if decision in {"intermediate", "not_applicable", "needs_more_evidence", "conflict"}:
                    issues.append(f"compute_result_classified_non_answer:{decision}")
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
    classifications = compute_result_classifications(state)
    for result in state.compute_results.values():
        if not result.ok or not result.rows:
            continue
        classification = classifications.get(result.id)
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
                "classification": classification,
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


def _positive_unbound_evidence(state: LoopState) -> list[dict[str, Any]]:
    bound_evidence = {
        evidence_ref
        for binding in state.bindings.values()
        for evidence_ref in binding.evidence_refs
    }
    output: list[dict[str, Any]] = []
    for evidence in state.evidence.values():
        if (
            evidence.tool_name in META_EVIDENCE_TOOLS
            or not evidence.ok
            or evidence.id in bound_evidence
            or evidence.negative_scope is not None
        ):
            continue
        payload = evidence.payload or {}
        positive = any(
            isinstance(payload.get(key), list) and bool(payload.get(key))
            for key in ("sample", "hits", "matches", "slice_matches", "slice_catalog", "records", "columns")
        ) or (isinstance(payload.get("text"), str) and bool(payload["text"].strip()))
        if positive:
            output.append(
                {
                    "evidence_ref": evidence.id,
                    "tool_name": evidence.tool_name,
                    "summary": evidence.summary,
                    "source_id": evidence.source_id,
                    "data_form": evidence.data_form,
                }
            )
    return output


def recovery_hints(state: LoopState, *, limit: int = 1) -> list[RecoveryHint]:
    hints: list[RecoveryHint] = []
    classifications = compute_result_classifications(state)
    for result in reversed(list(state.compute_results.values())):
        if result.ok and result.rows:
            if result.id in classifications and classifications[result.id].get("decision") == "candidate_answer":
                hints.append(
                    RecoveryHint(
                        tool_name="submit_final",
                        arguments={"compute_ref": result.id, "answer": {"columns": list(result.columns)}},
                        reason="A compute result was verified as a candidate answer; submit it with explicit final columns if the columns are exactly the requested output.",
                        source="verified_compute_candidate",
                        priority=100,
                    )
                )
            else:
                hints.append(
                    RecoveryHint(
                        tool_name="verify_alignment",
                        arguments={
                            "decision": "candidate_answer",
                            "target_kind": "compute_result",
                            "compute_refs": [result.id],
                            "binding_refs": list(result.binding_refs),
                            "evidence_refs": list(result.evidence_refs),
                            "alignment": "<explain whether this exact compute result satisfies the question and knowledge>",
                        },
                        reason="A successful compute result exists but is not verified as final; classify it before submit_final.",
                        source="unverified_compute",
                        priority=95,
                    )
                )
            break
    if not hints:
        for binding in reversed(list(state.bindings.values())):
            if binding.binding_type in DIRECT_FINAL_BINDING_TYPES:
                hints.append(
                    RecoveryHint(
                        tool_name="submit_final",
                        arguments={
                            "binding_refs": [binding.id],
                            "evidence_refs": list(binding.evidence_refs),
                            "answer": {},
                        },
                        reason="A direct evidence binding exists; submit an explicit answer if fully supported.",
                        source="direct_evidence_binding",
                        priority=90,
                    )
                )
                break
    return hints[:limit]


def _is_negative_like_evidence(state: LoopState, evidence_ref: str) -> bool:
    evidence = state.evidence.get(evidence_ref)
    if evidence is None:
        return False
    if not evidence.ok or evidence.negative_scope is not None:
        return True
    payload = evidence.payload or {}
    if evidence.tool_name == "search_values" and isinstance(payload.get("hits"), list):
        return len(payload["hits"]) == 0
    if evidence.tool_name == "search_document" and isinstance(payload.get("matches"), list):
        return len(payload["matches"]) == 0
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
    issues: list[str] = []
    recommendations: list[dict[str, Any]] = []
    cited = {ref for ref in cited_evidence_refs if ref in state.evidence}
    unknown_cited = [ref for ref in cited_evidence_refs if ref not in state.evidence]
    cited_negative = sorted(ref for ref in cited if _is_negative_like_evidence(state, ref))
    has_negative_context = bool(cited_negative or state.negative_scopes)

    compute_candidates = [item for item in candidates if item["kind"] == "compute_result"]
    direct_candidates = [item for item in candidates if item["kind"] == "direct_evidence_binding"]
    if compute_candidates:
        issues.append("successful_compute_available")
        recommendations.extend(
            {
                "tool_name": "submit_final",
                "arguments": {"compute_ref": item["compute_ref"]},
                "reason": "Submit this compute_ref if it answers the task.",
            }
            for item in compute_candidates[-3:]
        )
    if direct_candidates:
        issues.append("direct_evidence_binding_available")
        recommendations.extend(
            {
                "tool_name": "submit_final",
                "arguments": {
                    "binding_refs": [item["binding_ref"]],
                    "evidence_refs": item.get("evidence_refs") or [],
                    "answer": {},
                },
                "reason": "Submit a direct answer if the verified evidence binding fully supports it.",
            }
            for item in direct_candidates[-3:]
        )

    positive_unbound = _positive_unbound_evidence(state)
    uncited_positive = [
        item for item in positive_unbound if item["evidence_ref"] not in cited
    ]
    if uncited_positive and not candidates and not has_negative_context:
        issues.append("positive_unbound_evidence_available")

    if unknown_cited:
        issues.append("blocked_cites_unknown_evidence")
    if not cited and state.evidence:
        issues.append("blocked_without_cited_evidence")
    if cited and not has_negative_context and not candidates:
        issues.append("blocked_without_negative_evidence")
    if not cited and not state.negative_scopes and not candidates and not positive_unbound:
        issues.append("blocked_without_evidence_or_negative_scope")

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
        "recovery_hints": [hint.to_dict() for hint in recovery_hints(state, limit=3)],
        "recommended_next_actions": recommendations[:8],
        "audit_key": "|".join(issues + sorted(cited) + unknown_cited),
    }
