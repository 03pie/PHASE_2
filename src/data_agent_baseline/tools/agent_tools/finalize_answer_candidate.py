from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from data_agent_baseline.agents.deep_state import BenchmarkDeepAgentState
from data_agent_baseline.tools.answer import (
    answer_value_hash,
    normalize_answer_columns,
    validate_prepared_answer,
)


def _coerce_optional_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError("expected a list or JSON-encoded list")


def _source_aliases(path: str) -> set[str]:
    normalized = path.replace("\\", "/")
    aliases = {normalized}
    if "::" in normalized:
        aliases.add(normalized.split("::", 1)[0])
    return aliases


def _plan_is_transform(analysis_plan: dict[str, Any]) -> bool:
    output_spec = analysis_plan.get("output_spec") or {}
    return (
        isinstance(output_spec, dict)
        and (
            output_spec.get("row_policy") == "transform"
            or bool(output_spec.get("transformations"))
        )
    )


def _plan_has_source_bindings(analysis_plan: dict[str, Any]) -> bool:
    execution_spec = analysis_plan.get("execution_spec") or {}
    return (
        isinstance(execution_spec, dict)
        and isinstance(execution_spec.get("source_bindings"), list)
        and bool(execution_spec["source_bindings"])
    )


def _plan_operation_labels(analysis_plan: dict[str, Any]) -> list[Any]:
    execution_spec = analysis_plan.get("execution_spec") or {}
    if isinstance(execution_spec, dict):
        operations = execution_spec.get("operations")
        if isinstance(operations, list) and operations:
            return operations
    output_spec = analysis_plan.get("output_spec") or {}
    if isinstance(output_spec, dict):
        transformations = output_spec.get("transformations")
        if isinstance(transformations, list) and transformations:
            return transformations
    return []


def _declared_plan_sources(analysis_plan: dict[str, Any]) -> set[str]:
    declared: set[str] = set()
    for section_name in ("evidence", "execution_spec"):
        section = analysis_plan.get(section_name) or {}
        if not isinstance(section, dict):
            continue
        key = "context_sources" if section_name == "evidence" else "sources"
        for source in section.get(key) or []:
            if not isinstance(source, dict):
                continue
            path = str(source.get("path") or "").strip()
            if path:
                declared.update(_source_aliases(path))
    return declared


def _synthesize_candidate_audit(
    *,
    analysis_plan: dict[str, Any],
    candidate: dict[str, Any],
    columns: list[str],
    rows: list[list[Any]],
) -> dict[str, Any] | None:
    if not (_plan_is_transform(analysis_plan) or _plan_has_source_bindings(analysis_plan)):
        return None
    operations = _plan_operation_labels(analysis_plan)
    if not operations and not _plan_has_source_bindings(analysis_plan):
        return None
    raw_paths = candidate.get("code_context_paths")
    if not isinstance(raw_paths, list):
        return None
    declared_sources = _declared_plan_sources(analysis_plan)
    source_paths = sorted(
        str(path).replace("\\", "/")
        for path in raw_paths
        if str(path).strip()
        and (not declared_sources or _source_aliases(str(path)) & declared_sources)
    )
    if not source_paths:
        return None
    return {
        "source_paths": source_paths,
        "operations": operations or ["source_bound_projection"],
        "output_row_count": len(rows),
        "output_hash": answer_value_hash(columns, rows),
        "audit_origin": "answer_candidate_static_context_paths",
    }


@tool(
    "finalize_answer_candidate",
    description=(
        "Submit a previously saved answer_candidate after an execute_python "
        "set_answer failure. It can only project existing candidate columns by "
        "index and optionally rename those projected columns before running the "
        "same answer validator. column_indexes and columns may be lists or "
        "JSON-encoded lists."
    ),
)
def finalize_answer_candidate_tool(
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    column_indexes: Any = None,
    columns: Any = None,
) -> Command[BenchmarkDeepAgentState] | ToolMessage:
    candidate = state.get("answer_candidate")
    if not isinstance(candidate, dict):
        return ToolMessage(
            content="No answer_candidate is available to finalize.",
            name="finalize_answer_candidate",
            tool_call_id=tool_call_id,
            status="error",
        )
    candidate_columns = candidate.get("columns")
    candidate_rows = candidate.get("rows")
    if not isinstance(candidate_columns, list) or not isinstance(candidate_rows, list):
        return ToolMessage(
            content="answer_candidate is malformed: expected columns and rows lists.",
            name="finalize_answer_candidate",
            tool_call_id=tool_call_id,
            status="error",
        )
    source_columns = normalize_answer_columns(candidate_columns)
    if not all(isinstance(row, list) for row in candidate_rows):
        return ToolMessage(
            content="answer_candidate is malformed: expected every row to be a list.",
            name="finalize_answer_candidate",
            tool_call_id=tool_call_id,
            status="error",
        )
    source_rows = [list(row) for row in candidate_rows]
    try:
        coerced_indexes = _coerce_optional_list(column_indexes)
        coerced_columns = _coerce_optional_list(columns)
        indexes = (
            list(range(len(source_columns)))
            if coerced_indexes is None
            else [int(index) for index in coerced_indexes]
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return ToolMessage(
            content="column_indexes must be integer indexes.",
            name="finalize_answer_candidate",
            tool_call_id=tool_call_id,
            status="error",
        )
    if not indexes:
        return ToolMessage(
            content="column_indexes must select at least one candidate column.",
            name="finalize_answer_candidate",
            tool_call_id=tool_call_id,
            status="error",
        )
    if any(index < 0 or index >= len(source_columns) for index in indexes):
        return ToolMessage(
            content=(
                "column_indexes must refer to existing answer_candidate columns "
                f"0..{len(source_columns) - 1}."
            ),
            name="finalize_answer_candidate",
            tool_call_id=tool_call_id,
            status="error",
        )
    projected_columns = [source_columns[index] for index in indexes]
    if coerced_columns is not None:
        if len(coerced_columns) != len(indexes):
            return ToolMessage(
                content="columns length must match column_indexes length.",
                name="finalize_answer_candidate",
                tool_call_id=tool_call_id,
                status="error",
            )
        projected_columns = normalize_answer_columns(coerced_columns)
    projected_rows = [[row[index] for index in indexes] for row in source_rows]
    analysis_plan = state.get("analysis_plan") or {}
    audit = candidate.get("audit") if isinstance(candidate.get("audit"), dict) else None
    if audit is None and isinstance(analysis_plan, dict):
        audit = _synthesize_candidate_audit(
            analysis_plan=analysis_plan,
            candidate=candidate,
            columns=projected_columns,
            rows=projected_rows,
        )
    if audit is not None:
        audit = dict(audit)
        operations = audit.get("operations")
        if isinstance(operations, list):
            audit["operations"] = [
                *operations,
                {"operation": "candidate_projection", "column_indexes": indexes},
            ]
        audit["output_row_count"] = len(projected_rows)
        audit["output_hash"] = answer_value_hash(projected_columns, projected_rows)
    prepared_answer, answer_error = validate_prepared_answer(
        projected_columns,
        projected_rows,
        analysis_plan,
        audit,
    )
    if answer_error is not None:
        return ToolMessage(
            content=answer_error,
            name="finalize_answer_candidate",
            tool_call_id=tool_call_id,
            status="error",
        )
    assert prepared_answer is not None
    summary = json.dumps(
        {
            "status": "prepared_from_candidate",
            "column_count": len(prepared_answer.columns),
            "row_count": len(prepared_answer.rows),
        },
        ensure_ascii=False,
    )
    return Command(
        update={
            "prepared_answer": prepared_answer,
            "answer_candidate": None,
            "messages": [
                ToolMessage(
                    content=summary,
                    name="finalize_answer_candidate",
                    tool_call_id=tool_call_id,
                    status="success",
                )
            ],
        }
    )
