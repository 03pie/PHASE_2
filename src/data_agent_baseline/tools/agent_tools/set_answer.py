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


def _coerce_json_like(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in {"[", "{"}:
            return json.loads(stripped)
    return value


def _normalize_rows(
    columns: list[str],
    rows: list[Any],
) -> tuple[list[list[Any]] | None, str | None]:
    if all(isinstance(row, dict) for row in rows):
        return [
            [row.get(column) for column in columns]
            for row in rows
        ], None
    if all(isinstance(row, list) for row in rows):
        return [list(row) for row in rows], None
    return None, "set_answer rows must contain only row lists or row objects."


def _normalize_submission(
    *,
    columns: Any,
    rows: Any,
    audit: Any,
) -> tuple[list[str] | None, list[list[Any]] | None, dict[str, Any] | None, str | None]:
    try:
        coerced_columns = _coerce_json_like(columns)
        coerced_rows = _coerce_json_like(rows)
        coerced_audit = _coerce_json_like(audit)
    except json.JSONDecodeError as exc:
        return None, None, None, f"set_answer received invalid JSON arguments: {exc}."

    if not isinstance(coerced_columns, list) or not isinstance(coerced_rows, list):
        return None, None, None, "set_answer requires list columns and list rows."
    normalized_columns = normalize_answer_columns(coerced_columns)
    normalized_rows, row_error = _normalize_rows(normalized_columns, coerced_rows)
    if row_error is not None:
        return None, None, None, row_error
    assert normalized_rows is not None

    normalized_audit = coerced_audit if isinstance(coerced_audit, dict) else None
    if normalized_audit is not None:
        normalized_audit = dict(normalized_audit)
        normalized_audit["output_row_count"] = len(normalized_rows)
        normalized_audit["output_hash"] = answer_value_hash(
            normalized_columns,
            normalized_rows,
        )
    return normalized_columns, normalized_rows, normalized_audit, None


def _audit_context_paths(audit: dict[str, Any] | None) -> list[str]:
    if not isinstance(audit, dict):
        return []
    raw_paths = audit.get("source_paths") or audit.get("sources")
    if not isinstance(raw_paths, list):
        return []
    return [
        str(path).replace("\\", "/")
        for path in raw_paths
        if str(path).strip() and not str(path).lower().endswith("/knowledge.md")
    ]


def _candidate_payload(
    *,
    columns: list[str],
    rows: list[list[Any]],
    audit: dict[str, Any] | None,
    validation_error: str,
) -> dict[str, Any]:
    return {
        "columns": columns,
        "rows": rows,
        "audit": audit,
        "column_count": len(columns),
        "row_count": len(rows),
        "code_context_paths": _audit_context_paths(audit),
        "validation_error": validation_error,
    }


@tool(
    "set_answer",
    description=(
        "Submit the final answer table directly after a successful analysis_plan "
        "and write_todos. columns and rows may be lists or JSON-encoded lists. "
        "For transform plans include audit.source_paths and audit.operations; "
        "output_row_count and output_hash are stamped by the tool."
    ),
)
def set_answer_tool(
    columns: Any,
    rows: Any,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    audit: Any = None,
) -> Command[BenchmarkDeepAgentState] | ToolMessage:
    normalized_columns, normalized_rows, normalized_audit, normalize_error = (
        _normalize_submission(columns=columns, rows=rows, audit=audit)
    )
    if normalize_error is not None:
        return ToolMessage(
            content=normalize_error,
            name="set_answer",
            tool_call_id=tool_call_id,
            status="error",
        )
    assert normalized_columns is not None
    assert normalized_rows is not None

    analysis_plan = state.get("analysis_plan")
    if not isinstance(analysis_plan, dict):
        answer_error = "set_answer requires a successful analysis_plan first."
        candidate = _candidate_payload(
            columns=normalized_columns,
            rows=normalized_rows,
            audit=normalized_audit,
            validation_error=answer_error,
        )
        return Command(
            update={
                "answer_candidate": candidate,
                "messages": [
                    ToolMessage(
                        content=answer_error,
                        name="set_answer",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                ],
            }
        )

    prepared_answer, answer_error = validate_prepared_answer(
        normalized_columns,
        normalized_rows,
        analysis_plan,
        normalized_audit,
    )
    if answer_error is not None:
        candidate = _candidate_payload(
            columns=normalized_columns,
            rows=normalized_rows,
            audit=normalized_audit,
            validation_error=answer_error,
        )
        candidate_summary = {
            "status": "candidate_saved",
            "column_count": len(normalized_columns),
            "row_count": len(normalized_rows),
            "validation_error": answer_error,
            "recovery": (
                "Revise analysis_plan to the submitted shape or call "
                "finalize_answer_candidate to submit a projected candidate table."
            ),
        }
        return Command(
            update={
                "answer_candidate": candidate,
                "messages": [
                    ToolMessage(
                        content=(
                            f"{answer_error}\n\n"
                            f"{json.dumps(candidate_summary, ensure_ascii=False)}"
                        ),
                        name="set_answer",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                ],
            }
        )

    assert prepared_answer is not None
    summary = json.dumps(
        {
            "status": "prepared",
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
                    name="set_answer",
                    tool_call_id=tool_call_id,
                    status="success",
                )
            ],
        }
    )
