from __future__ import annotations

from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable


def validate_prepared_answer(
    columns: list[str],
    rows: list[list[Any]],
    analysis_plan: dict[str, Any],
) -> tuple[AnswerTable | None, str | None]:
    """校验内存结果是否满足计划契约，并构造最终答案。"""

    output_spec = analysis_plan.get("output_spec") or {}
    expected_columns = [
        str(column.get("name") or "")
        for column in output_spec.get("columns") or []
        if isinstance(column, dict)
    ]
    if columns != expected_columns:
        return (
            None,
            (
                "answer.columns must exactly match analysis_plan.output_spec: "
                f"{expected_columns}."
            ),
        )
    expected_row_count = output_spec.get("expected_row_count")
    if expected_row_count is not None and len(rows) != expected_row_count:
        detail = ""
        if (
            output_spec.get("row_policy") == "preserve"
            and not output_spec.get("transformations")
        ):
            detail = (
                " The active plan preserves observed source rows with no "
                "authorized transformations; submit one answer row per observed "
                "source record instead of aggregating, filtering, sorting, "
                "deduplicating, or changing expected_row_count to fit this answer."
            )
        return (
            None,
            (
                "answer row count must match "
                f"analysis_plan.output_spec.expected_row_count={expected_row_count}."
                f"{detail}"
            ),
        )

    if not columns or not all(isinstance(column, str) and column for column in columns):
        return (
            None,
            "answer.columns must be a non-empty list of non-empty strings.",
        )
    if not rows:
        return None, "answer.rows must contain at least one row."

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if len(row) != len(columns):
            return None, "Each answer row must match the number of columns."
        normalized_rows.append(list(row))

    return (
        AnswerTable(columns=list(columns), rows=normalized_rows),
        None,
    )
