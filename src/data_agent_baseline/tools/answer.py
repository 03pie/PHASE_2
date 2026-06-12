from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from data_agent_baseline.agents.deep_state import BenchmarkDeepAgentState
from data_agent_baseline.benchmark.schema import AnswerTable


@tool("answer")
def answer_tool(
    columns: list[str],
    rows: list[list[Any]],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[BenchmarkDeepAgentState] | ToolMessage:
    """Submit the final answer table and terminate the benchmark task."""

    if not columns or not all(isinstance(column, str) and column for column in columns):
        return ToolMessage(
            content="answer.columns must be a non-empty list of non-empty strings.",
            name="answer",
            tool_call_id=tool_call_id,
            status="error",
        )
    if not rows:
        return ToolMessage(
            content="answer.rows must contain at least one row.",
            name="answer",
            tool_call_id=tool_call_id,
            status="error",
        )

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if len(row) != len(columns):
            return ToolMessage(
                content="Each answer row must match the number of columns.",
                name="answer",
                tool_call_id=tool_call_id,
                status="error",
            )
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    content = json.dumps(
        {
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        ensure_ascii=False,
    )
    return Command(
        update={
            "answer": answer,
            "messages": [
                ToolMessage(
                    content=content,
                    name="answer",
                    tool_call_id=tool_call_id,
                    status="success",
                )
            ],
        }
    )
