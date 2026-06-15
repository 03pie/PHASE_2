from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.tools._helpers import (
    count_csv_rows,
    error,
    jsonable,
    resolve_context_path,
    success,
    virtual_path,
)


def create_read_csv_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """Create a paginated CSV preview tool."""

    context_root = (workspace / "context").resolve()

    @tool("read_csv")
    def read_csv(
        path: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        start_row: int = 0,
        max_rows: int = 50,
    ) -> Any:
        """Read a CSV preview with columns, dtypes, paging, and null-safe rows."""

        resolved, path_error = resolve_context_path(
            context_root,
            path,
            allowed_suffixes={".csv"},
        )
        if path_error:
            return error(
                name="read_csv",
                tool_call_id=tool_call_id,
                message=path_error,
                max_output_bytes=config.max_output_bytes,
            )
        if start_row < 0 or max_rows < 1:
            return error(
                name="read_csv",
                tool_call_id=tool_call_id,
                message="start_row >= 0 and max_rows >= 1 required.",
                max_output_bytes=config.max_output_bytes,
            )

        try:
            import pandas as pd

            assert resolved is not None
            total_rows = count_csv_rows(resolved)
            if start_row > 0:
                frame = pd.read_csv(
                    resolved,
                    skiprows=range(1, start_row + 1),
                    nrows=max_rows,
                )
            else:
                frame = pd.read_csv(resolved, nrows=max_rows)
            rows = [[jsonable(value) for value in row] for row in frame.values.tolist()]
            return success(
                name="read_csv",
                tool_call_id=tool_call_id,
                payload={
                    "path": virtual_path(resolved, context_root),
                    "columns": [str(column) for column in frame.columns.tolist()],
                    "dtypes": {
                        str(column): str(dtype) for column, dtype in frame.dtypes.items()
                    },
                    "rows": rows,
                    "total_rows": total_rows,
                    "returned_rows": len(rows),
                    "start_row": start_row,
                    "has_more": start_row + len(rows) < total_rows,
                },
                max_output_bytes=config.max_output_bytes,
            )
        except Exception as exc:
            return error(
                name="read_csv",
                tool_call_id=tool_call_id,
                message=f"Failed to read CSV: {exc}",
                max_output_bytes=config.max_output_bytes,
            )

    return read_csv
