from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools._helpers import (
    DB_SUFFIXES,
    empty_result_hint,
    error,
    is_readonly_sql,
    list_context_files,
    resolve_context_path,
    success,
    virtual_path,
)

def create_execute_sql_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """Create a read-only SQLite query tool."""

    context_root = (workspace / "context").resolve()

    @tool("execute_sql", description=load_tool_prompt("execute_sql"))
    def execute_sql(
        path: str,
        sql: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        max_rows: int = 100,
    ) -> Any:
        """Run the execute_sql tool."""

        if max_rows < 1:
            return error(
                name="execute_sql",
                tool_call_id=tool_call_id,
                message="max_rows must be positive.",
                max_output_bytes=config.max_output_bytes,
            )
        if not is_readonly_sql(sql):
            return error(
                name="execute_sql",
                tool_call_id=tool_call_id,
                message="Only read-only SQL is allowed: SELECT, WITH, PRAGMA, or EXPLAIN.",
                max_output_bytes=config.max_output_bytes,
            )

        resolved, path_error = resolve_context_path(
            context_root,
            path,
            allowed_suffixes=DB_SUFFIXES,
        )
        if path_error:
            databases = list_context_files(context_root, DB_SUFFIXES)
            if databases:
                path_error += f" Available databases: {', '.join(databases[:10])}."
            return error(
                name="execute_sql",
                tool_call_id=tool_call_id,
                message=path_error,
                max_output_bytes=config.max_output_bytes,
            )

        try:
            assert resolved is not None
            with sqlite3.connect(str(resolved)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA query_only = ON")
                cursor = connection.cursor()
                cursor.execute(sql)
                rows = cursor.fetchmany(max_rows + 1)
                columns = [item[0] for item in cursor.description or []]
                has_more = len(rows) > max_rows
                rows = rows[:max_rows]
                payload: dict[str, Any] = {
                    "path": virtual_path(resolved, context_root),
                    "columns": columns,
                    "rows": [list(row) for row in rows],
                    "row_count": len(rows),
                    "has_more": has_more,
                }
                if not rows:
                    hint = empty_result_hint(sql)
                    if hint:
                        payload["empty_result_hint"] = hint
                return success(
                    name="execute_sql",
                    tool_call_id=tool_call_id,
                    payload=payload,
                    max_output_bytes=config.max_output_bytes,
                )
        except sqlite3.Error as exc:
            return error(
                name="execute_sql",
                tool_call_id=tool_call_id,
                message=f"SQL error: {exc}",
                max_output_bytes=config.max_output_bytes,
            )

    return execute_sql
