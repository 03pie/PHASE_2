from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools._helpers import (
    DB_SUFFIXES,
    error,
    list_context_files,
    quote_identifier,
    resolve_context_path,
    success,
    virtual_path,
)
from data_agent_baseline.tools.observed_sources import (
    observed_sources_command,
    sample_hash,
)

def create_inspect_sqlite_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """Create a SQLite schema/sample inspection tool."""

    context_root = (workspace / "context").resolve()

    @tool("inspect_sqlite", description=load_tool_prompt("inspect_sqlite"))
    def inspect_sqlite(
        path: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict[str, Any], InjectedState],
        table: str = "",
        sample_rows: int = 5,
    ) -> Any:
        """Run the inspect_sqlite tool."""

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
                name="inspect_sqlite",
                tool_call_id=tool_call_id,
                message=path_error,
                max_output_bytes=config.max_output_bytes,
            )
        if sample_rows < 0:
            return error(
                name="inspect_sqlite",
                tool_call_id=tool_call_id,
                message="sample_rows cannot be negative.",
                max_output_bytes=config.max_output_bytes,
            )

        try:
            assert resolved is not None
            with sqlite3.connect(str(resolved)) as connection:
                connection.row_factory = sqlite3.Row
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                tables = [str(row[0]) for row in cursor.fetchall()]
                target_tables = [table] if table else tables
                result: dict[str, Any] = {
                    "path": virtual_path(resolved, context_root),
                    "mode": "table_detail" if table else "database_overview",
                    "table_count": len(tables),
                    "tables": {},
                }
                sources: list[dict[str, Any]] = [
                    {
                        "path": result["path"],
                        "source_type": "sqlite",
                        "tables": tables,
                        "observed_by": "inspect_sqlite",
                    }
                ]
                for table_name in target_tables:
                    if table_name not in tables:
                        continue
                    quoted = quote_identifier(table_name)
                    cursor.execute(f"PRAGMA table_info({quoted})")
                    columns = [
                        {
                            "name": row[1],
                            "type": row[2],
                            "nullable": not bool(row[3]),
                            "pk": row[5],
                        }
                        for row in cursor.fetchall()
                    ]
                    cursor.execute(f"SELECT COUNT(*) FROM {quoted}")
                    row_count = int(cursor.fetchone()[0])
                    table_info: dict[str, Any] = {
                        "columns": columns,
                        "row_count": row_count,
                    }
                    if table:
                        cursor.execute(f"SELECT * FROM {quoted} LIMIT ?", (sample_rows,))
                        table_info["sample_rows"] = [
                            dict(row) for row in cursor.fetchall()
                        ]
                        fields = [str(column["name"]) for column in columns]
                        sources.append(
                            {
                                "path": f"{result['path']}::{table_name}",
                                "source_type": "sqlite_table",
                                "base_path": result["path"],
                                "table": table_name,
                                "row_count": row_count,
                                "fields": fields,
                                "sample_hash": sample_hash(
                                    {
                                        "columns": fields,
                                        "rows": table_info["sample_rows"],
                                    }
                                ),
                                "observed_by": "inspect_sqlite",
                            }
                        )
                    result["tables"][table_name] = table_info
                if table and not result["tables"]:
                    result["warning"] = (
                        f"Table {table!r} not found. Available tables: {tables}."
                    )
                elif not table:
                    result["hint"] = (
                        "Pass table=<name> to inspect sample_rows for one table; "
                        "use execute_sql for targeted queries."
                    )
                message = success(
                    name="inspect_sqlite",
                    tool_call_id=tool_call_id,
                    payload=result,
                    max_output_bytes=config.max_output_bytes,
                )
                return observed_sources_command(
                    state=state,
                    message=message,
                    sources=sources,
                )
        except sqlite3.Error as exc:
            return error(
                name="inspect_sqlite",
                tool_call_id=tool_call_id,
                message=f"SQLite error: {exc}",
                max_output_bytes=config.max_output_bytes,
            )

    return inspect_sqlite
