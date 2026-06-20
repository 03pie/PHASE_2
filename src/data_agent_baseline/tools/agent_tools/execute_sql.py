from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from data_agent_baseline.agents.deep_state import BenchmarkDeepAgentState
from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools.answer import answer_value_hash, validate_prepared_answer
from data_agent_baseline.tools.agent_tools.execute_python import _try_candidate_plan_execution
from data_agent_baseline.tools._helpers import (
    DB_SUFFIXES,
    empty_result_hint,
    error,
    is_readonly_sql,
    list_context_files,
    quote_identifier,
    resolve_context_path,
    success,
    virtual_path,
)
from data_agent_baseline.tools.observed_sources import (
    merge_observed_sources,
    sample_hash,
)


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


def _sql_result_audit(
    *,
    source_path: str,
    sql: str,
    columns: list[str],
    rows: list[list[Any]],
    analysis_plan: dict[str, Any],
) -> dict[str, Any] | None:
    if not (_plan_is_transform(analysis_plan) or _plan_has_source_bindings(analysis_plan)):
        return None
    operations = _plan_operation_labels(analysis_plan)
    if not operations and not _plan_has_source_bindings(analysis_plan):
        return None
    return {
        "source_paths": [source_path],
        "operations": operations or ["source_bound_projection"],
        "execution_tool": "execute_sql",
        "sql": sql,
        "output_row_count": len(rows),
        "output_hash": answer_value_hash(columns, rows),
        "audit_origin": "execute_sql_result",
    }


def _column_is_generic_calculation(column: dict[str, Any]) -> bool:
    role = str(column.get("role") or "").strip().casefold()
    names = {
        str(column.get("name") or "").strip().casefold(),
        *(
            str(field or "").strip().casefold()
            for field in column.get("source_fields") or []
            if str(field or "").strip()
        ),
    }
    aliases = {
        "count",
        "count(*)",
        "frequency",
        "freq",
        "percentage",
        "percent",
        "pct",
        "ratio",
        "sum",
        "avg",
        "average",
        "min",
        "max",
    }
    return role in {"calculation", "metric_value", "aggregate_value"} or bool(
        names & aliases
    )


def _sql_result_column_matches_expected(
    result_names: set[str],
    expected: dict[str, Any],
) -> bool:
    aliases = {
        str(expected.get("name") or "").casefold(),
        *(
            str(field or "").casefold()
            for field in expected.get("source_fields") or []
            if str(field or "").strip()
        ),
    }
    if result_names & {alias for alias in aliases if alias}:
        return True
    if not _column_is_generic_calculation(expected):
        return False
    calculation_result_aliases = {
        "count",
        "count(*)",
        "sum",
        "avg",
        "average",
        "min",
        "max",
        "percentage",
        "percent",
        "pct",
        "ratio",
    }
    return bool(result_names & calculation_result_aliases)


def _sql_result_matches_output_spec(
    columns: list[str],
    analysis_plan: dict[str, Any],
) -> bool:
    output_spec = analysis_plan.get("output_spec") or {}
    if not isinstance(output_spec, dict):
        return False
    expected_columns = output_spec.get("columns") or []
    if not isinstance(expected_columns, list) or not expected_columns:
        return False
    result_names = {str(column).casefold() for column in columns if str(column).strip()}
    if not result_names:
        return False
    for expected in expected_columns:
        if not isinstance(expected, dict):
            return False
        if not _sql_result_column_matches_expected(result_names, expected):
            return False
    return True


def _sql_result_candidate(
    *,
    source_path: str,
    audit_source_path: str,
    columns: list[str],
    rows: list[list[Any]],
    audit: dict[str, Any] | None,
    validation_error: str | None = None,
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "columns": list(columns),
        "rows": [list(row) for row in rows],
        "audit": audit,
        "column_count": len(columns),
        "row_count": len(rows),
        "code_context_paths": list(
            dict.fromkeys(path for path in (audit_source_path, source_path) if path)
        ),
        "source": "execute_sql",
    }
    if validation_error is not None:
        candidate["validation_error"] = validation_error
    return candidate


def _table_names(cursor: sqlite3.Cursor) -> list[str]:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [str(row[0]) for row in cursor.fetchall()]


def _observe_sql_reads(
    *,
    connection: sqlite3.Connection,
    tables: set[str],
    read_tables: set[str],
) -> None:
    def authorize(
        action_code: int,
        arg1: str | None,
        arg2: str | None,
        database_name: str | None,
        trigger_or_view: str | None,
    ) -> int:
        del database_name, trigger_or_view
        if action_code == sqlite3.SQLITE_READ and arg1 in tables:
            read_tables.add(str(arg1))
        elif action_code == sqlite3.SQLITE_PRAGMA and arg2 in tables:
            read_tables.add(str(arg2))
        return sqlite3.SQLITE_OK

    connection.set_authorizer(authorize)


def _sqlite_table_sources(
    *,
    cursor: sqlite3.Cursor,
    base_path: str,
    table_names: list[str],
    read_tables: set[str],
) -> list[dict[str, Any]]:
    columns_by_table: dict[str, list[str]] = {}
    for table_name in table_names:
        try:
            cursor.execute(f"PRAGMA table_info({quote_identifier(table_name)})")
            columns_by_table[table_name] = [str(row[1]) for row in cursor.fetchall()]
        except sqlite3.Error:
            columns_by_table[table_name] = []
    observed_tables = [
        table_name for table_name in table_names if table_name in read_tables
    ]
    field_to_tables: dict[str, list[str]] = {}
    for table_name in observed_tables:
        for field in columns_by_table.get(table_name, []):
            field_to_tables.setdefault(field.casefold(), [])
            if table_name not in field_to_tables[field.casefold()]:
                field_to_tables[field.casefold()].append(table_name)
    join_evidence = [
        {"field": field, "tables": tables}
        for field, tables in sorted(field_to_tables.items())
        if len(tables) > 1
    ][:20]
    sources: list[dict[str, Any]] = [
        {
            "path": base_path,
            "source_type": "sqlite",
            "tables": table_names,
            "observed_tables": observed_tables,
            "join_evidence": join_evidence,
            "observed_by": "execute_sql",
        }
    ]
    for table_name in table_names:
        if table_name not in read_tables:
            continue
        quoted = quote_identifier(table_name)
        table_source: dict[str, Any] = {
            "path": f"{base_path}::{table_name}",
            "source_type": "sqlite_table",
            "base_path": base_path,
            "table": table_name,
            "observed_by": "execute_sql",
        }
        try:
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
            fields = [str(column["name"]) for column in columns]
            cursor.execute(f"SELECT COUNT(*) FROM {quoted}")
            row_count = int(cursor.fetchone()[0])
            cursor.execute(f"SELECT * FROM {quoted} LIMIT 5")
            sample_rows = [dict(row) for row in cursor.fetchall()]
            table_source.update(
                {
                    "row_count": row_count,
                    "fields": fields,
                    "sample_hash": sample_hash(
                        {"columns": fields, "rows": sample_rows}
                    ),
                }
            )
        except sqlite3.Error:
            table_source["metadata_warning"] = "table metadata could not be read"
        sources.append(table_source)
    return sources


def _sql_command(
    *,
    state: dict[str, Any],
    message: ToolMessage,
    sources: list[dict[str, Any]],
    update: dict[str, Any] | None = None,
) -> Command[BenchmarkDeepAgentState]:
    command_update: dict[str, Any] = {
        "observed_sources": merge_observed_sources(
            state.get("observed_sources"),
            sources,
        ),
        "messages": [message],
    }
    if update:
        command_update.update(update)
    return Command(update=command_update)


def create_execute_sql_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """Create a read-only SQLite query tool."""

    context_root = (workspace / "context").resolve()

    @tool("execute_sql", description=load_tool_prompt("execute_sql"))
    def execute_sql(
        path: str,
        sql: str,
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        max_rows: int = 100,
    ) -> ToolMessage | Command[BenchmarkDeepAgentState]:
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
                table_names = _table_names(cursor)
                read_tables: set[str] = set()
                _observe_sql_reads(
                    connection=connection,
                    tables=set(table_names),
                    read_tables=read_tables,
                )
                try:
                    cursor.execute(sql)
                    rows = cursor.fetchmany(max_rows + 1)
                finally:
                    connection.set_authorizer(None)
                columns = [item[0] for item in cursor.description or []]
                has_more = len(rows) > max_rows
                rows = rows[:max_rows]
                result_columns = [str(column) for column in columns]
                result_rows = [list(row) for row in rows]
                source_path = virtual_path(resolved, context_root)
                observed_sources = _sqlite_table_sources(
                    cursor=cursor,
                    base_path=source_path,
                    table_names=table_names,
                    read_tables=read_tables,
                )
                payload: dict[str, Any] = {
                    "path": source_path,
                    "observed_tables": [
                        table_name
                        for table_name in table_names
                        if table_name in read_tables
                    ],
                    "columns": result_columns,
                    "rows": result_rows,
                    "row_count": len(rows),
                    "has_more": has_more,
                }
                if not rows:
                    hint = empty_result_hint(sql)
                    if hint:
                        payload["empty_result_hint"] = hint
                analysis_plan = state.get("analysis_plan")
                if (
                    isinstance(analysis_plan, dict)
                    and state.get("todos")
                    and result_rows
                    and not has_more
                    and _sql_result_matches_output_spec(
                        result_columns,
                        analysis_plan,
                    )
                ):
                    audit_source_path = next(
                        (
                            str(source.get("path") or "")
                            for source in observed_sources
                            if source.get("source_type") == "sqlite_table"
                            and str(source.get("table") or "") in read_tables
                            and str(source.get("path") or "").strip()
                        ),
                        source_path,
                    )
                    audit = _sql_result_audit(
                        source_path=audit_source_path,
                        sql=sql,
                        columns=result_columns,
                        rows=result_rows,
                        analysis_plan=analysis_plan,
                    )
                    candidate = _sql_result_candidate(
                        source_path=source_path,
                        audit_source_path=audit_source_path,
                        columns=result_columns,
                        rows=result_rows,
                        audit=audit,
                    )
                    candidate_plan_answer = _try_candidate_plan_execution(
                        workspace=workspace,
                        analysis_plan=analysis_plan,
                        candidate=candidate,
                    )
                    if candidate_plan_answer is not None:
                        payload["prepared_answer"] = {
                            "column_count": len(candidate_plan_answer.columns),
                            "row_count": len(candidate_plan_answer.rows),
                            "source": "candidate_plan_execution",
                        }
                        return _sql_command(
                            state=state,
                            message=success(
                                name="execute_sql",
                                tool_call_id=tool_call_id,
                                payload=payload,
                                max_output_bytes=config.max_output_bytes,
                            ),
                            sources=observed_sources,
                            update={
                                "prepared_answer": candidate_plan_answer,
                                "answer_candidate": None,
                            },
                        )
                    prepared_answer, answer_error = validate_prepared_answer(
                        result_columns,
                        result_rows,
                        analysis_plan,
                        audit,
                    )
                    if prepared_answer is not None and answer_error is None:
                        payload["prepared_answer"] = {
                            "column_count": len(prepared_answer.columns),
                            "row_count": len(prepared_answer.rows),
                        }
                        return _sql_command(
                            state=state,
                            message=success(
                                name="execute_sql",
                                tool_call_id=tool_call_id,
                                payload=payload,
                                max_output_bytes=config.max_output_bytes,
                            ),
                            sources=observed_sources,
                            update={
                                "prepared_answer": prepared_answer,
                                "answer_candidate": None,
                            },
                        )
                    payload["answer_candidate"] = {
                        "column_count": len(candidate["columns"]),
                        "row_count": len(candidate["rows"]),
                        "validation_error": answer_error,
                    }
                    candidate = _sql_result_candidate(
                        source_path=source_path,
                        audit_source_path=audit_source_path,
                        columns=result_columns,
                        rows=result_rows,
                        audit=audit,
                        validation_error=answer_error,
                    )
                    return _sql_command(
                        state=state,
                        message=success(
                            name="execute_sql",
                            tool_call_id=tool_call_id,
                            payload=payload,
                            max_output_bytes=config.max_output_bytes,
                        ),
                        sources=observed_sources,
                        update={"answer_candidate": candidate},
                    )
                return _sql_command(
                    state=state,
                    message=success(
                        name="execute_sql",
                        tool_call_id=tool_call_id,
                        payload=payload,
                        max_output_bytes=config.max_output_bytes,
                    ),
                    sources=observed_sources,
                )
        except sqlite3.Error as exc:
            return error(
                name="execute_sql",
                tool_call_id=tool_call_id,
                message=f"SQL error: {exc}",
                max_output_bytes=config.max_output_bytes,
            )

    return execute_sql
