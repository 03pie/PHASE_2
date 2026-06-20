from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.agents.semantic_layer import query_semantic_context
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools._helpers import (
    error,
    query_context_schema,
    quote_identifier,
    success,
    virtual_path,
)
from data_agent_baseline.tools.observed_sources import (
    observed_sources_command,
    sample_hash,
)


def _state_scope_terms(state: dict[str, Any]) -> list[str]:
    question_structure = state.get("question_structure")
    if not isinstance(question_structure, dict):
        return []
    terms: list[str] = []
    for item in question_structure.get("target_constraints") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("constraint_type") or "") not in {"scope", "entity"}:
            continue
        for key in ("value", "quote"):
            value = str(item.get(key) or "").strip()
            if value and value not in terms:
                terms.append(value)
    return terms


def _semantic_observed_sources(
    semantic_matches: dict[str, Any],
    *,
    field_text: str,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for candidate in semantic_matches.get("source_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        path = str(candidate.get("source_path") or "").replace("\\", "/")
        source_type = str(candidate.get("source_type") or "")
        if not path:
            continue
        if source_type == "doc":
            line_evidence = candidate.get("line_evidence")
            matched_lines = [
                {
                    "line_number": item.get("line_number"),
                    "content": item.get("content"),
                    **(
                        {"score": item.get("score")}
                        if isinstance(item.get("score"), int)
                        else {}
                    ),
                }
                for item in line_evidence
                if isinstance(item, dict)
            ][:10] if isinstance(line_evidence, list) else []
            sources.append(
                {
                    "path": path,
                    "source_type": "doc",
                    "source_name_hint": candidate.get("source_name_hint"),
                    "semantic_query": field_text,
                    "semantic_confidence": candidate.get("confidence"),
                    "query_match_type": candidate.get("query_match_type"),
                    "field_match": bool(candidate.get("field_match")),
                    "matched_fields": candidate.get("matched_fields") or [],
                    "match_reasons": candidate.get("match_reasons") or [],
                    "evidence_type": (
                        "line" if matched_lines else "source_candidate"
                    ),
                    "matched_lines": matched_lines,
                    "sample_hash": sample_hash(matched_lines),
                    "observed_by": "query_schema",
                }
            )
            continue
        fields = candidate.get("fields")
        if source_type in {"csv", "json", "sqlite"} and isinstance(fields, list | tuple):
            table_or_path = str(candidate.get("table_or_path") or "")
            observed_path = (
                f"{path}::{table_or_path}"
                if source_type == "sqlite" and table_or_path
                else path
            )
            query_match_type = str(candidate.get("query_match_type") or "")
            field_match = bool(candidate.get("field_match"))
            evidence_type = (
                "schema_field"
                if field_match
                else (
                    "knowledge_related_source"
                    if query_match_type == "knowledge_related"
                    else "schema_source"
                )
            )
            sources.append(
                {
                    "path": observed_path,
                    "source_type": (
                        "sqlite_table" if source_type == "sqlite" else source_type
                    ),
                    "base_path": path if source_type == "sqlite" else None,
                    "table": table_or_path if source_type == "sqlite" else None,
                    "source_name_hint": candidate.get("source_name_hint"),
                    "semantic_query": field_text,
                    "semantic_confidence": candidate.get("confidence"),
                    "query_match_type": query_match_type or None,
                    "field_match": field_match,
                    "matched_fields": candidate.get("matched_fields") or [],
                    "match_reasons": candidate.get("match_reasons") or [],
                    "evidence_type": evidence_type,
                    "fields": list(fields)[:80],
                    "sample_hash": sample_hash(
                        {
                            "path": observed_path,
                            "fields": list(fields)[:80],
                            "confidence": candidate.get("confidence"),
                            "query_match_type": query_match_type,
                            "matched_fields": candidate.get("matched_fields") or [],
                        }
                    ),
                    "observed_by": "query_schema",
                }
            )
    return sources


def _schema_match_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("source_type") or ""),
        str(item.get("path") or "").replace("\\", "/"),
        str(item.get("table") or ""),
        str(item.get("field") or item.get("column") or ""),
    )


def _merge_schema_matches(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for group in groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            key = _schema_match_key(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _knowledge_field_queries(
    semantic_matches: dict[str, Any],
    *,
    original_field: str,
) -> list[str]:
    normalized_original = original_field.casefold()
    queries: list[str] = []
    for fact in semantic_matches.get("knowledge_facts") or []:
        if not isinstance(fact, dict):
            continue
        field_key = str(fact.get("field_key") or "").strip()
        if not field_key:
            continue
        if field_key.casefold() == normalized_original:
            continue
        if field_key not in queries:
            queries.append(field_key)
    return queries[:8]


def _is_simple_schema_query(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"`?[A-Za-z_][A-Za-z0-9_.]*`?",
            value.strip(),
        )
    )


def _schema_observed_sources(
    matches: list[dict[str, Any]],
    *,
    field_text: str,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in matches:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").replace("\\", "/")
        source_type = str(item.get("source_type") or "")
        field = str(item.get("field") or item.get("column") or "").strip()
        if not path or not source_type:
            continue
        table = str(item.get("table") or "").strip()
        observed_path = (
            f"{path}::{table}"
            if source_type == "sqlite" and table
            else path
        )
        key = (observed_path, source_type, field)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "path": observed_path,
                "source_type": (
                    "sqlite_table" if source_type == "sqlite" else source_type
                ),
                "base_path": path if source_type == "sqlite" else None,
                "table": table if source_type == "sqlite" else None,
                "semantic_query": field_text,
                "query_match_type": "schema_field",
                "field_match": True,
                "matched_fields": [field] if field else [],
                "match_reasons": ["schema_match"],
                "evidence_type": "schema_field",
                "fields": [field] if field else [],
                "sample_hash": sample_hash(item),
                "observed_by": "query_schema",
            }
        )
    return sources


def _json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return [record for record in records if isinstance(record, dict)]
        for value in payload.values():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return list(value)
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    return []


def _value_matches(cell: Any, value_text: str) -> bool:
    cell_text = str(cell)
    return cell_text == value_text or (
        len(value_text) >= 3 and value_text.casefold() in cell_text.casefold()
    )


def _query_value_evidence(
    context_root: Path,
    value_text: str,
    *,
    max_matches: int,
) -> list[dict[str, Any]]:
    if not value_text:
        return []
    evidence: list[dict[str, Any]] = []
    for path in sorted(context_root.rglob("*.csv")):
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle)
                for row_index, row in enumerate(reader, start=1):
                    for column, cell in row.items():
                        if _value_matches(cell, value_text):
                            evidence.append(
                                {
                                    "source_type": "csv",
                                    "source_path": virtual_path(path, context_root),
                                    "field": column,
                                    "row_number": row_index,
                                    "sample_row": row,
                                }
                            )
                            break
                    if len(evidence) >= max_matches:
                        return evidence
        except (OSError, csv.Error):
            continue

    for path in sorted(context_root.rglob("*.json")):
        try:
            records = _json_records(json.loads(path.read_text(encoding="utf-8-sig")))
        except (OSError, json.JSONDecodeError):
            continue
        for row_index, row in enumerate(records, start=1):
            for column, cell in row.items():
                if _value_matches(cell, value_text):
                    evidence.append(
                        {
                            "source_type": "json",
                            "source_path": virtual_path(path, context_root),
                            "field": column,
                            "row_number": row_index,
                            "sample_row": row,
                        }
                    )
                    break
            if len(evidence) >= max_matches:
                return evidence

    for path in sorted(
        [
            *context_root.rglob("*.sqlite"),
            *context_root.rglob("*.db"),
            *context_root.rglob("*.sqlite3"),
        ]
    ):
        try:
            with sqlite3.connect(str(path)) as connection:
                connection.row_factory = sqlite3.Row
                cursor = connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [str(row[0]) for row in cursor.fetchall()]
                for table in tables:
                    cursor.execute(f"PRAGMA table_info({quote_identifier(table)})")
                    columns = [str(row[1]) for row in cursor.fetchall()]
                    for column in columns:
                        sql = (
                            f"SELECT * FROM {quote_identifier(table)} "
                            f"WHERE CAST({quote_identifier(column)} AS TEXT) = ? LIMIT 3"
                        )
                        try:
                            cursor.execute(sql, (value_text,))
                        except sqlite3.Error:
                            continue
                        rows = [dict(row) for row in cursor.fetchall()]
                        for row in rows:
                            evidence.append(
                                {
                                    "source_type": "sqlite",
                                    "source_path": virtual_path(path, context_root),
                                    "table": table,
                                    "field": column,
                                    "sample_row": row,
                                }
                            )
                            if len(evidence) >= max_matches:
                                return evidence
        except sqlite3.Error:
            continue
    return evidence


def create_query_schema_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """Create a lightweight field lookup tool across context data sources."""

    context_root = (workspace / "context").resolve()

    @tool("query_schema", description=load_tool_prompt("query_schema"))
    def query_schema(
        field: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict[str, Any], InjectedState],
        scope: str | None = None,
        value: str | None = None,
        max_matches: int = 25,
    ) -> Any:
        """Run the query_schema tool."""

        field_text = field.strip()
        if not field_text:
            return error(
                name="query_schema",
                tool_call_id=tool_call_id,
                message="field must be non-empty.",
                max_output_bytes=config.max_output_bytes,
            )
        if max_matches < 1:
            return error(
                name="query_schema",
                tool_call_id=tool_call_id,
                message="max_matches must be positive.",
                max_output_bytes=config.max_output_bytes,
            )

        scope_terms = [scope.strip()] if isinstance(scope, str) and scope.strip() else []
        if not scope_terms:
            scope_terms = _state_scope_terms(state)
        semantic_matches = query_semantic_context(
            context_root,
            field_text,
            max_matches=max_matches,
            scope=scope_terms,
        )
        direct_matches = query_context_schema(
            context_root,
            field_text,
            max_matches=max_matches,
        )
        expanded_field_queries = (
            []
            if _is_simple_schema_query(field_text)
            else _knowledge_field_queries(
                semantic_matches,
                original_field=field_text,
            )
        )
        expanded_matches: list[dict[str, Any]] = []
        per_query_limit = max(1, max_matches // max(1, len(expanded_field_queries) or 1))
        for query in expanded_field_queries:
            for item in query_context_schema(
                context_root,
                query,
                max_matches=per_query_limit,
            ):
                expanded_matches.append({**item, "matched_query": query})
        matches = _merge_schema_matches(expanded_matches, direct_matches)[:max_matches]
        value_terms = []
        explicit_value = str(value or "").strip()
        if explicit_value:
            value_terms.append(explicit_value)
        elif scope_terms:
            value_terms.extend(scope_terms)
        value_evidence: list[dict[str, Any]] = []
        seen_value_evidence: set[tuple[str, str, str, str]] = set()
        for value_term in value_terms:
            for item in _query_value_evidence(
                context_root,
                value_term,
                max_matches=max_matches,
            ):
                key = (
                    str(item.get("source_path") or ""),
                    str(item.get("table") or ""),
                    str(item.get("field") or ""),
                    str(item.get("row_number") or ""),
                )
                if key in seen_value_evidence:
                    continue
                seen_value_evidence.add(key)
                value_evidence.append({**item, "matched_value": value_term})
                if len(value_evidence) >= max_matches:
                    break
            if len(value_evidence) >= max_matches:
                break
        message = success(
            name="query_schema",
            tool_call_id=tool_call_id,
            payload={
                "field": field_text,
                "value": str(value or "").strip() or None,
                "value_search_terms": value_terms,
                "expanded_field_queries": expanded_field_queries,
                "scope": scope_terms,
                "matches": matches,
                "match_count": len(matches),
                "value_evidence": value_evidence,
                "source_candidates": semantic_matches["source_candidates"],
                "section_bindings": semantic_matches["section_bindings"],
                "binding_issues": semantic_matches["binding_issues"],
                "knowledge_facts": semantic_matches["knowledge_facts"],
                "hint": "Inspect the reported source before relying on a field.",
            },
            max_output_bytes=config.max_output_bytes,
        )
        sources = _semantic_observed_sources(
            semantic_matches,
            field_text=field_text,
        )
        sources = [
            *sources,
            *_schema_observed_sources(matches, field_text=field_text),
        ]
        for item in value_evidence:
            path = str(item.get("source_path") or "").replace("\\", "/")
            if not path:
                continue
            sources.append(
                {
                    "path": path,
                    "source_type": item.get("source_type"),
                    "table": item.get("table"),
                    "fields": [item.get("field")] if item.get("field") else [],
                    "value_evidence": [item],
                    "sample_hash": sample_hash(item),
                    "observed_by": "query_schema",
                }
            )
        if not sources:
            return message
        return observed_sources_command(
            state=state,
            message=message,
            sources=sources,
        )

    return query_schema
