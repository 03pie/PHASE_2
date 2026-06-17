from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable


def normalize_answer_column(column: Any) -> str:
    if isinstance(column, Mapping):
        for key in ("name", "column", "field", "source_field"):
            value = column.get(key)
            if value is None or isinstance(value, (list, tuple, dict)):
                continue
            text = str(value).strip()
            if text:
                return text
    return str(column)


def normalize_answer_columns(columns: list[Any]) -> list[str]:
    return [normalize_answer_column(column) for column in columns]


def answer_value_hash(columns: list[str], rows: list[list[Any]]) -> str:
    payload = {
        "columns": columns,
        "rows": rows,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_aliases(path: str) -> set[str]:
    normalized = path.replace("\\", "/")
    aliases = {normalized}
    if "::" in normalized:
        aliases.add(normalized.split("::", 1)[0])
    return aliases


def _is_transform_plan(output_spec: dict[str, Any]) -> bool:
    return (
        output_spec.get("row_policy") == "transform"
        or bool(output_spec.get("transformations"))
    )


def _source_binding_path_groups(analysis_plan: dict[str, Any]) -> list[set[str]]:
    execution_spec = analysis_plan.get("execution_spec")
    if not isinstance(execution_spec, Mapping):
        return []
    groups: list[set[str]] = []
    for binding in execution_spec.get("source_bindings") or []:
        if not isinstance(binding, Mapping):
            continue
        raw_paths = binding.get("source_paths")
        if not isinstance(raw_paths, list):
            continue
        aliases = {
            alias
            for path in raw_paths
            for alias in _source_aliases(str(path or ""))
            if alias.strip()
        }
        if aliases:
            groups.append(aliases)
    return groups


def _source_binding_fields(analysis_plan: dict[str, Any]) -> set[str]:
    execution_spec = analysis_plan.get("execution_spec")
    if not isinstance(execution_spec, Mapping):
        return set()
    return {
        str(binding.get("source_field") or "").casefold()
        for binding in execution_spec.get("source_bindings") or []
        if isinstance(binding, Mapping)
        and str(binding.get("source_field") or "").strip()
    }


def _output_source_fields(analysis_plan: dict[str, Any]) -> set[str]:
    output_spec = analysis_plan.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return set()
    return {
        str(field or "").casefold()
        for column in output_spec.get("columns") or []
        if isinstance(column, Mapping)
        for field in column.get("source_fields") or []
        if str(field or "").strip()
    }


def _requires_execution_audit(
    output_spec: dict[str, Any],
    analysis_plan: dict[str, Any],
) -> bool:
    return _is_transform_plan(output_spec) or bool(
        _source_binding_path_groups(analysis_plan)
    )


def _validate_execution_audit(
    audit: Any,
    *,
    columns: list[str],
    rows: list[list[Any]],
    analysis_plan: dict[str, Any],
) -> str | None:
    if not isinstance(audit, dict):
        return "transform answers require an execution audit object."
    source_paths = audit.get("source_paths") or audit.get("sources")
    if not isinstance(source_paths, list) or not source_paths:
        return "execution audit must include non-empty source_paths."
    evidence = analysis_plan.get("evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}
    execution_spec = analysis_plan.get("execution_spec")
    if not isinstance(execution_spec, Mapping):
        execution_spec = {}
    allowed_sources = {
        alias
        for source in evidence.get("context_sources") or []
        if isinstance(source, dict)
        for alias in _source_aliases(str(source.get("path") or ""))
        if alias.strip()
    }
    allowed_sources.update(
        alias
        for source in execution_spec.get("sources") or []
        if isinstance(source, dict)
        for alias in _source_aliases(str(source.get("path") or ""))
        if alias.strip()
    )
    normalized_sources = {str(source).replace("\\", "/") for source in source_paths}
    if allowed_sources and not normalized_sources.issubset(allowed_sources):
        return (
            "execution audit source_paths must be declared in the active plan: "
            f"{sorted(normalized_sources - allowed_sources)}."
        )
    normalized_source_aliases = {
        alias
        for source in normalized_sources
        for alias in _source_aliases(source)
        if alias.strip()
    }
    for index, required_aliases in enumerate(
        _source_binding_path_groups(analysis_plan)
    ):
        if not normalized_source_aliases & required_aliases:
            return (
                "execution audit source_paths must satisfy active "
                f"source_bindings[{index}]: {sorted(required_aliases)}."
            )
    output_source_fields = _output_source_fields(analysis_plan)
    binding_fields = _source_binding_fields(analysis_plan)
    binding_path_aliases = {
        alias
        for group in _source_binding_path_groups(analysis_plan)
        for alias in group
    }
    if (
        output_source_fields
        and output_source_fields.issubset(binding_fields)
        and binding_path_aliases
        and not normalized_source_aliases.issubset(binding_path_aliases)
    ):
        return (
            "execution audit source_paths for source-bound-only outputs must "
            "not include unbound sources: "
            f"{sorted(normalized_source_aliases - binding_path_aliases)}."
        )
    operations = audit.get("operations")
    if not isinstance(operations, list) or not operations:
        return "execution audit must include non-empty operations."
    output_row_count = audit.get("output_row_count")
    if output_row_count is not None and output_row_count != len(rows):
        return (
            "execution audit output_row_count must match submitted row count "
            f"{len(rows)}."
        )
    output_hash = audit.get("output_hash") or audit.get("value_hash")
    if output_hash is None:
        return "execution audit must include output_hash for submitted rows."
    if output_hash != answer_value_hash(columns, rows):
        return "execution audit output_hash does not match submitted rows."
    return None


def validate_prepared_answer(
    columns: list[str],
    rows: list[list[Any]],
    analysis_plan: dict[str, Any],
    audit: dict[str, Any] | None = None,
) -> tuple[AnswerTable | None, str | None]:
    """Validate a prepared answer against the active plan contract."""

    output_spec = analysis_plan.get("output_spec") or {}
    if not isinstance(output_spec, Mapping):
        output_spec = {}
    expected_columns = [
        str(column.get("name") or "")
        for column in output_spec.get("columns") or []
        if isinstance(column, dict)
    ]
    if not columns or not all(isinstance(column, str) and column for column in columns):
        return (
            None,
            "answer.columns must be a non-empty list of non-empty strings.",
        )
    if expected_columns and len(columns) < len(expected_columns):
        return (
            None,
            (
                "answer must include at least as many value columns as "
                f"analysis_plan.output_spec.columns ({len(expected_columns)}). "
                "Column names are not a scoring boundary; aliases and redundant "
                "columns are allowed."
            ),
        )
    if not rows:
        return None, "answer.rows must contain at least one row."

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if len(row) != len(columns):
            return None, "Each answer row must match the number of columns."
        normalized_rows.append(list(row))

    if _requires_execution_audit(dict(output_spec), analysis_plan):
        audit_error = _validate_execution_audit(
            audit,
            columns=columns,
            rows=normalized_rows,
            analysis_plan=analysis_plan,
        )
        if audit_error is not None:
            return None, audit_error

    expected_row_count = output_spec.get("expected_row_count")
    if expected_row_count is not None and len(normalized_rows) != expected_row_count:
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

    return (
        AnswerTable(columns=list(columns), rows=normalized_rows),
        None,
    )
