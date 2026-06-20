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


def _column_aliases_from_plan_column(column: Mapping[str, Any]) -> set[str]:
    aliases = {str(column.get("name") or "").casefold()}
    aliases.update(
        str(field or "").casefold()
        for field in column.get("source_fields") or []
        if str(field or "").strip()
    )
    return {alias for alias in aliases if alias}


def _expected_output_aliases(analysis_plan: dict[str, Any]) -> set[str]:
    output_spec = analysis_plan.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return set()
    return {
        alias
        for column in output_spec.get("columns") or []
        if isinstance(column, Mapping)
        for alias in _column_aliases_from_plan_column(column)
    }


def _supporting_field_aliases(analysis_plan: dict[str, Any]) -> set[str]:
    execution_spec = analysis_plan.get("execution_spec")
    if not isinstance(execution_spec, Mapping):
        return set()
    aliases: set[str] = set()
    for field in execution_spec.get("supporting_fields") or []:
        if not isinstance(field, Mapping):
            continue
        aliases.add(str(field.get("name") or "").casefold())
        aliases.update(
            str(item or "").casefold()
            for item in field.get("source_fields") or []
            if str(item or "").strip()
        )
    return {alias for alias in aliases if alias}


def _normalized_alias(value: Any) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


def _plan_output_columns(analysis_plan: dict[str, Any]) -> list[Mapping[str, Any]]:
    output_spec = analysis_plan.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return []
    return [
        column
        for column in output_spec.get("columns") or []
        if isinstance(column, Mapping)
    ]


def _plan_column_aliases(column: Mapping[str, Any]) -> list[str]:
    aliases: list[str] = []
    for value in [column.get("name"), *(column.get("source_fields") or [])]:
        text = str(value or "").strip()
        if text and text not in aliases:
            aliases.append(text)
    return aliases


def _plan_column_value_aliases(column: Mapping[str, Any]) -> list[str]:
    source_fields = [
        str(field or "").strip()
        for field in column.get("source_fields") or []
        if str(field or "").strip()
    ]
    if source_fields:
        return list(dict.fromkeys(source_fields))
    name = str(column.get("name") or "").strip()
    return [name] if name else []


def _display_column_name(
    column: Mapping[str, Any],
    selected_key: str | None,
    analysis_plan: dict[str, Any] | None = None,
) -> str:
    name = str(column.get("name") or "").strip()
    source_fields = [
        str(field or "").strip()
        for field in column.get("source_fields") or []
        if str(field or "").strip()
    ]
    if selected_key is None and len(source_fields) == 1:
        source_field = source_fields[0]
        selector_aliases = (
            _selector_field_aliases(analysis_plan)
            if isinstance(analysis_plan, dict)
            else set()
        )
        output_column_count = (
            len(_plan_output_columns(analysis_plan))
            if isinstance(analysis_plan, dict)
            else 0
        )
        if (
            source_field
            and _normalized_alias(source_field) == _normalized_alias(name)
            and source_field != name
        ):
            return source_field
        if (
            source_field
            and _normalized_alias(source_field) != _normalized_alias(name)
            and (
                _normalized_alias(source_field) not in selector_aliases
                or output_column_count == 1
            )
        ):
            return source_field
    if not selected_key:
        return name
    selector_aliases = (
        _selector_field_aliases(analysis_plan)
        if isinstance(analysis_plan, dict)
        else set()
    )
    if (
        name
        and _normalized_alias(selected_key) == _normalized_alias(name)
        and len(source_fields) == 1
        and _normalized_alias(source_fields[0]) not in selector_aliases
    ):
        return source_fields[0]
    if name and name != selected_key and selected_key in source_fields:
        return selected_key
    return name or selected_key


def _selector_field_aliases(analysis_plan: dict[str, Any] | None) -> set[str]:
    if not isinstance(analysis_plan, dict):
        return set()
    output_spec = analysis_plan.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return set()
    aliases = {
        _normalized_alias(item.get("field"))
        for item in output_spec.get("sort_keys") or []
        if isinstance(item, Mapping)
        and str(item.get("field") or "").strip()
    }
    execution_spec = analysis_plan.get("execution_spec")
    if isinstance(execution_spec, Mapping):
        for field in execution_spec.get("supporting_fields") or []:
            if not isinstance(field, Mapping):
                continue
            if str(field.get("purpose") or "").casefold() != "selector":
                continue
            aliases.add(_normalized_alias(field.get("name")))
            aliases.update(
                _normalized_alias(item)
                for item in field.get("source_fields") or []
                if str(item or "").strip()
            )
    return {alias for alias in aliases if alias}


def _select_dict_key_for_column(
    value: Mapping[str, Any],
    column: Mapping[str, Any],
) -> str | None:
    keys = [str(key) for key in value.keys()]
    if not keys:
        return None
    key_by_alias = {_normalized_alias(key): key for key in keys}
    if len(keys) == 1:
        return keys[0]
    aliases = _plan_column_value_aliases(column)
    matched_keys: list[str] = []
    for alias in aliases:
        key = key_by_alias.get(_normalized_alias(alias))
        if key is not None and key not in matched_keys:
            matched_keys.append(key)
    if len(matched_keys) == 1:
        return matched_keys[0]
    return None


def _merge_row_dict_cells(row: list[Any]) -> Mapping[str, Any] | None:
    merged: dict[str, Any] = {}
    found = False
    for cell in row:
        if not isinstance(cell, Mapping):
            continue
        found = True
        for key, value in cell.items():
            text_key = str(key)
            if text_key not in merged:
                merged[text_key] = value
    return merged if found else None


def _project_structured_cells(
    *,
    columns: list[str],
    rows: list[list[Any]],
    analysis_plan: dict[str, Any],
    audit: dict[str, Any] | None,
) -> tuple[list[str], list[list[Any]], dict[str, Any] | None, str | None]:
    if not any(any(isinstance(cell, Mapping) for cell in row) for row in rows):
        return columns, rows, audit, None
    plan_columns = _plan_output_columns(analysis_plan)
    if not plan_columns:
        return (
            columns,
            rows,
            audit,
            (
                "answer rows contain structured object cells but the active plan "
                "does not define output columns for projection."
            ),
        )
    projected_columns: list[str] = []
    selected_keys: list[str] = []
    for column in plan_columns:
        selected_key: str | None = None
        for row in rows:
            merged = _merge_row_dict_cells(row)
            if merged is None:
                continue
            selected_key = _select_dict_key_for_column(merged, column)
            if selected_key is not None:
                break
        if selected_key is None:
            available_keys = sorted(
                {
                    str(key)
                    for row in rows
                    if (merged := _merge_row_dict_cells(row)) is not None
                    for key in merged.keys()
                }
            )
            return (
                columns,
                rows,
                audit,
                (
                    "answer structured cells could not be uniquely projected to "
                    f"analysis_plan output column {column.get('name')!r}; "
                    f"available object keys: {available_keys}."
                ),
            )
        selected_keys.append(selected_key)
        projected_columns.append(
            _display_column_name(column, selected_key, analysis_plan)
        )

    projected_rows: list[list[Any]] = []
    for row in rows:
        merged = _merge_row_dict_cells(row)
        if merged is None:
            if len(row) == len(projected_columns):
                projected_rows.append(list(row))
                continue
            return (
                columns,
                rows,
                audit,
                "answer rows mix structured object cells with incompatible scalar rows.",
            )
        projected_rows.append([merged.get(key) for key in selected_keys])

    projected_audit = dict(audit) if isinstance(audit, dict) else audit
    if isinstance(projected_audit, dict):
        projected_audit["output_row_count"] = len(projected_rows)
        projected_audit["output_hash"] = answer_value_hash(
            projected_columns,
            projected_rows,
        )
        projected_audit["projection"] = {
            "from_structured_cell_keys": selected_keys,
        }
    return projected_columns, projected_rows, projected_audit, None


def _project_to_plan_columns(
    *,
    columns: list[str],
    rows: list[list[Any]],
    analysis_plan: dict[str, Any],
    audit: dict[str, Any] | None,
) -> tuple[list[str], list[list[Any]], dict[str, Any] | None]:
    plan_columns = _plan_output_columns(analysis_plan)
    if not plan_columns:
        return columns, rows, audit
    if (
        isinstance(audit, Mapping)
        and isinstance(audit.get("projection"), Mapping)
        and audit["projection"].get("from_structured_cell_keys")
    ):
        return columns, rows, audit
    if len(columns) == len(plan_columns):
        renamed = [
            _display_column_name(column, None, analysis_plan) or columns[index]
            for index, column in enumerate(plan_columns)
        ]
        if renamed == columns:
            return columns, rows, audit
        projected_audit = dict(audit) if isinstance(audit, dict) else audit
        if isinstance(projected_audit, dict):
            projected_audit["output_row_count"] = len(rows)
            projected_audit["output_hash"] = answer_value_hash(renamed, rows)
        return renamed, rows, projected_audit

    source_columns_by_alias: dict[str, int] = {}
    for index, column in enumerate(columns):
        source_columns_by_alias.setdefault(_normalized_alias(column), index)
    indexes: list[int] = []
    projected_columns: list[str] = []
    for plan_column in plan_columns:
        matched_index: int | None = None
        matched_key: str | None = None
        for alias in _plan_column_value_aliases(plan_column):
            index = source_columns_by_alias.get(_normalized_alias(alias))
            if index is not None:
                matched_index = index
                matched_key = columns[index]
                break
        if matched_index is None:
            return columns, rows, audit
        indexes.append(matched_index)
        projected_columns.append(
            _display_column_name(plan_column, matched_key, analysis_plan)
        )

    projected_rows = [[row[index] for index in indexes] for row in rows]
    projected_audit = dict(audit) if isinstance(audit, dict) else audit
    if isinstance(projected_audit, dict):
        projected_audit["output_row_count"] = len(projected_rows)
        projected_audit["output_hash"] = answer_value_hash(
            projected_columns,
            projected_rows,
        )
        projected_audit["projection"] = {
            "column_indexes": indexes,
            "from_columns": [columns[index] for index in indexes],
        }
    return projected_columns, projected_rows, projected_audit


def _project_supporting_columns(
    *,
    columns: list[str],
    rows: list[list[Any]],
    analysis_plan: dict[str, Any],
    audit: dict[str, Any] | None,
) -> tuple[list[str], list[list[Any]], dict[str, Any] | None]:
    expected_aliases = _expected_output_aliases(analysis_plan)
    supporting_aliases = _supporting_field_aliases(analysis_plan) - expected_aliases
    if not supporting_aliases:
        return columns, rows, audit
    drop_indexes = [
        index
        for index, column in enumerate(columns)
        if column.casefold() in supporting_aliases
    ]
    if not drop_indexes:
        return columns, rows, audit
    keep_indexes = [
        index for index in range(len(columns)) if index not in set(drop_indexes)
    ]
    output_spec = analysis_plan.get("output_spec")
    expected_count = 0
    if isinstance(output_spec, Mapping):
        expected_count = len(
            [
                column
                for column in output_spec.get("columns") or []
                if isinstance(column, Mapping)
            ]
        )
    if not keep_indexes or len(keep_indexes) < expected_count:
        return columns, rows, audit
    projected_columns = [columns[index] for index in keep_indexes]
    projected_rows = [[row[index] for index in keep_indexes] for row in rows]
    projected_audit = dict(audit) if isinstance(audit, dict) else audit
    if isinstance(projected_audit, dict):
        projected_audit["output_row_count"] = len(projected_rows)
        projected_audit["output_hash"] = answer_value_hash(
            projected_columns,
            projected_rows,
        )
    return projected_columns, projected_rows, projected_audit


def _row_sort_token(value: Any) -> tuple[int, Any, str]:
    if value is None or value == "":
        return (2, "", "")
    if isinstance(value, bool):
        return (0, int(value), str(value))
    if isinstance(value, (int, float)):
        return (0, float(value), str(value))
    text = str(value)
    return (1, text.casefold(), text)


def _unordered_aggregate_sort_indexes(
    *,
    columns: list[str],
    analysis_plan: dict[str, Any],
) -> list[int]:
    output_spec = analysis_plan.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return []
    if output_spec.get("sort_keys"):
        return []
    ordering = str(output_spec.get("ordering") or "").strip().casefold()
    if ordering not in {"", "none", "source", "unspecified"}:
        return []
    operations = _declared_plan_operations(analysis_plan)
    if "aggregate" not in operations or "sort" in operations:
        return []

    plan_columns = _plan_output_columns(analysis_plan)
    indexes: list[int] = []
    calculation_aliases = {
        "count",
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
    for index, column in enumerate(columns):
        plan_column = plan_columns[index] if index < len(plan_columns) else {}
        role = (
            str(plan_column.get("role") or "").strip().casefold()
            if isinstance(plan_column, Mapping)
            else ""
        )
        if role in {"calculation", "metric_value", "aggregate_value"}:
            continue
        if _normalized_alias(column) in calculation_aliases:
            continue
        indexes.append(index)
    return indexes or list(range(len(columns)))


def _stabilize_unordered_aggregate_rows(
    *,
    columns: list[str],
    rows: list[list[Any]],
    analysis_plan: dict[str, Any],
    audit: dict[str, Any] | None,
) -> tuple[list[list[Any]], dict[str, Any] | None]:
    indexes = _unordered_aggregate_sort_indexes(
        columns=columns,
        analysis_plan=analysis_plan,
    )
    if not indexes or len(rows) < 2:
        return rows, audit
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            tuple(_row_sort_token(row[index]) for index in indexes),
            tuple(_row_sort_token(value) for value in row),
        ),
    )
    if sorted_rows == rows:
        return rows, audit
    updated_audit = dict(audit) if isinstance(audit, dict) else audit
    if isinstance(updated_audit, dict):
        updated_audit["output_row_count"] = len(sorted_rows)
        updated_audit["output_hash"] = answer_value_hash(columns, sorted_rows)
        normalization = dict(updated_audit.get("normalization") or {})
        normalization["row_order"] = {
            "policy": "stable_unordered_aggregate",
            "sort_columns": [columns[index] for index in indexes],
        }
        updated_audit["normalization"] = normalization
    return sorted_rows, updated_audit


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
    declared_operations = _declared_plan_operations(analysis_plan)
    audit_operations = _audit_operation_names(operations)
    if declared_operations and not audit_operations.issubset(declared_operations):
        return (
            "execution audit operations must not exceed active plan operations: "
            f"{sorted(audit_operations - declared_operations)}."
        )
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


def _declared_plan_operations(analysis_plan: dict[str, Any]) -> set[str]:
    output_spec = analysis_plan.get("output_spec")
    execution_spec = analysis_plan.get("execution_spec")
    operations = {
        str(item.get("operation") or "").casefold()
        for item in (
            output_spec.get("transformations") if isinstance(output_spec, Mapping) else []
        )
        or []
        if isinstance(item, Mapping) and str(item.get("operation") or "").strip()
    }
    operations.update(
        str(item.get("operation") or "").casefold()
        for item in (
            execution_spec.get("operations") if isinstance(execution_spec, Mapping) else []
        )
        or []
        if isinstance(item, Mapping) and str(item.get("operation") or "").strip()
    )
    return {operation for operation in operations if operation}


def _audit_operation_names(operations: list[Any]) -> set[str]:
    return {
        str(item.get("operation") or "").casefold()
        for item in operations
        if isinstance(item, Mapping) and str(item.get("operation") or "").strip()
    }


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
    if not rows:
        return None, "answer.rows must contain at least one row."

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if len(row) != len(columns):
            return None, "Each answer row must match the number of columns."
        normalized_rows.append(list(row))

    structured_columns, structured_rows, structured_audit, structured_error = (
        _project_structured_cells(
            columns=columns,
            rows=normalized_rows,
            analysis_plan=analysis_plan,
            audit=audit,
        )
    )
    if structured_error is not None:
        return None, structured_error
    columns = structured_columns
    normalized_rows = structured_rows
    audit = structured_audit

    if any(any(isinstance(cell, Mapping) for cell in row) for row in normalized_rows):
        return (
            None,
            "answer rows must contain scalar cells after projection; object cells are not final CSV values.",
        )

    columns, normalized_rows, audit = _project_to_plan_columns(
        columns=columns,
        rows=normalized_rows,
        analysis_plan=analysis_plan,
        audit=audit,
    )

    columns, normalized_rows, audit = _project_supporting_columns(
        columns=columns,
        rows=normalized_rows,
        analysis_plan=analysis_plan,
        audit=audit,
    )
    columns, normalized_rows, audit = _project_to_plan_columns(
        columns=columns,
        rows=normalized_rows,
        analysis_plan=analysis_plan,
        audit=audit,
    )
    normalized_rows, audit = _stabilize_unordered_aggregate_rows(
        columns=columns,
        rows=normalized_rows,
        analysis_plan=analysis_plan,
        audit=audit,
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
