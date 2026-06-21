from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from typing import Any

import duckdb
import pandas as pd

from data_agent_baseline.evidence_agent.codex_loop.protocol import Binding, LoopState


def _json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if all(isinstance(item, dict) for item in payload):
            return list(payload)
        return [{"value": item} for item in payload]
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return list(value)
        return [payload]
    return [{"value": payload}]


def _load_binding_frame(state: LoopState, binding: Binding) -> pd.DataFrame:
    if binding.binding_type == "document_record_set":
        records = binding.metadata.get("records")
        if not isinstance(records, list):
            records = []
        return pd.DataFrame(records)

    source = state.sources.get(binding.source_id or "")
    if source is None:
        raise ValueError(f"Binding {binding.id} has no observed source.")

    if source.data_form == "csv_records":
        return pd.read_csv(source.path, dtype=object)
    if source.data_form == "json_records":
        payload = json.loads(source.path.read_text(encoding="utf-8", errors="replace"))
        return pd.json_normalize(_json_records(payload))
    if source.data_form == "sqlite_database":
        table = binding.table
        if not table:
            if len(source.tables) == 1:
                table = source.tables[0]
            else:
                raise ValueError(f"Binding {binding.id} does not identify a SQLite table.")
        uri = f"file:{source.path.resolve().as_posix()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            return pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
    raise ValueError(f"Data form {source.data_form} cannot be registered as a compute relation.")


def load_binding_frame(state: LoopState, binding: Binding) -> pd.DataFrame:
    return _load_binding_frame(state, binding)


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return str(value)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat") and not isinstance(value, (str, bytes)):
        return str(value)
    return value


def run_sql_over_bindings(
    state: LoopState,
    *,
    sql: str,
    binding_refs: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...], tuple[str, ...]]:
    if not binding_refs:
        binding_refs = tuple(state.bindings)
    bindings = []
    for binding_ref in binding_refs:
        binding = state.bindings.get(binding_ref)
        if binding is None:
            raise ValueError(f"Unknown binding_ref: {binding_ref}")
        if not binding.relation_name:
            raise ValueError(f"Binding {binding_ref} has no relation name.")
        bindings.append(binding)

    evidence_refs: list[str] = []
    with duckdb.connect(database=":memory:") as connection:
        for binding in bindings:
            frame = _load_binding_frame(state, binding)
            connection.register(binding.relation_name, frame)
            evidence_refs.extend(binding.evidence_refs)
        result = connection.execute(sql).fetchdf()

    columns = tuple(str(column) for column in result.columns)
    rows = tuple(tuple(_cell(value) for value in row) for row in result.itertuples(index=False, name=None))
    return columns, rows, tuple(dict.fromkeys(evidence_refs))
