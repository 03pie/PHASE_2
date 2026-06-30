from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from typing import Any

import duckdb
import pandas as pd

from data_agent_baseline.evidence_agent.codex_loop.protocol import Binding, LoopState

_CJK_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "兩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CJK_NUMBER_CHARS = "零〇一二两兩三四五六七八九十廿"


def _parse_year_token(value: str) -> int | None:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{4}", text):
        return int(text)
    if not re.fullmatch(r"[零〇一二两兩三四五六七八九]{4}", text):
        return None
    digits = [_CJK_DIGITS.get(char) for char in text]
    if any(digit is None for digit in digits):
        return None
    return int("".join(str(digit) for digit in digits))


def _parse_month_day_token(value: str) -> int | None:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{1,2}", text):
        return int(text)
    if not text:
        return None
    text = text.replace("廿", "二十")
    if "十" in text:
        left, right = text.split("十", 1)
        tens = _CJK_DIGITS.get(left, 1) if left else 1
        ones = _CJK_DIGITS.get(right, 0) if right else 0
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    digits = [_CJK_DIGITS.get(char) for char in text]
    if any(digit is None for digit in digits):
        return None
    return int("".join(str(digit) for digit in digits))


def _date_key(year: int | None, month: int | None, day: int | None) -> int | None:
    if year is None or month is None:
        return None
    day = day if day is not None else 1
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return year * 10000 + month * 100 + day


def parse_date_key(value: Any) -> int | None:
    """Normalize common date strings to an integer key suitable for sorting."""
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(
        r"\b(?P<year>\d{4})\s*[-/.]\s*(?P<month>\d{1,2})(?:\s*[-/.]\s*(?P<day>\d{1,2}))?\b",
        text,
    )
    if match:
        return _date_key(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")) if match.group("day") else None,
        )
    match = re.search(r"\b(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})\b", text)
    if match:
        return _date_key(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    cjk_or_digit = rf"\d{{1,2}}|[{_CJK_NUMBER_CHARS}]{{1,4}}"
    match = re.search(
        rf"(?P<year>\d{{4}}|[零〇一二两兩三四五六七八九]{{4}})\s*年\s*"
        rf"(?P<month>{cjk_or_digit})\s*月(?:\s*(?P<day>{cjk_or_digit})\s*(?:日|号)?)?",
        text,
    )
    if not match:
        return None
    return _date_key(
        _parse_year_token(match.group("year")),
        _parse_month_day_token(match.group("month")),
        _parse_month_day_token(match.group("day") or "1"),
    )


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


def _register_compute_functions(connection: duckdb.DuckDBPyConnection) -> None:
    connection.create_function(
        "parse_date_key",
        parse_date_key,
        return_type="BIGINT",
        null_handling="special",
    )


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
        _register_compute_functions(connection)
        for binding in bindings:
            frame = _load_binding_frame(state, binding)
            connection.register(binding.relation_name, frame)
            evidence_refs.extend(binding.evidence_refs)
        result = connection.execute(sql).fetchdf()

    columns = tuple(str(column) for column in result.columns)
    rows = tuple(tuple(_cell(value) for value in row) for row in result.itertuples(index=False, name=None))
    return columns, rows, tuple(dict.fromkeys(evidence_refs))
