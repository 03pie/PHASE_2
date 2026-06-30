from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from data_agent_baseline.evidence_agent.codex_loop.protocol import DataForm, SourceRef

_SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
_CSV_SUFFIXES = {".csv"}
_JSON_SUFFIXES = {".json"}
_PDF_SUFFIXES = {".pdf"}
_MD_SUFFIXES = {".md", ".markdown"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def virtual_path(path: Path, context_dir: Path) -> str:
    try:
        return "/context/" + path.relative_to(context_dir).as_posix()
    except ValueError:
        return path.as_posix()


def data_form_for_path(path: Path) -> DataForm:
    suffix = path.suffix.casefold()
    if suffix in _SQLITE_SUFFIXES:
        return "sqlite_database"
    if suffix in _CSV_SUFFIXES:
        return "csv_records"
    if suffix in _JSON_SUFFIXES:
        return "json_records"
    if suffix in _PDF_SUFFIXES:
        return "pdf_document"
    if suffix in _MD_SUFFIXES:
        return "markdown_document"
    if suffix in _VIDEO_SUFFIXES:
        return "video"
    return "unknown_file"


def _sqlite_table_columns(path: Path) -> dict[str, tuple[str, ...]]:
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            table_rows = connection.execute(
                """
                SELECT name
                FROM sqlite_schema
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name COLLATE NOCASE
                """
            ).fetchall()
            output: dict[str, tuple[str, ...]] = {}
            for row in table_rows:
                table = str(row[0])
                column_rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
                output[table] = tuple(str(column[1]) for column in column_rows if str(column[1]).strip())
    except sqlite3.Error:
        return {}
    return output


def _csv_columns(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            row = next(reader, [])
    except (OSError, csv.Error):
        return ()
    return tuple(str(cell).strip() for cell in row if str(cell).strip())


def _json_columns(path: Path) -> tuple[str, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return ()
    records: list[dict[str, Any]] = []
    if isinstance(payload, list):
        records = [item for item in payload[:50] if isinstance(item, dict)]
    elif isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                records = [item for item in value[:50] if isinstance(item, dict)]
                if records:
                    break
        if not records:
            records = [payload]
    keys: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            normalized = str(key).casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            keys.append(str(key))
    return tuple(keys)


def build_inventory(context_dir: Path) -> dict[str, SourceRef]:
    sources: dict[str, SourceRef] = {}
    seq = 0
    for path in sorted(context_dir.rglob("*")):
        if not path.is_file() or path.name.casefold() == "knowledge.md":
            continue
        seq += 1
        data_form = data_form_for_path(path)
        tables: tuple[str, ...] = ()
        columns: tuple[str, ...] = ()
        metadata: dict[str, Any] = {}
        if data_form == "sqlite_database":
            columns_by_table = _sqlite_table_columns(path)
            tables = tuple(columns_by_table)
            metadata["table_count"] = len(tables)
            metadata["columns_by_table"] = {
                table: list(columns)
                for table, columns in columns_by_table.items()
            }
        elif data_form == "csv_records":
            columns = _csv_columns(path)
            metadata["column_count"] = len(columns)
        elif data_form == "json_records":
            columns = _json_columns(path)
            metadata["observed_key_count"] = len(columns)
        elif data_form == "video":
            metadata["video_adapter"] = "unsupported_v1"
        source = SourceRef(
            id=f"src_{seq:04d}",
            path=path,
            virtual_path=virtual_path(path, context_dir),
            basename=path.name,
            stem=path.stem,
            suffix=path.suffix.casefold(),
            data_form=data_form,
            size_bytes=path.stat().st_size,
            tables=tables,
            columns=columns,
            metadata=metadata,
        )
        sources[source.id] = source
    return sources
