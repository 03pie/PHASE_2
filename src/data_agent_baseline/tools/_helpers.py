from __future__ import annotations

import csv
import fnmatch
import json
import math
import re
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import ToolMessage

DB_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".sql",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
DOC_SUFFIXES = {".log", ".md", ".pdf", ".txt"}
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
SEARCHABLE_SUFFIXES = TEXT_SUFFIXES


def tool_message(
    *,
    name: str,
    tool_call_id: str,
    payload: Mapping[str, Any] | list[Any] | str,
    max_output_bytes: int,
    status: Literal["success", "error"] = "success",
) -> ToolMessage:
    if isinstance(payload, str):
        content = payload
    else:
        content = json.dumps(jsonable(payload), ensure_ascii=False, default=str)
        limit = max(1_000, max_output_bytes)
        if len(content.encode("utf-8")) > limit:
            preview = content.encode("utf-8")[:limit].decode(
                "utf-8",
                errors="replace",
            )
            content = json.dumps(
                {
                    "truncated": True,
                    "max_output_bytes": limit,
                    "preview": preview,
                },
                ensure_ascii=False,
            )
    return ToolMessage(
        content=content,
        name=name,
        tool_call_id=tool_call_id,
        status=status,
    )


def success(
    *,
    name: str,
    tool_call_id: str,
    payload: Mapping[str, Any] | list[Any],
    max_output_bytes: int,
) -> ToolMessage:
    return tool_message(
        name=name,
        tool_call_id=tool_call_id,
        payload=payload,
        max_output_bytes=max_output_bytes,
    )


def error(
    *,
    name: str,
    tool_call_id: str,
    message: str,
    max_output_bytes: int,
) -> ToolMessage:
    return tool_message(
        name=name,
        tool_call_id=tool_call_id,
        payload=message,
        max_output_bytes=max_output_bytes,
        status="error",
    )


def resolve_context_path(
    context_root: Path,
    raw_path: str,
    *,
    allowed_suffixes: set[str] | None = None,
    allow_directory: bool = False,
    filename_fallback_dirs: Iterable[str] = ("csv", "json", "db", "doc", "video"),
) -> tuple[Path | None, str | None]:
    path_text = str(raw_path or ".").strip().replace("\\", "/")
    rel_path = virtual_context_relative_path(path_text)
    candidate = (context_root / rel_path).resolve()
    if is_inside(candidate, context_root) and candidate.exists():
        validation_error = validate_resolved_path(
            candidate,
            raw_path,
            allowed_suffixes=allowed_suffixes,
            allow_directory=allow_directory,
        )
        return (candidate, validation_error) if validation_error else (candidate, None)

    filename = pure_path_name(rel_path.as_posix())
    fallback_candidates = [
        context_root / filename,
        *[context_root / dirname / filename for dirname in filename_fallback_dirs],
    ]
    for fallback in fallback_candidates:
        resolved = fallback.resolve()
        if not is_inside(resolved, context_root) or not resolved.exists():
            continue
        validation_error = validate_resolved_path(
            resolved,
            raw_path,
            allowed_suffixes=allowed_suffixes,
            allow_directory=allow_directory,
        )
        return (resolved, validation_error) if validation_error else (resolved, None)

    suggestions = suggest_similar_paths(context_root, path_text)
    detail = f"Path not found: {raw_path!r}."
    if suggestions:
        detail += f" Did you mean: {', '.join(suggestions[:3])}?"
    return None, detail


def virtual_context_relative_path(path_text: str) -> Path:
    if path_text in {"", ".", "/context", "context"}:
        return Path()
    if path_text.startswith("/context/"):
        return Path(path_text.removeprefix("/context/"))
    if path_text.startswith("context/"):
        return Path(path_text.removeprefix("context/"))
    return Path(path_text)


def virtual_path(path: Path, context_root: Path) -> str:
    return f"/context/{path.resolve().relative_to(context_root).as_posix()}"


def pure_path_name(value: str) -> str:
    normalized = value.replace("\\", "/").rstrip("/")
    if not normalized:
        return ""
    return normalized.split("/")[-1]


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_resolved_path(
    path: Path,
    original_path: str,
    *,
    allowed_suffixes: set[str] | None,
    allow_directory: bool,
) -> str | None:
    if path.is_dir():
        if allow_directory:
            return None
        return f"Expected a file but got directory: {original_path!r}."
    if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
        return (
            f"Unsupported file extension {path.suffix!r} for {original_path!r}; "
            f"expected one of {sorted(allowed_suffixes)}."
        )
    return None


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def classify_file_role(rel_path: str, size: int) -> str:
    parts = rel_path.replace("\\", "/").split("/")
    filename = parts[-1].lower()
    parent = parts[-2].lower() if len(parts) > 1 else ""
    suffix = Path(filename).suffix.lower()
    if filename == "knowledge.md":
        return "SCHEMA_GUIDE: read first; defines terminology, metrics, and conventions"
    if suffix == ".csv":
        return "TABULAR_DATA: use read_csv to inspect; execute_python for full analysis"
    if suffix == ".json":
        return "JSON_DATA: use read_json to inspect; execute_python for full analysis"
    if suffix in DB_SUFFIXES:
        return "SQLITE_DB: use inspect_sqlite, query_schema, and execute_sql"
    if parent == "doc" or suffix in {".md", ".txt", ".pdf"}:
        hint = f"; large file {format_size(size)}, use grep_file/read_doc paging"
        return f"NARRATIVE_DATA: use read_doc or grep_file{hint if size > 10_000 else ''}"
    if suffix in VIDEO_SUFFIXES:
        return "VIDEO_DATA: video analysis is not exposed in the current tool set"
    return "DATA_FILE"


def suggest_similar_paths(context_root: Path, query: str) -> list[str]:
    from difflib import SequenceMatcher

    query_name = pure_path_name(query).lower()
    if not query_name:
        return []
    candidates: list[tuple[float, str]] = []
    try:
        files = [path for path in context_root.rglob("*") if path.is_file()]
    except OSError:
        return []
    for path in files:
        rendered = virtual_path(path, context_root)
        name = path.name.lower()
        if name == query_name:
            candidates.append((1.0, rendered))
            continue
        ratio = SequenceMatcher(None, query_name, name).ratio()
        if ratio >= 0.5:
            candidates.append((ratio, rendered))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [item[1] for item in candidates[:5]]


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        line_count = sum(
            chunk.count(b"\n") for chunk in iter(lambda: handle.read(65536), b"")
        )
    return max(0, line_count - 1)


def navigate_json_path(data: Any, path: str) -> Any:
    current = data
    for part in [item for item in re.split(r"\.|\[|\]", path) if item]:
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def extract_pdf_text(path: Path) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF text extraction requires PyMuPDF. Install the `pymupdf` dependency."
        ) from exc
    with fitz.open(path) as document:
        return "\n".join(page.get_text("text") for page in document)


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def is_readonly_sql(sql: str) -> bool:
    stripped = sql.strip()
    if not stripped:
        return False
    stripped = re.sub(r"^\s*--.*?$", "", stripped, flags=re.MULTILINE).strip()
    upper = stripped.upper()
    if not upper.startswith(("SELECT", "WITH", "PRAGMA", "EXPLAIN")):
        return False
    forbidden = re.search(
        (
            r"\b(ATTACH|CREATE|DELETE|DETACH|DROP|INSERT|REINDEX|REPLACE|"
            r"UPDATE|VACUUM|ALTER)\b"
        ),
        upper,
    )
    return forbidden is None


def empty_result_hint(sql: str) -> str | None:
    matches = re.findall(r"(\w+)\s*=\s*['\"]([^'\"]+)['\"]", sql, re.IGNORECASE)
    if not matches:
        return None
    column, value = matches[0]
    return (
        f"0 rows. Stored format for column {column!r} may differ from {value!r}; "
        f"probe with SELECT DISTINCT {column} FROM <table> LIMIT 20."
    )


def list_context_files(context_root: Path, suffixes: set[str]) -> list[str]:
    return sorted(
        virtual_path(path, context_root)
        for path in context_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def collect_search_files(
    target: Path,
    *,
    context_root: Path,
    include: str,
) -> list[tuple[Path, str]]:
    if target.is_file():
        return [(target, virtual_path(target, context_root))]
    files: list[tuple[Path, str]] = []
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SEARCHABLE_SUFFIXES:
            continue
        try:
            if path.stat().st_size > 10 * 1024 * 1024:
                continue
        except OSError:
            continue
        rel = path.relative_to(context_root).as_posix()
        if include and not fnmatch.fnmatch(rel, include):
            continue
        files.append((path, virtual_path(path, context_root)))
    return files


def regex_search(
    files: list[tuple[Path, str]],
    pattern: re.Pattern[str],
    *,
    context_lines: int,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    raw_cap = 50_000
    for path, rendered_path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        emitted: set[int] = set()
        for index, line in enumerate(lines):
            if not pattern.search(line):
                continue
            start = max(0, index - context_lines)
            end = min(len(lines), index + context_lines + 1)
            for line_index in range(start, end):
                if line_index in emitted:
                    continue
                emitted.add(line_index)
                matches.append(
                    {
                        "file": rendered_path,
                        "line_number": line_index + 1,
                        "content": lines[line_index][:300],
                        **({"is_match": True} if line_index == index else {}),
                    }
                )
                if len(matches) >= raw_cap:
                    return matches
    return matches


def pattern_to_keywords(pattern: str) -> str:
    text = re.sub(r"[()]", " ", pattern)
    text = text.replace("|", " ")
    text = re.sub(r"\\[dDsSwWbBntrAZ]", " ", text)
    text = re.sub(r"[+*?]|\{\d*,?\d*\}", " ", text)
    text = re.sub(r"[\\^$.\[\]]", " ", text)
    return " ".join(
        word
        for word in text.split()
        if len(word) > 1 and not (word.isdigit() and len(word) <= 2)
    )


def fuzzy_search(
    files: list[tuple[Path, str]],
    query: str,
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    from difflib import SequenceMatcher

    query_lower = query.lower()
    matches: list[dict[str, Any]] = []
    for path, rendered_path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for index, line in enumerate(lines):
            line_lower = line.lower()
            score = 1.0 if query_lower in line_lower else SequenceMatcher(
                None,
                query_lower,
                line_lower,
            ).ratio()
            if score >= threshold:
                matches.append(
                    {
                        "file": rendered_path,
                        "line_number": index + 1,
                        "content": line[:300],
                        "score": round(score, 3),
                    }
                )
    matches.sort(key=lambda item: float(item.get("score", 0)), reverse=True)
    return matches[:50_000]


def apply_head_limit(items: list[Any], limit: int, offset: int) -> tuple[list[Any], int | None]:
    if limit == 0:
        return items[offset:], None
    effective_limit = limit if limit > 0 else 200
    sliced = items[offset : offset + effective_limit]
    applied_limit = effective_limit if len(items) - offset > effective_limit else None
    return sliced, applied_limit


def query_context_schema(
    context_root: Path,
    field: str,
    *,
    max_matches: int,
) -> list[dict[str, Any]]:
    field_lower = field.casefold()
    matches: list[dict[str, Any]] = []
    matches.extend(_query_csv_schema(context_root, field_lower, max_matches))
    if len(matches) >= max_matches:
        return matches[:max_matches]
    matches.extend(_query_json_schema(context_root, field_lower, max_matches - len(matches)))
    if len(matches) >= max_matches:
        return matches[:max_matches]
    matches.extend(_query_sqlite_schema(context_root, field_lower, max_matches - len(matches)))
    return matches[:max_matches]


def _query_csv_schema(
    context_root: Path,
    field_lower: str,
    max_matches: int,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for path in sorted(context_root.rglob("*.csv")):
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                columns = next(csv.reader(handle), [])
        except (OSError, csv.Error):
            continue
        for column in columns:
            if field_lower in str(column).casefold():
                matches.append(
                    {
                        "source_type": "csv",
                        "path": virtual_path(path, context_root),
                        "column": column,
                    }
                )
                if len(matches) >= max_matches:
                    return matches
    return matches


def _query_json_schema(
    context_root: Path,
    field_lower: str,
    max_matches: int,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for path in sorted(context_root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for json_key in iter_json_keys(data):
            if field_lower in json_key.casefold():
                matches.append(
                    {
                        "source_type": "json",
                        "path": virtual_path(path, context_root),
                        "field": json_key,
                    }
                )
                if len(matches) >= max_matches:
                    return matches
    return matches


def _query_sqlite_schema(
    context_root: Path,
    field_lower: str,
    max_matches: int,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    databases = sorted(
        path for path in context_root.rglob("*") if path.suffix.lower() in DB_SUFFIXES
    )
    for path in databases:
        try:
            with sqlite3.connect(str(path)) as connection:
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                tables = [str(row[0]) for row in cursor.fetchall()]
                for table in tables:
                    cursor.execute(f"PRAGMA table_info({quote_identifier(table)})")
                    for row in cursor.fetchall():
                        column = str(row[1])
                        if field_lower not in column.casefold() and field_lower not in table.casefold():
                            continue
                        matches.append(
                            {
                                "source_type": "sqlite",
                                "path": virtual_path(path, context_root),
                                "table": table,
                                "column": column,
                                "type": row[2],
                            }
                        )
                        if len(matches) >= max_matches:
                            return matches
        except sqlite3.Error:
            continue
    return matches


def iter_json_keys(data: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(data, Mapping):
        for key, value in data.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            yield path
            yield from iter_json_keys(value, path)
    elif isinstance(data, list):
        for item in data[:5]:
            yield from iter_json_keys(item, prefix)
