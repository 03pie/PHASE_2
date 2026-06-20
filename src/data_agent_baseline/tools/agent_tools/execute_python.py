from __future__ import annotations

import ast
import csv
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from data_agent_baseline.agents.deep_state import (
    BenchmarkDeepAgentState,
    DeepAgentConfig,
)
from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools.answer import (
    answer_value_hash,
    normalize_answer_columns,
    validate_prepared_answer,
)
from data_agent_baseline.tools.observed_sources import merge_observed_sources

_CONTEXT_PATH_RE = re.compile(r"/context/[^\s\"'<>),;\]]+")
_ANSWER_TABLE_ASSIGNMENT_RE = re.compile(
    r"(?m)^[ \t]*(columns|rows|result|set_answer_result)\s*(?::[^=\n]+)?="
)


def _build_shell_environment() -> dict[str, str]:
    """只向隔离子进程传递运行 Python 所需的最小环境变量集合。"""

    allowed_names = {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
    environment = {
        name: value for name, value in os.environ.items() if name.upper() in allowed_names
    }
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _format_process_output(
    stdout: bytes,
    stderr: bytes,
    *,
    exit_code: int | None,
    max_output_bytes: int,
) -> str:
    sections: list[bytes] = []
    if stdout:
        sections.append(b"[stdout]\n" + stdout)
    if stderr:
        sections.append(b"[stderr]\n" + stderr)
    output = b"\n".join(sections) or b"<no output>"
    truncated = len(output) > max_output_bytes
    output = output[:max_output_bytes]
    text = output.decode("utf-8", errors="replace")
    if truncated:
        text += f"\n[output truncated at {max_output_bytes} bytes]"
    if exit_code is not None:
        text += f"\n\nExit code: {exit_code}"
    return text


def _code_declares_answer_candidate_shape(code: str) -> bool:
    if re.search(r"\bset_answer\s*\(", code):
        return True
    assigned = {
        match.group(1)
        for match in _ANSWER_TABLE_ASSIGNMENT_RE.finditer(code)
    }
    return bool({"columns", "rows"} <= assigned or assigned & {"result", "set_answer_result"})


class _VirtualPathRewriter(ast.NodeTransformer):
    """把模型代码中的虚拟路径安全映射到单次任务工作区。"""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not isinstance(node.value, str):
            return node
        mapped = self._map_path(node.value)
        if mapped == node.value:
            return node
        return ast.copy_location(ast.Constant(value=mapped), node)

    def _map_path(self, value: str) -> str:
        for virtual_root in ("context", "scratch"):
            prefix = f"/{virtual_root}"
            if value != prefix and not value.startswith(f"{prefix}/"):
                continue
            relative_path = value.removeprefix(prefix).lstrip("/")
            root = (self.workspace / virtual_root).resolve()
            mapped = (root / Path(relative_path)).resolve()
            if not mapped.is_relative_to(root):
                raise ValueError(f"Virtual path escapes /{virtual_root}: {value}")
            return str(mapped)
        return value


def _rewrite_virtual_python_paths(code: str, workspace: Path) -> str:
    tree = ast.parse(code, filename="<execute_python>", mode="exec")
    rewritten = _VirtualPathRewriter(workspace).visit(tree)
    ast.fix_missing_locations(rewritten)
    return ast.unparse(rewritten)


def _extract_context_paths(code: str) -> set[str]:
    return {
        path.replace("\\", "/").rstrip(".,")
        for path in _CONTEXT_PATH_RE.findall(code)
        if not path.lower().endswith("/knowledge.md")
    }


def _source_type_for_context_path(path: str) -> str:
    suffix = Path(path.rstrip("/").rsplit("::", 1)[0]).suffix.casefold()
    return {
        ".csv": "csv",
        ".json": "json",
        ".sqlite": "sqlite",
        ".db": "sqlite",
        ".pdf": "pdf",
        ".md": "document",
        ".markdown": "document",
        ".txt": "text",
    }.get(suffix, "directory" if path.rstrip("/").count("/") <= 2 else "file")


def _observed_sources_from_context_paths(
    code_context_paths: set[str],
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in sorted(code_context_paths):
        path = raw_path.replace("\\", "/").rstrip("/")
        if not path or path in seen:
            continue
        seen.add(path)
        sources.append(
            {
                "path": path,
                "source_type": _source_type_for_context_path(path),
                "observed_by": "execute_python",
                "observation_type": "context_path_reference",
            }
        )
    return sources


def _write_answer_candidate_artifact(
    *,
    workspace: Path,
    state: dict[str, Any],
) -> Path | None:
    candidate = state.get("answer_candidate")
    if not isinstance(candidate, dict):
        return None
    payload = {
        "columns": candidate.get("columns"),
        "rows": candidate.get("rows"),
        "audit": candidate.get("audit"),
        "column_count": candidate.get("column_count"),
        "row_count": candidate.get("row_count"),
        "code_context_paths": candidate.get("code_context_paths"),
        "validation_error": candidate.get("validation_error"),
    }
    scratch_dir = workspace / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    path = scratch_dir / "answer_candidate.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _answer_helper_source(
    result_path: Path,
    *,
    auto_capture: bool,
    candidate_path: Path | None = None,
) -> str:
    """向隔离脚本注入结果提交函数，完整数据只写入临时工作区。"""

    context_root = result_path.parent.parent / "context"
    helper_source = f"""
import hashlib as __answer_hashlib
import builtins as __answer_builtins
import io as __answer_io
import json as __answer_json
import math as __answer_math
from pathlib import Path as __AnswerPath
__ContextRoot = __AnswerPath({str(context_root)!r}).resolve()
__OriginalOpen = __answer_builtins.open
__AnswerCandidatePath = (
    __AnswerPath({str(candidate_path)!r}).resolve()
    if {candidate_path is not None!r}
    else None
)

def __normalize_answer_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if __answer_math.isfinite(value) else None
    if isinstance(value, dict):
        return {{str(key): __normalize_answer_value(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        return [__normalize_answer_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return __normalize_answer_value(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)

def __normalize_answer_column(column):
    if isinstance(column, dict):
        for key in ("name", "column", "field", "source_field"):
            value = column.get(key)
            if value is None or isinstance(value, (list, tuple, dict)):
                continue
            text = str(value).strip()
            if text:
                return text
    return str(column)

def answer_hash(columns, rows):
    payload = {{
        "columns": [__normalize_answer_column(column) for column in columns],
        "rows": __normalize_answer_value(list(rows)),
    }}
    encoded = __answer_json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return __answer_hashlib.sha256(encoded).hexdigest()

def __virtualize_audit_source(value):
    text = str(value)
    table_suffix = ""
    path_text = text
    if "::" in text:
        path_text, table_name = text.split("::", 1)
        table_suffix = "::" + table_name
    try:
        resolved = __AnswerPath(path_text).resolve()
        relative = resolved.relative_to(__ContextRoot)
    except (OSError, ValueError):
        return text.replace("\\\\", "/")
    return "/context/" + relative.as_posix() + table_suffix

def __resolve_context_path(value):
    text = str(value)
    if text == "/context" or text.startswith("/context/"):
        relative = text.removeprefix("/context").lstrip("/")
        resolved = (__ContextRoot / relative).resolve()
        resolved.relative_to(__ContextRoot)
        return resolved
    return __AnswerPath(text).resolve()

def read_pdf_text(path):
    resolved = __resolve_context_path(path)
    import fitz as __answer_fitz
    with __answer_fitz.open(str(resolved)) as document:
        return "\\n".join(page.get_text() for page in document)

def read_context_text(path, encoding="utf-8"):
    resolved = __resolve_context_path(path)
    if resolved.suffix.casefold() == ".pdf":
        return read_pdf_text(str(resolved))
    return resolved.read_text(encoding=encoding, errors="replace")

def __open_context_pdf_as_text(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    text_mode = "b" not in str(mode)
    read_only = "r" in str(mode) and not any(flag in str(mode) for flag in "wax+")
    if text_mode and read_only:
        try:
            resolved = __AnswerPath(file).resolve()
            resolved.relative_to(__ContextRoot)
        except (TypeError, OSError, ValueError):
            resolved = None
        if resolved is not None and resolved.suffix.casefold() == ".pdf":
            return __answer_io.StringIO(read_pdf_text(str(resolved)))
    return __OriginalOpen(file, mode, buffering, encoding, errors, newline, closefd, opener)

__answer_builtins.open = __open_context_pdf_as_text

def read_answer_candidate():
    if __AnswerCandidatePath is None or not __AnswerCandidatePath.exists():
        return {{"columns": [], "rows": [], "audit": None}}
    return __answer_json.loads(__AnswerCandidatePath.read_text(encoding="utf-8"))

answer_candidate = read_answer_candidate()

def __normalize_answer_audit(audit):
    audit = __normalize_answer_value(audit)
    if not isinstance(audit, dict):
        return audit
    normalized = dict(audit)
    for key in ("source_paths", "sources"):
        values = normalized.get(key)
        if isinstance(values, list):
            normalized[key] = [__virtualize_audit_source(item) for item in values]
    return normalized

def set_answer(columns, rows, audit=None):
    normalized_columns = [__normalize_answer_column(column) for column in columns]
    normalized_rows = __normalize_answer_value(list(rows))
    normalized_audit = __normalize_answer_audit(audit)
    if isinstance(normalized_audit, dict):
        normalized_audit["output_row_count"] = len(normalized_rows)
        normalized_audit["output_hash"] = answer_hash(normalized_columns, normalized_rows)
    payload = {{
        "columns": normalized_columns,
        "rows": normalized_rows,
        "audit": normalized_audit,
    }}
    result_path = __AnswerPath({str(result_path)!r})
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        __answer_json.dumps(payload, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

"""
    if not auto_capture:
        return helper_source
    return helper_source + f"""
def __auto_capture_answer():
    result_file = __AnswerPath({str(result_path)!r})
    if result_file.exists():
        return
    namespace = globals()
    for candidate_name in ("result", "set_answer_result"):
        candidate = namespace.get(candidate_name)
        if isinstance(candidate, dict) and "columns" in candidate and "rows" in candidate:
            try:
                set_answer(
                    candidate["columns"],
                    candidate["rows"],
                    audit=candidate.get("audit"),
                )
            except Exception:
                pass
            return
    if "columns" in namespace and "rows" in namespace:
        try:
            set_answer(namespace["columns"], namespace["rows"], audit=namespace.get("audit"))
        except Exception:
            pass

import atexit as __answer_atexit
__answer_atexit.register(__auto_capture_answer)
"""


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


def _source_aliases(path: str) -> set[str]:
    normalized = path.replace("\\", "/")
    aliases = {normalized}
    if "::" in normalized:
        aliases.add(normalized.split("::", 1)[0])
    return aliases


def _declared_plan_sources(analysis_plan: dict[str, Any]) -> set[str]:
    declared: set[str] = set()
    for section_name in ("evidence", "execution_spec"):
        section = analysis_plan.get(section_name) or {}
        if not isinstance(section, dict):
            continue
        key = "context_sources" if section_name == "evidence" else "sources"
        for source in section.get(key) or []:
            if not isinstance(source, dict):
                continue
            path = str(source.get("path") or "").strip()
            if path:
                declared.update(_source_aliases(path))
    return declared


def _synthesize_execution_audit(
    *,
    analysis_plan: dict[str, Any],
    code_context_paths: set[str],
    columns: list[str],
    rows: list[list[Any]],
) -> dict[str, Any] | None:
    if not (_plan_is_transform(analysis_plan) or _plan_has_source_bindings(analysis_plan)):
        return None
    operations = _plan_operation_labels(analysis_plan)
    if not operations and not _plan_has_source_bindings(analysis_plan):
        return None
    declared_sources = _declared_plan_sources(analysis_plan)
    source_paths = sorted(
        path
        for path in code_context_paths
        if not declared_sources or _source_aliases(path) & declared_sources
    )
    if not source_paths:
        source_paths = _plan_source_paths(analysis_plan)
    if not source_paths:
        return None
    return {
        "source_paths": source_paths,
        "operations": operations or ["source_bound_projection"],
        "output_row_count": len(rows),
        "output_hash": answer_value_hash(columns, rows),
        "audit_origin": "execute_python_static_context_paths",
    }


def _iter_json_field_names(data: Any, *, depth: int = 0) -> list[str]:
    if depth > 8:
        return []
    names: list[str] = []
    if isinstance(data, Mapping):
        for key, value in data.items():
            text = str(key)
            if text:
                names.append(text)
            names.extend(_iter_json_field_names(value, depth=depth + 1))
    elif isinstance(data, list):
        for item in data[:20]:
            names.extend(_iter_json_field_names(item, depth=depth + 1))
    return names


@lru_cache(maxsize=64)
def _context_field_case_map(workspace_path: str) -> dict[str, str]:
    context_dir = Path(workspace_path) / "context"
    variants: dict[str, set[str]] = {}

    def add_field(name: Any) -> None:
        text = str(name or "").strip()
        alias = _normalized_recovery_alias(text)
        if text and alias:
            variants.setdefault(alias, set()).add(text)

    if not context_dir.exists():
        return {}
    for path in context_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.casefold()
        if suffix == ".csv":
            try:
                with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                    for field in next(csv.reader(handle), []):
                        add_field(field)
            except (OSError, csv.Error):
                continue
        elif suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                continue
            for field in _iter_json_field_names(data):
                add_field(field)
        elif suffix in {".sqlite", ".sqlite3", ".db"}:
            try:
                with sqlite3.connect(str(path)) as connection:
                    cursor = connection.cursor()
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    )
                    for (table_name,) in cursor.fetchall():
                        table = str(table_name)
                        cursor.execute(
                            'PRAGMA table_info("' + table.replace('"', '""') + '")'
                        )
                        for row in cursor.fetchall():
                            add_field(row[1])
            except sqlite3.Error:
                continue
    return {
        alias: next(iter(values))
        for alias, values in variants.items()
        if len(values) == 1
    }


def _canonicalize_prepared_answer_field_casing(
    prepared_answer: Any,
    *,
    workspace: Path,
    analysis_plan: dict[str, Any] | None,
) -> Any:
    if not isinstance(prepared_answer, AnswerTable):
        return prepared_answer
    if not isinstance(analysis_plan, dict):
        return prepared_answer
    output_spec = analysis_plan.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return prepared_answer
    plan_columns = [
        column
        for column in output_spec.get("columns") or []
        if isinstance(column, Mapping)
    ]
    case_map = _context_field_case_map(str(workspace.resolve()))
    if not case_map:
        return prepared_answer

    updated_columns = list(prepared_answer.columns)
    changed = False
    for index, column in enumerate(updated_columns):
        if index >= len(plan_columns):
            continue
        source_fields = [
            str(field or "").strip()
            for field in plan_columns[index].get("source_fields") or []
            if str(field or "").strip()
        ]
        if not source_fields:
            continue
        aliases = [column, *source_fields]
        replacement = next(
            (
                case_map[_normalized_recovery_alias(alias)]
                for alias in aliases
                if _normalized_recovery_alias(alias) in case_map
            ),
            None,
        )
        if replacement and replacement != column:
            updated_columns[index] = replacement
            changed = True
    if not changed:
        return prepared_answer
    return AnswerTable(columns=updated_columns, rows=prepared_answer.rows)


def _load_prepared_answer(
    *,
    workspace: Path,
    result_path: Path,
    analysis_plan: dict[str, Any] | None,
    code_context_paths: set[str],
) -> tuple[Any | None, str | None, dict[str, Any] | None]:
    if not result_path.exists():
        return None, None, None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"set_answer produced an unreadable result: {exc}", None
    columns = payload.get("columns")
    rows = payload.get("rows")
    audit = payload.get("audit")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return None, "set_answer requires list columns and list rows.", None
    normalized_columns = normalize_answer_columns(columns)
    if all(isinstance(row, dict) for row in rows):
        normalized_rows = [
            [row.get(column) for column in normalized_columns]
            for row in rows
        ]
    elif all(isinstance(row, list) for row in rows):
        normalized_rows = [list(row) for row in rows]
    else:
        return None, "set_answer rows must contain only row lists or row objects.", None
    candidate = {
        "columns": normalized_columns,
        "rows": normalized_rows,
        "audit": audit,
        "column_count": len(normalized_columns),
        "row_count": len(normalized_rows),
        "code_context_paths": sorted(code_context_paths),
    }
    if analysis_plan is None:
        answer_error = "set_answer requires a successful analysis_plan first."
        candidate["validation_error"] = answer_error
        return None, answer_error, candidate
    if not isinstance(audit, dict):
        audit = _synthesize_execution_audit(
            analysis_plan=analysis_plan,
            code_context_paths=code_context_paths,
            columns=normalized_columns,
            rows=normalized_rows,
        )
        candidate["audit"] = audit
    candidate_plan_answer = _try_candidate_plan_execution(
        workspace=workspace,
        analysis_plan=analysis_plan,
        candidate=candidate,
    )
    if candidate_plan_answer is not None:
        return (
            _canonicalize_prepared_answer_field_casing(
                candidate_plan_answer,
                workspace=workspace,
                analysis_plan=analysis_plan,
            ),
            None,
            candidate,
        )
    prepared_answer, answer_error = validate_prepared_answer(
        normalized_columns,
        normalized_rows,
        analysis_plan,
        audit if isinstance(audit, dict) else None,
    )
    if answer_error is not None:
        candidate["validation_error"] = answer_error
    if prepared_answer is not None:
        prepared_answer = _canonicalize_prepared_answer_field_casing(
            prepared_answer,
            workspace=workspace,
            analysis_plan=analysis_plan,
        )
    return prepared_answer, answer_error, candidate


def _normalized_recovery_alias(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value or "").casefold())


def _stdout_literal_candidates(stdout: bytes) -> list[Any]:
    text = stdout.decode("utf-8", errors="replace")
    candidates: list[Any] = []
    snippets: list[str] = []
    marker_pattern = re.compile(
        r"(?:final result|result|answer)\s*:\s*(\{[^\n]+\}|\[[^\n]+\])",
        flags=re.IGNORECASE,
    )
    snippets.extend(match.group(1).strip() for match in marker_pattern.finditer(text))
    for line in text.splitlines():
        stripped = line.strip()
        if (
            (stripped.startswith("{") and stripped.endswith("}"))
            or (stripped.startswith("[") and stripped.endswith("]"))
        ):
            snippets.append(stripped)
    for snippet in reversed(snippets):
        try:
            value = ast.literal_eval(snippet)
        except (SyntaxError, ValueError):
            try:
                value = json.loads(snippet)
            except json.JSONDecodeError:
                continue
        candidates.append(value)
    return candidates


def _candidate_from_stdout_literal(
    literal: Any,
    analysis_plan: dict[str, Any] | None,
) -> tuple[list[str], list[list[Any]]] | None:
    if not isinstance(analysis_plan, dict):
        if isinstance(literal, Mapping):
            if "columns" in literal and "rows" in literal:
                columns = literal.get("columns")
                rows = literal.get("rows")
                if isinstance(columns, list) and isinstance(rows, list):
                    return normalize_answer_columns(columns), [
                        list(row.values()) if isinstance(row, Mapping) else list(row)
                        for row in rows
                        if isinstance(row, (list, Mapping))
                    ]
            columns = normalize_answer_columns([str(key) for key in literal.keys()])
            return columns, [[literal.get(key) for key in literal.keys()]]
        if isinstance(literal, list) and all(
            isinstance(row, Mapping) for row in literal
        ):
            columns = normalize_answer_columns(
                list(
                    dict.fromkeys(
                        str(key)
                        for row in literal
                        for key in row.keys()
                    )
                )
            )
            if not columns:
                return None
            return columns, [
                [row.get(column) for column in columns]
                for row in literal
                if isinstance(row, Mapping)
            ]
        return None
    if not isinstance(analysis_plan, dict):
        return None
    output_spec = analysis_plan.get("output_spec") or {}
    if not isinstance(output_spec, dict):
        return None
    plan_columns = [
        column
        for column in output_spec.get("columns") or []
        if isinstance(column, dict)
    ]
    if not plan_columns:
        return None

    if isinstance(literal, Mapping):
        if "columns" in literal and "rows" in literal:
            columns = literal.get("columns")
            rows = literal.get("rows")
            if isinstance(columns, list) and isinstance(rows, list):
                return normalize_answer_columns(columns), [
                    list(row.values()) if isinstance(row, Mapping) else list(row)
                    for row in rows
                    if isinstance(row, (list, Mapping))
                ]
        selected_columns: list[str] = []
        selected_values: list[Any] = []
        literal_by_alias = {
            _normalized_recovery_alias(key): (str(key), value)
            for key, value in literal.items()
        }
        for plan_column in plan_columns:
            aliases = [
                str(plan_column.get("name") or ""),
                *[
                    str(field or "")
                    for field in plan_column.get("source_fields") or []
                    if str(field or "").strip()
                ],
            ]
            selected: tuple[str, Any] | None = None
            for alias in aliases:
                selected = literal_by_alias.get(_normalized_recovery_alias(alias))
                if selected is not None:
                    break
            if selected is None and len(plan_columns) == 1 and len(literal) == 1:
                key, value = next(iter(literal.items()))
                selected = (str(key), value)
            if selected is None:
                return None
            selected_columns.append(selected[0])
            selected_values.append(selected[1])
        return selected_columns, [selected_values]

    if isinstance(literal, list) and all(isinstance(row, Mapping) for row in literal):
        rows: list[list[Any]] = []
        selected_columns: list[str] = []
        for column_index, plan_column in enumerate(plan_columns):
            aliases = [
                str(plan_column.get("name") or ""),
                *[
                    str(field or "")
                    for field in plan_column.get("source_fields") or []
                    if str(field or "").strip()
                ],
            ]
            selected_key: str | None = None
            for row in literal:
                row_by_alias = {
                    _normalized_recovery_alias(key): str(key)
                    for key in row.keys()
                }
                for alias in aliases:
                    selected_key = row_by_alias.get(_normalized_recovery_alias(alias))
                    if selected_key is not None:
                        break
                if selected_key is not None:
                    break
            if selected_key is None:
                return None
            selected_columns.append(selected_key)
            for row_index, row in enumerate(literal):
                if column_index == 0:
                    rows.append([])
                rows[row_index].append(row.get(selected_key))
        return selected_columns, rows
    return None


def _recover_stdout_answer(
    *,
    workspace: Path,
    stdout: bytes,
    analysis_plan: dict[str, Any] | None,
    code_context_paths: set[str],
) -> tuple[Any | None, str | None, dict[str, Any] | None]:
    if not stdout:
        return None, None, None
    for literal in _stdout_literal_candidates(stdout):
        candidate_table = _candidate_from_stdout_literal(literal, analysis_plan)
        if candidate_table is None:
            continue
        columns, rows = candidate_table
        if not columns or not rows:
            continue
        audit = (
            _synthesize_execution_audit(
                analysis_plan=analysis_plan,
                code_context_paths=code_context_paths,
                columns=columns,
                rows=rows,
            )
            if isinstance(analysis_plan, dict)
            else None
        )
        candidate = {
            "columns": columns,
            "rows": rows,
            "audit": audit,
            "column_count": len(columns),
            "row_count": len(rows),
            "code_context_paths": sorted(code_context_paths),
            "recovered_from": "stdout_literal",
        }
        if not isinstance(analysis_plan, dict):
            answer_error = "stdout answer recovery requires a successful analysis_plan first."
            candidate["validation_error"] = answer_error
            return None, answer_error, candidate
        prepared_answer, answer_error = validate_prepared_answer(
            columns,
            rows,
            analysis_plan,
            audit if isinstance(audit, dict) else None,
        )
        if answer_error is not None:
            candidate["validation_error"] = answer_error
            return None, answer_error, candidate
        return (
            _canonicalize_prepared_answer_field_casing(
                prepared_answer,
                workspace=workspace,
                analysis_plan=analysis_plan,
            ),
            None,
            candidate,
        )
    return None, None, None


def _plan_preserves_source_rows(analysis_plan: dict[str, Any] | None) -> bool:
    if not isinstance(analysis_plan, dict):
        return False
    output_spec = analysis_plan.get("output_spec") or {}
    return (
        isinstance(output_spec, dict)
        and output_spec.get("row_policy") == "preserve"
        and not output_spec.get("transformations")
    )


def _plan_has_constraining_requirements(analysis_plan: dict[str, Any]) -> bool:
    intent = analysis_plan.get("intent")
    if not isinstance(intent, Mapping):
        return False
    constraining_types = {
        "calculation",
        "deduplication",
        "entity",
        "filter",
        "grouping",
        "limit",
        "ordering",
        "scope",
        "selector",
        "time_range",
        "value",
    }
    for requirement in intent.get("requirements") or []:
        if not isinstance(requirement, Mapping):
            continue
        requirement_type = str(requirement.get("requirement_type") or "").casefold()
        if requirement_type in constraining_types:
            return True
    return False


def _plan_output_projection(analysis_plan: dict[str, Any]) -> tuple[list[str], list[str]]:
    output_spec = analysis_plan.get("output_spec") or {}
    if not isinstance(output_spec, dict):
        return [], []
    output_columns = [
        column
        for column in output_spec.get("columns") or []
        if isinstance(column, dict)
    ]
    columns: list[str] = []
    source_fields: list[str] = []
    for column in output_columns:
        name = str(column.get("name") or "").strip()
        fields = [
            str(field or "").strip()
            for field in column.get("source_fields") or []
            if str(field or "").strip()
        ]
        if not name or not fields:
            return [], []
        columns.append(name)
        source_fields.append(fields[0])
    return columns, source_fields


def _plan_source_paths(analysis_plan: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for section_name, key in (("execution_spec", "sources"), ("evidence", "context_sources")):
        section = analysis_plan.get(section_name) or {}
        if not isinstance(section, dict):
            continue
        for source in section.get(key) or []:
            if not isinstance(source, dict):
                continue
            path = str(source.get("path") or "").replace("\\", "/").strip()
            if path and not path.lower().endswith("/knowledge.md") and path not in paths:
                paths.append(path)
    return paths


def _resolve_virtual_context_path(
    workspace: Path,
    virtual_path: str,
) -> tuple[Path | None, str | None]:
    path_text = virtual_path.replace("\\", "/")
    table_name = None
    if "::" in path_text:
        path_text, table_name = path_text.split("::", 1)
    if not path_text.startswith("/context/"):
        return None, table_name
    relative = path_text.removeprefix("/context/").lstrip("/")
    root = (workspace / "context").resolve()
    resolved = (root / Path(relative)).resolve()
    if not resolved.is_relative_to(root):
        return None, table_name
    return resolved, table_name


def _case_get(record: dict[str, Any], field: str) -> Any:
    if field in record:
        return record[field]
    folded = field.casefold()
    for key, value in record.items():
        if str(key).casefold() == folded:
            return value
    return None


def _case_key(record: Mapping[str, Any], field: str) -> str | None:
    if field in record:
        return field
    folded = field.casefold()
    for key in record.keys():
        if str(key).casefold() == folded:
            return str(key)
    return None


def _read_projection_from_csv(
    path: Path,
    source_fields: list[str],
) -> list[list[Any]] | None:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                return [
                    [_case_get(dict(row), field) for field in source_fields]
                    for row in csv.DictReader(handle)
                ]
        except UnicodeDecodeError:
            continue
    return None


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


def _read_projection_from_json(
    path: Path,
    source_fields: list[str],
) -> list[list[Any]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    records = _json_records(payload)
    if not records:
        return None
    return [[_case_get(record, field) for field in source_fields] for record in records]


def _quote_sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _read_projection_from_sqlite(
    path: Path,
    table_name: str,
    source_fields: list[str],
) -> list[list[Any]] | None:
    try:
        with sqlite3.connect(str(path)) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            cursor.execute(f"PRAGMA table_info({_quote_sql_identifier(table_name)})")
            actual_by_folded = {
                str(row[1]).casefold(): str(row[1])
                for row in cursor.fetchall()
            }
            actual_fields = [
                actual_by_folded.get(field.casefold(), field)
                for field in source_fields
            ]
            select_list = ", ".join(_quote_sql_identifier(field) for field in actual_fields)
            cursor.execute(
                f"SELECT {select_list} FROM {_quote_sql_identifier(table_name)}"
            )
            return [list(row) for row in cursor.fetchall()]
    except sqlite3.Error:
        return None


def _records_from_candidate(candidate: Any) -> list[dict[str, Any]]:
    if not isinstance(candidate, Mapping):
        return []
    columns = candidate.get("columns")
    rows = candidate.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return []
    normalized_columns = normalize_answer_columns(columns)
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(normalized_columns):
            continue
        records.append(dict(zip(normalized_columns, row, strict=True)))
    return records


def _candidate_source_aliases(candidate: Any) -> set[str]:
    if not isinstance(candidate, Mapping):
        return set()
    audit = candidate.get("audit")
    paths: list[Any] = []
    if isinstance(audit, Mapping):
        raw_paths = audit.get("source_paths") or audit.get("sources")
        if isinstance(raw_paths, list):
            paths.extend(raw_paths)
    raw_context_paths = candidate.get("code_context_paths")
    if isinstance(raw_context_paths, list):
        paths.extend(raw_context_paths)
    return {
        alias
        for path in paths
        for alias in _source_aliases(str(path or ""))
        if alias.strip()
    }


def _join_key(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value or "").strip()
    try:
        number = float(text)
    except ValueError:
        return text.casefold()
    if number.is_integer():
        return str(int(number))
    return text.casefold()


def _field_alias(value: Any) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


def _read_sqlite_records_for_source(
    *,
    workspace: Path,
    source: str,
) -> list[dict[str, Any]]:
    path, table_name = _resolve_virtual_context_path(workspace, source)
    if path is None or table_name is None or path.suffix.casefold() not in {".db", ".sqlite"}:
        return []
    try:
        with sqlite3.connect(str(path)) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.execute(
                f"SELECT * FROM {_quote_sql_identifier(table_name)}"
            )
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error:
        return []


def _read_records_for_source(
    *,
    workspace: Path,
    source: str,
) -> list[dict[str, Any]]:
    path, table_name = _resolve_virtual_context_path(workspace, source)
    if path is None or not path.exists():
        return []
    suffix = path.suffix.casefold()
    if table_name and suffix in {".db", ".sqlite", ".sqlite3"}:
        return _read_sqlite_records_for_source(workspace=workspace, source=source)
    if suffix == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "gbk"):
            try:
                with path.open(newline="", encoding=encoding) as handle:
                    return [dict(row) for row in csv.DictReader(handle)]
            except UnicodeDecodeError:
                continue
            except (OSError, csv.Error):
                return []
        return []
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return []
        return _json_records(payload)
    return []


def _record_field_by_alias(record: Mapping[str, Any]) -> dict[str, str]:
    return {
        _field_alias(key): str(key)
        for key in record.keys()
        if _field_alias(key)
    }


def _apply_candidate_join_operations(
    *,
    workspace: Path,
    records: list[dict[str, Any]],
    candidate_aliases: set[str],
    operations: list[Any],
) -> list[dict[str, Any]]:
    joined = records
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        if str(operation.get("operation") or "").casefold() != "join":
            continue
        left_source = str(operation.get("left_source") or "").replace("\\", "/")
        right_source = str(operation.get("right_source") or "").replace("\\", "/")
        left_key = str(operation.get("left_key") or "").strip()
        right_key = str(operation.get("right_key") or "").strip()
        if not (left_source and right_source and left_key and right_key):
            continue
        if candidate_aliases & _source_aliases(left_source):
            candidate_key, other_source, other_key = left_key, right_source, right_key
        elif candidate_aliases & _source_aliases(right_source):
            candidate_key, other_source, other_key = right_key, left_source, left_key
        else:
            continue
        other_records = _read_sqlite_records_for_source(
            workspace=workspace,
            source=other_source,
        )
        if not other_records:
            continue
        other_by_key = {
            _join_key(_case_get(record, other_key)): record
            for record in other_records
            if _case_get(record, other_key) is not None
        }
        merged_records: list[dict[str, Any]] = []
        for record in joined:
            match = other_by_key.get(_join_key(_case_get(record, candidate_key)))
            if match is None:
                continue
            merged = dict(record)
            for key, value in match.items():
                merged.setdefault(str(key), value)
            merged_records.append(merged)
        joined = merged_records
    return joined


def _apply_candidate_source_completion_joins(
    *,
    workspace: Path,
    records: list[dict[str, Any]],
    analysis_plan: dict[str, Any],
    candidate_aliases: set[str],
) -> list[dict[str, Any]]:
    if not records:
        return records
    needed_group_fields = [
        source_field
        for _, source_field in _plan_group_columns(analysis_plan)
        if not any(_case_get(record, source_field) is not None for record in records)
    ]
    if not needed_group_fields:
        return records
    candidate_field_aliases = _record_field_by_alias(records[0])
    for source_path in _plan_source_paths(analysis_plan):
        other_records = _read_records_for_source(
            workspace=workspace,
            source=source_path,
        )
        if not other_records:
            continue
        other_field_aliases = _record_field_by_alias(other_records[0])
        needed_aliases = {_field_alias(field) for field in needed_group_fields}
        if not (needed_aliases & set(other_field_aliases)):
            continue
        common_aliases = set(candidate_field_aliases) & set(other_field_aliases)
        if not common_aliases:
            continue
        join_aliases = sorted(
            common_aliases,
            key=lambda item: (
                0 if ("id" in item or "code" in item or "key" in item) else 1,
                item,
            ),
        )
        candidate_keys = [candidate_field_aliases[alias] for alias in join_aliases]
        other_keys = [other_field_aliases[alias] for alias in join_aliases]
        other_by_key: dict[tuple[str, ...], dict[str, Any] | None] = {}
        for record in other_records:
            key = tuple(_join_key(_case_get(record, field)) for field in other_keys)
            if any(not item for item in key):
                continue
            if key in other_by_key:
                other_by_key[key] = None
            else:
                other_by_key[key] = record
        merged_records: list[dict[str, Any]] = []
        for record in records:
            key = tuple(_join_key(_case_get(record, field)) for field in candidate_keys)
            if any(not item for item in key):
                continue
            match = other_by_key.get(key)
            if match is None:
                continue
            merged = dict(record)
            for key, value in match.items():
                merged.setdefault(str(key), value)
            merged_records.append(merged)
        if merged_records and all(
            any(_case_get(record, field) is not None for record in merged_records)
            for field in needed_group_fields
        ):
            return merged_records
    return records


_FILTER_COMPARISON_RE = re.compile(
    r"\b(?P<field>[A-Za-z_][A-Za-z0-9_]*)\b\s*"
    r"(?P<op>>=|<=|>|<|==|=)\s*"
    r"(?P<value>[-+]?\d+(?:\.\d+)?)"
)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def _explicit_limit_count(operation: Mapping[str, Any]) -> int | None:
    for key in ("limit", "count", "n", "value"):
        count = _positive_int(operation.get(key))
        if count is not None:
            return count
    return None


def _apply_comparison(value: float, op: str, threshold: float) -> bool:
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    return value == threshold


def _apply_candidate_filter_operations(
    records: list[dict[str, Any]],
    operations: list[Any],
) -> list[dict[str, Any]]:
    filtered = records
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        if str(operation.get("operation") or "").casefold() != "filter":
            continue
        authorization = operation.get("authorization")
        quote = (
            str(authorization.get("quote") or "")
            if isinstance(authorization, Mapping)
            else ""
        )
        text = " ".join(
            str(item or "")
            for item in (operation.get("description"), quote)
        )
        match = _FILTER_COMPARISON_RE.search(text)
        if match is None:
            continue
        field = match.group("field")
        op = match.group("op")
        threshold = float(match.group("value"))
        if not any(_case_get(record, field) is not None for record in filtered):
            continue
        filtered = [
            record
            for record in filtered
            if (
                (value := _coerce_float(_case_get(record, field))) is not None
                and _apply_comparison(value, op, threshold)
            )
        ]
    return filtered


def _plan_count_column_name(analysis_plan: dict[str, Any]) -> str:
    output_spec = analysis_plan.get("output_spec") or {}
    if not isinstance(output_spec, Mapping):
        return "count"
    for column in output_spec.get("columns") or []:
        if not isinstance(column, Mapping):
            continue
        aliases = {
            str(column.get("name") or "").casefold(),
            *(
                str(item or "").casefold()
                for item in column.get("source_fields") or []
                if str(item or "").strip()
            ),
        }
        normalized_aliases = {_field_alias(alias) for alias in aliases}
        if normalized_aliases & {"count", "frequency", "freq", "recordcount", "rowcount"}:
            return str(column.get("name") or "count")
    return "count"


def _plan_group_columns(analysis_plan: dict[str, Any]) -> list[tuple[str, str]]:
    output_spec = analysis_plan.get("output_spec") or {}
    if not isinstance(output_spec, Mapping):
        return []
    columns: list[tuple[str, str]] = []
    for column in output_spec.get("columns") or []:
        if not isinstance(column, Mapping):
            continue
        aliases = {
            str(column.get("name") or "").casefold(),
            *(
                str(item or "").casefold()
                for item in column.get("source_fields") or []
                if str(item or "").strip()
            ),
        }
        normalized_aliases = {_field_alias(alias) for alias in aliases}
        if normalized_aliases & {"count", "frequency", "freq", "recordcount", "rowcount"}:
            continue
        name = str(column.get("name") or "").strip()
        source_fields = [
            str(item or "").strip()
            for item in column.get("source_fields") or []
            if str(item or "").strip()
        ]
        source_field = source_fields[0] if source_fields else name
        if name and source_field:
            columns.append((name, source_field))
    return columns


def _aggregate_candidate_records(
    *,
    records: list[dict[str, Any]],
    analysis_plan: dict[str, Any],
) -> tuple[list[str], list[list[Any]]] | None:
    operations = _plan_operation_labels(analysis_plan)
    if not any(
        isinstance(operation, Mapping)
        and str(operation.get("operation") or "").casefold() == "aggregate"
        for operation in operations
    ):
        return None
    group_columns = _plan_group_columns(analysis_plan)
    if not group_columns:
        return None
    count_column = _plan_count_column_name(analysis_plan)
    counts: dict[tuple[Any, ...], int] = {}
    for record in records:
        key = tuple(_case_get(record, source_field) for _, source_field in group_columns)
        if any(value is None for value in key):
            continue
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    columns = [name for name, _ in group_columns] + [count_column]
    rows = [list(key) + [count] for key, count in counts.items()]
    rows.sort(key=lambda row: tuple(str(item) for item in row[:-1]))
    return columns, rows


def _apply_candidate_sort_limit_operations(
    records: list[dict[str, Any]],
    analysis_plan: dict[str, Any],
    operations: list[Any],
) -> list[dict[str, Any]]:
    output_spec = analysis_plan.get("output_spec") or {}
    sorted_records = records
    sort_keys: list[Mapping[str, Any]] = []
    if isinstance(output_spec, Mapping):
        sort_keys = [
            item
            for item in output_spec.get("sort_keys") or []
            if isinstance(item, Mapping) and str(item.get("field") or "").strip()
        ]
        for sort_key in reversed(sort_keys):
            field = str(sort_key.get("field") or "").strip()
            descending = (
                str(sort_key.get("direction") or "").strip().casefold()
                in {"desc", "descending"}
            )
            non_null_records = [
                record
                for record in sorted_records
                if _case_get(record, field) is not None
            ]
            null_records = [
                record
                for record in sorted_records
                if _case_get(record, field) is None
            ]
            sorted_records = [
                *sorted(
                    non_null_records,
                    key=lambda record: _case_get(record, field),
                    reverse=descending,
                ),
                *null_records,
            ]
    limit_operations = [
        operation
        for operation in operations
        if isinstance(operation, Mapping)
        and str(operation.get("operation") or "").strip().casefold() == "limit"
    ]
    limit_counts = [
        count
        for count in (_explicit_limit_count(operation) for operation in limit_operations)
        if count is not None
    ]
    if limit_counts:
        return sorted_records[: min(limit_counts)]
    if limit_operations and sort_keys and sorted_records:
        primary_field = str(sort_keys[0].get("field") or "").strip()
        top_value = _case_get(sorted_records[0], primary_field)
        return [
            record
            for record in sorted_records
            if _case_get(record, primary_field) == top_value
        ]
    return sorted_records


def _project_candidate_records_to_plan(
    records: list[dict[str, Any]],
    analysis_plan: dict[str, Any],
) -> tuple[list[str], list[list[Any]]] | None:
    output_spec = analysis_plan.get("output_spec") or {}
    if not isinstance(output_spec, Mapping):
        return None
    plan_columns = [
        column
        for column in output_spec.get("columns") or []
        if isinstance(column, Mapping)
    ]
    if not plan_columns:
        return None
    projected_columns: list[str] = []
    source_fields: list[str] = []
    for column in plan_columns:
        fields = [
            str(item or "").strip()
            for item in column.get("source_fields") or []
            if str(item or "").strip()
        ]
        field = fields[0] if fields else str(column.get("name") or "").strip()
        if not field:
            return None
        actual_key = next(
            (
                key
                for record in records
                if (key := _case_key(record, field)) is not None
            ),
            None,
        )
        if actual_key is None:
            return None
        source_fields.append(field)
        projected_columns.append(actual_key)
    projected_rows = [
        [_case_get(record, field) for field in source_fields]
        for record in records
        if all(_case_get(record, field) is not None for field in source_fields)
    ]
    if not projected_rows:
        return None
    return projected_columns, projected_rows


def _try_candidate_plan_execution(
    *,
    workspace: Path,
    analysis_plan: Any,
    candidate: Any,
) -> Any | None:
    if not isinstance(analysis_plan, dict):
        return None
    records = _records_from_candidate(candidate)
    if not records:
        return None
    operations = _plan_operation_labels(analysis_plan)
    candidate_aliases = _candidate_source_aliases(candidate)
    records = _apply_candidate_join_operations(
        workspace=workspace,
        records=records,
        candidate_aliases=candidate_aliases,
        operations=operations,
    )
    records = _apply_candidate_source_completion_joins(
        workspace=workspace,
        records=records,
        analysis_plan=analysis_plan,
        candidate_aliases=candidate_aliases,
    )
    records = _apply_candidate_filter_operations(records, operations)
    aggregated = _aggregate_candidate_records(
        records=records,
        analysis_plan=analysis_plan,
    )
    if aggregated is None:
        projected = _project_candidate_records_to_plan(
            _apply_candidate_sort_limit_operations(
                records,
                analysis_plan,
                operations,
            ),
            analysis_plan,
        )
        if projected is None:
            return None
        columns, rows = projected
    else:
        columns, rows = aggregated
    source_paths = _plan_source_paths(analysis_plan)
    audit = {
        "source_paths": source_paths,
        "operations": [
            operation
            for operation in operations
            if isinstance(operation, Mapping)
            and str(operation.get("operation") or "").strip()
        ],
        "output_row_count": len(rows),
        "output_hash": answer_value_hash(columns, rows),
        "audit_origin": "execute_python_candidate_plan_execution",
    }
    prepared_answer, answer_error = validate_prepared_answer(
        columns,
        rows,
        analysis_plan,
        audit,
    )
    if answer_error is not None:
        return None
    return _canonicalize_prepared_answer_field_casing(
        prepared_answer,
        workspace=workspace,
        analysis_plan=analysis_plan,
    )


def _candidate_for_update(
    *,
    existing: Any,
    new: Any,
    analysis_plan: Any,
) -> Any:
    if not isinstance(existing, Mapping) or not isinstance(new, Mapping):
        return new
    existing_rows = existing.get("row_count")
    new_rows = new.get("row_count")
    if not isinstance(existing_rows, int):
        existing_rows = len(existing.get("rows") or [])
    if not isinstance(new_rows, int):
        new_rows = len(new.get("rows") or [])
    if not isinstance(analysis_plan, dict) and existing_rows >= new_rows:
        return existing
    if isinstance(analysis_plan, dict) and new_rows == 0 and existing_rows > 0:
        return existing
    return new


def _try_preserve_source_projection(
    *,
    workspace: Path,
    analysis_plan: dict[str, Any] | None,
    allow_bound_scope_requirements: bool = False,
) -> Any | None:
    if not isinstance(analysis_plan, dict):
        return None
    columns, source_fields = _plan_output_projection(analysis_plan)
    if not columns or not source_fields:
        return None
    for source_path in _plan_source_paths(analysis_plan):
        resolved, table_name = _resolve_virtual_context_path(workspace, source_path)
        if resolved is None or not resolved.exists():
            continue
        rows: list[list[Any]] | None = None
        suffix = resolved.suffix.casefold()
        if table_name and suffix in {".sqlite", ".db", ".sqlite3"}:
            rows = _read_projection_from_sqlite(resolved, table_name, source_fields)
        elif suffix == ".csv":
            rows = _read_projection_from_csv(resolved, source_fields)
        elif suffix == ".json":
            rows = _read_projection_from_json(resolved, source_fields)
        if not rows:
            continue
        if not _plan_allows_source_projection(
            analysis_plan,
            len(rows),
            allow_bound_scope_requirements=allow_bound_scope_requirements,
        ):
            continue
        audit = {
            "source_paths": [source_path],
            "operations": (
                _plan_operation_labels(analysis_plan)
                if _plan_is_transform(analysis_plan)
                else ["source_preserve_projection"]
            ),
            "output_row_count": len(rows),
            "output_hash": answer_value_hash(columns, rows),
            "audit_origin": "execute_python_preserve_fallback",
        }
        prepared_answer, answer_error = validate_prepared_answer(
            columns,
            rows,
            analysis_plan,
            audit,
        )
        if prepared_answer is not None and answer_error is None:
            return _canonicalize_prepared_answer_field_casing(
                prepared_answer,
                workspace=workspace,
                analysis_plan=analysis_plan,
            )
    return None


def _plan_allows_source_projection(
    analysis_plan: dict[str, Any],
    row_count: int,
    *,
    allow_bound_scope_requirements: bool = False,
) -> bool:
    if _plan_preserves_source_rows(analysis_plan):
        output_spec = analysis_plan.get("output_spec") or {}
        expected_row_count = (
            output_spec.get("expected_row_count")
            if isinstance(output_spec, dict)
            else None
        )
        row_count_matches = expected_row_count is None or expected_row_count == row_count
        if allow_bound_scope_requirements:
            return row_count_matches and not _plan_has_executable_requirements(
                analysis_plan,
            )
        return (
            not _plan_has_constraining_requirements(analysis_plan)
            and row_count_matches
        )
    output_spec = analysis_plan.get("output_spec") or {}
    if not isinstance(output_spec, dict):
        return False
    if output_spec.get("sort_keys"):
        return False
    expected_row_count = output_spec.get("expected_row_count")
    if expected_row_count is not None and expected_row_count != row_count:
        return False
    operation_names = {
        str(item.get("operation") or "").casefold()
        for item in _plan_operation_labels(analysis_plan)
        if isinstance(item, dict) and str(item.get("operation") or "").strip()
    }
    return bool(operation_names) and operation_names.issubset({"filter"})


def _plan_has_executable_requirements(analysis_plan: dict[str, Any]) -> bool:
    intent = analysis_plan.get("intent")
    if not isinstance(intent, Mapping):
        return False
    executable_types = {
        "calculation",
        "deduplication",
        "filter",
        "grouping",
        "limit",
        "ordering",
        "selector",
        "time_range",
        "value",
    }
    for requirement in intent.get("requirements") or []:
        if not isinstance(requirement, Mapping):
            continue
        requirement_type = str(requirement.get("requirement_type") or "").casefold()
        if requirement_type in executable_types:
            return True
    return False


def create_execute_python_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """创建绑定到当前临时工作区的 execute_python 模型工具。"""

    workspace = workspace.resolve()

    @tool("execute_python", description=load_tool_prompt("execute_python"))
    def execute_python(
        code: str,
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> ToolMessage | Command[BenchmarkDeepAgentState]:
        """Run the execute_python tool."""

        if not code.strip():
            return ToolMessage(
                content="code must be a non-empty Python source string.",
                name="execute_python",
                tool_call_id=tool_call_id,
                status="error",
            )

        try:
            executable_code = _rewrite_virtual_python_paths(code, workspace)
        except (SyntaxError, ValueError) as exc:
            return ToolMessage(
                content=f"Invalid Python source or virtual path: {exc}",
                name="execute_python",
                tool_call_id=tool_call_id,
                status="error",
            )

        scratch_dir = workspace / "scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        result_path = scratch_dir / f"prepared-answer-{tool_call_id}.json"
        candidate_path = _write_answer_candidate_artifact(
            workspace=workspace,
            state=state,
        )
        code_context_paths = _extract_context_paths(code)
        observed_sources = _observed_sources_from_context_paths(code_context_paths)

        def with_observed(update: dict[str, Any]) -> dict[str, Any]:
            if not observed_sources:
                return update
            return {
                "observed_sources": merge_observed_sources(
                    state.get("observed_sources"),
                    observed_sources,
                ),
                **update,
            }

        auto_capture_answer = _code_declares_answer_candidate_shape(code) and bool(
            isinstance(state.get("analysis_plan"), dict)
            or isinstance(state.get("question_structure"), Mapping)
            or state.get("answer_candidate") is not None
        )
        capture_suffix = (
            "\n\ntry:\n"
            "    __auto_capture_answer()\n"
            "except NameError:\n"
            "    pass\n"
            if auto_capture_answer
            else ""
        )
        executable_code = (
            f"{_answer_helper_source(result_path, auto_capture=auto_capture_answer, candidate_path=candidate_path)}\n"
            f"{executable_code}"
            f"{capture_suffix}"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-X", "utf8", "-I", "-B", "-"],
                cwd=workspace,
                env=_build_shell_environment(),
                input=executable_code.encode("utf-8"),
                capture_output=True,
                check=False,
                timeout=config.execute_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            content = _format_process_output(
                exc.stdout or b"",
                exc.stderr or b"",
                exit_code=None,
                max_output_bytes=config.max_output_bytes,
            )
            return ToolMessage(
                content=(
                    f"{content}\n\nPython script timed out after "
                    f"{config.execute_timeout_seconds} seconds."
                ),
                name="execute_python",
                tool_call_id=tool_call_id,
                status="error",
            )
        except OSError as exc:
            return ToolMessage(
                content=f"Failed to start Python: {exc}",
                name="execute_python",
                tool_call_id=tool_call_id,
                status="error",
            )

        content = _format_process_output(
            completed.stdout,
            completed.stderr,
            exit_code=completed.returncode,
            max_output_bytes=config.max_output_bytes,
        )

        def candidate_plan_command(recovered_from: str) -> Command[BenchmarkDeepAgentState] | None:
            candidate_plan_answer = _try_candidate_plan_execution(
                workspace=workspace,
                analysis_plan=state.get("analysis_plan"),
                candidate=state.get("answer_candidate"),
            )
            if candidate_plan_answer is None:
                return None
            summary = json.dumps(
                {
                    "status": "prepared_from_answer_candidate_plan_execution",
                    "column_count": len(candidate_plan_answer.columns),
                    "row_count": len(candidate_plan_answer.rows),
                    "recovered_from": recovered_from,
                },
                ensure_ascii=False,
            )
            return Command(
                update=with_observed({
                    "prepared_answer": candidate_plan_answer,
                    "answer_candidate": None,
                    "messages": [
                        ToolMessage(
                            content=f"{content}\n\n{summary}",
                            name="execute_python",
                            tool_call_id=tool_call_id,
                            status="success",
                        )
                    ],
                })
            )

        if completed.returncode == 0:
            if (command := candidate_plan_command("answer_candidate_priority")) is not None:
                return command
            prepared_answer, answer_error, answer_candidate = _load_prepared_answer(
                workspace=workspace,
                result_path=result_path,
                analysis_plan=state.get("analysis_plan"),
                code_context_paths=code_context_paths,
            )
            if prepared_answer is None and answer_error is None:
                if (
                    command := candidate_plan_command("no_set_answer_submission")
                ) is not None:
                    return command
                stdout_answer, stdout_error, stdout_candidate = _recover_stdout_answer(
                    workspace=workspace,
                    stdout=completed.stdout,
                    analysis_plan=state.get("analysis_plan"),
                    code_context_paths=code_context_paths,
                )
                if stdout_answer is not None:
                    summary = json.dumps(
                        {
                            "status": "prepared_from_stdout_literal",
                            "column_count": len(stdout_answer.columns),
                            "row_count": len(stdout_answer.rows),
                            "recovered_from": "no_set_answer_submission",
                        },
                        ensure_ascii=False,
                    )
                    return Command(
                        update=with_observed({
                            "prepared_answer": stdout_answer,
                            "answer_candidate": None,
                            "messages": [
                                ToolMessage(
                                    content=f"{content}\n\n{summary}",
                                    name="execute_python",
                                    tool_call_id=tool_call_id,
                                    status="success",
                                )
                            ],
                        })
                    )
                if stdout_candidate is not None:
                    summary = json.dumps(
                        {
                            "status": "candidate_saved",
                            "column_count": stdout_candidate.get("column_count"),
                            "row_count": stdout_candidate.get("row_count"),
                            "validation_error": stdout_error,
                            "recovered_from": "stdout_literal",
                        },
                        ensure_ascii=False,
                    )
                    return Command(
                        update=with_observed({
                            "answer_candidate": stdout_candidate,
                            "messages": [
                                ToolMessage(
                                    content=f"{content}\n\n{stdout_error}\n\n{summary}",
                                    name="execute_python",
                                    tool_call_id=tool_call_id,
                                    status="error",
                                )
                            ],
                        })
                    )
                fallback_answer = _try_preserve_source_projection(
                    workspace=workspace,
                    analysis_plan=state.get("analysis_plan"),
                )
                if fallback_answer is not None:
                    summary = json.dumps(
                        {
                            "status": "prepared_from_source_projection",
                            "column_count": len(fallback_answer.columns),
                            "row_count": len(fallback_answer.rows),
                            "recovered_from": "no_set_answer_submission",
                        },
                        ensure_ascii=False,
                    )
                    return Command(
                        update=with_observed({
                            "prepared_answer": fallback_answer,
                            "answer_candidate": None,
                            "messages": [
                                ToolMessage(
                                    content=f"{content}\n\n{summary}",
                                    name="execute_python",
                                    tool_call_id=tool_call_id,
                                    status="success",
                                )
                            ],
                        })
                    )
                if auto_capture_answer:
                    summary = json.dumps(
                        {
                            "status": "missing_answer_submission",
                            "recovery": (
                                "Call set_answer(columns, rows, audit=...) or define "
                                "columns and rows/result so execute_python can capture "
                                "a candidate table."
                            ),
                        },
                        ensure_ascii=False,
                    )
                    return Command(
                        update=with_observed({
                            "messages": [
                                ToolMessage(
                                    content=(
                                        f"{content}\n\n"
                                        "execute_python completed after an active "
                                        "analysis_plan, but did not produce an answer "
                                        "table.\n\n"
                                        f"{summary}"
                                    ),
                                    name="execute_python",
                                    tool_call_id=tool_call_id,
                                    status="error",
                                )
                            ],
                        })
                    )
            if answer_error is not None:
                if (command := candidate_plan_command(answer_error)) is not None:
                    return command
                fallback_answer = _try_preserve_source_projection(
                    workspace=workspace,
                    analysis_plan=state.get("analysis_plan"),
                )
                if fallback_answer is not None:
                    summary = json.dumps(
                        {
                            "status": "prepared_from_source_projection",
                            "column_count": len(fallback_answer.columns),
                            "row_count": len(fallback_answer.rows),
                            "recovered_from": answer_error,
                        },
                        ensure_ascii=False,
                    )
                    return Command(
                        update=with_observed({
                            "prepared_answer": fallback_answer,
                            "answer_candidate": None,
                            "messages": [
                                ToolMessage(
                                    content=f"{content}\n\n{summary}",
                                    name="execute_python",
                                    tool_call_id=tool_call_id,
                                    status="success",
                                )
                            ],
                        })
                    )
                candidate_summary = {
                    "status": "candidate_saved",
                    "column_count": (
                        answer_candidate.get("column_count")
                        if answer_candidate is not None
                        else None
                    ),
                    "row_count": (
                        answer_candidate.get("row_count")
                        if answer_candidate is not None
                        else None
                    ),
                    "validation_error": answer_error,
                    "recovery": (
                        "Revise analysis_plan to the observed candidate shape or "
                        "call finalize_answer_candidate to submit a projected "
                        "candidate table."
                    ),
                }
                candidate_to_store = _candidate_for_update(
                    existing=state.get("answer_candidate"),
                    new=answer_candidate,
                    analysis_plan=state.get("analysis_plan"),
                )
                return Command(
                    update=with_observed({
                        **(
                            {"answer_candidate": candidate_to_store}
                            if candidate_to_store is not None
                            else {}
                        ),
                        "messages": [
                            ToolMessage(
                                content=(
                                    f"{content}\n\n{answer_error}\n\n"
                                    f"{json.dumps(candidate_summary, ensure_ascii=False)}"
                                ),
                                name="execute_python",
                                tool_call_id=tool_call_id,
                                status="error",
                            )
                        ],
                    })
                )
            if prepared_answer is not None:
                fallback_answer = _try_preserve_source_projection(
                    workspace=workspace,
                    analysis_plan=state.get("analysis_plan"),
                )
                if fallback_answer is not None:
                    prepared_answer = fallback_answer
                summary = json.dumps(
                    {
                        "status": (
                            "prepared_from_source_projection"
                            if fallback_answer is not None
                            else "prepared"
                        ),
                        "column_count": len(prepared_answer.columns),
                        "row_count": len(prepared_answer.rows),
                    },
                    ensure_ascii=False,
                )
                return Command(
                    update=with_observed({
                        "prepared_answer": prepared_answer,
                        "answer_candidate": None,
                        "messages": [
                            ToolMessage(
                                content=f"{content}\n\n{summary}",
                                name="execute_python",
                                tool_call_id=tool_call_id,
                                status="success",
                            )
                        ],
                    })
                )
        if (command := candidate_plan_command("python_error")) is not None:
            return command
        message = ToolMessage(
            content=content,
            name="execute_python",
            tool_call_id=tool_call_id,
            status="success" if completed.returncode == 0 else "error",
        )
        if observed_sources:
            return Command(
                update=with_observed(
                    {
                        "messages": [message],
                    }
                )
            )
        return message

    return execute_python
