from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
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
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools.answer import (
    answer_value_hash,
    normalize_answer_columns,
    validate_prepared_answer,
)

_CONTEXT_PATH_RE = re.compile(r"/context/[^\s\"'<>),;\]]+")


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


def _answer_helper_source(result_path: Path, *, auto_capture: bool) -> str:
    """向隔离脚本注入结果提交函数，完整数据只写入临时工作区。"""

    context_root = result_path.parent.parent / "context"
    helper_source = f"""
import hashlib as __answer_hashlib
import json as __answer_json
import math as __answer_math
from pathlib import Path as __AnswerPath
__ContextRoot = __AnswerPath({str(context_root)!r}).resolve()

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
    __AnswerPath({str(result_path)!r}).write_text(
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
        return None
    return {
        "source_paths": source_paths,
        "operations": operations or ["source_bound_projection"],
        "output_row_count": len(rows),
        "output_hash": answer_value_hash(columns, rows),
        "audit_origin": "execute_python_static_context_paths",
    }


def _load_prepared_answer(
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
    prepared_answer, answer_error = validate_prepared_answer(
        normalized_columns,
        normalized_rows,
        analysis_plan,
        audit if isinstance(audit, dict) else None,
    )
    if answer_error is not None:
        candidate["validation_error"] = answer_error
    return prepared_answer, answer_error, candidate


def create_execute_python_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """创建绑定到当前临时工作区的 execute_python 模型工具。"""

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

        result_path = workspace / "scratch" / f"prepared-answer-{tool_call_id}.json"
        code_context_paths = _extract_context_paths(code)
        auto_capture_answer = isinstance(state.get("analysis_plan"), dict) and bool(
            state.get("todos")
        )
        executable_code = (
            f"{_answer_helper_source(result_path, auto_capture=auto_capture_answer)}\n"
            f"{executable_code}"
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
        if completed.returncode == 0:
            prepared_answer, answer_error, answer_candidate = _load_prepared_answer(
                result_path,
                state.get("analysis_plan"),
                code_context_paths,
            )
            if answer_error is not None:
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
                return Command(
                    update={
                        **(
                            {"answer_candidate": answer_candidate}
                            if answer_candidate is not None
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
                    }
                )
            if prepared_answer is not None:
                summary = json.dumps(
                    {
                        "status": "prepared",
                        "column_count": len(prepared_answer.columns),
                        "row_count": len(prepared_answer.rows),
                    },
                    ensure_ascii=False,
                )
                return Command(
                    update={
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
                    }
                )
        return ToolMessage(
            content=content,
            name="execute_python",
            tool_call_id=tool_call_id,
            status="success" if completed.returncode == 0 else "error",
        )

    return execute_python
