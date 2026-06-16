from __future__ import annotations

import ast
import json
import os
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
from data_agent_baseline.tools.answer import validate_prepared_answer


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


def _answer_helper_source(result_path: Path) -> str:
    """向隔离脚本注入结果提交函数，完整数据只写入临时工作区。"""

    return f"""
import json as __answer_json
import math as __answer_math
from pathlib import Path as __AnswerPath

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

def set_answer(columns, rows):
    payload = {{
        "columns": [str(column) for column in columns],
        "rows": __normalize_answer_value(list(rows)),
    }}
    __AnswerPath({str(result_path)!r}).write_text(
        __answer_json.dumps(payload, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
"""


def _load_prepared_answer(
    result_path: Path,
    analysis_plan: dict[str, Any] | None,
) -> tuple[Any | None, str | None]:
    if not result_path.exists():
        return None, None
    if analysis_plan is None:
        return None, "set_answer requires a successful analysis_plan first."
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"set_answer produced an unreadable result: {exc}"
    columns = payload.get("columns")
    rows = payload.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return None, "set_answer requires list columns and list rows."
    normalized_columns = [str(column) for column in columns]
    if all(isinstance(row, dict) for row in rows):
        normalized_rows = [
            [row.get(column) for column in normalized_columns]
            for row in rows
        ]
    elif all(isinstance(row, list) for row in rows):
        normalized_rows = [list(row) for row in rows]
    else:
        return None, "set_answer rows must contain only row lists or row objects."
    return validate_prepared_answer(
        normalized_columns,
        normalized_rows,
        analysis_plan,
    )


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
        executable_code = f"{_answer_helper_source(result_path)}\n{executable_code}"
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
            prepared_answer, answer_error = _load_prepared_answer(
                result_path,
                state.get("analysis_plan"),
            )
            if answer_error is not None:
                return ToolMessage(
                    content=f"{content}\n\n{answer_error}",
                    name="execute_python",
                    tool_call_id=tool_call_id,
                    status="error",
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
