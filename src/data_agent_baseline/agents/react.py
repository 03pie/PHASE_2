from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, NotRequired

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.graph import DeepAgentState
from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask

TraceCallback = Callable[[AgentRunResult, str], None]


DEEP_AGENT_SYSTEM_PROMPT = """
You are a data agent solving a benchmark task from local task assets.

Required workflow:
1. Analyze the question's intent before using data tools. Identify the requested
   entities, measures, filters, time range, grouping, ordering, units, precision,
   and exact output-table shape. Resolve wording against the supplied data rather
   than silently changing the requested meaning. Your first and only first-round
   action must be one `analyze_plan` call that records this interpretation, the
   output specification, execution steps, and possible delegations.
2. Your second-round action must be one `write_todos` call that converts the
   analysis plan into an actionable todo list. No data, execution, delegation, or
   answer tools are available until this succeeds. Keep the list updated as major
   steps finish or new evidence changes the approach.
3. Delegate complex or independent steps with `task`. Good delegation candidates
   include multi-source investigation, database schema exploration, long-document
   research, non-trivial calculations, and independent result verification. Give
   each subagent a narrow objective, relevant candidate files, required output,
   and calculation or citation expectations. Do not delegate trivial reads.
4. Inspect only the files relevant to the plan, compute the result, then validate
   filters, units, row count, ordering, and arithmetic before submission.
5. Submit the final table with `answer`.

Tool and data rules:
1. The first user message contains a complete recursive inventory of `/context/`.
   Do not spend model calls listing directories again.
2. Use `read_file`, `glob`, and `grep` to inspect relevant task files.
3. Shell commands and persistent script files are unavailable. Use
   `execute_python(code=...)` to execute Python source directly.
4. Inside Python code, use the same virtual paths as the file tools:
   `/context/...` for task data and `/scratch/...` for temporary outputs. The
   executor maps these paths to the isolated task workspace on every operating
   system. Python standard output and standard error use UTF-8. Do not use shell
   commands or subprocesses.
5. Treat subagent reports as evidence to verify, not automatically as the final
   answer. Reconcile conflicting findings before submission.
6. Only the main agent can call `answer`. A plain-text final response is not a valid
   answer.
7. Call `answer` exactly once after validation. Do not call it in parallel with
   other tools.
8. Base the answer only on information observed in `/context/`.
""".strip()

SUBAGENT_SYSTEM_PROMPT = """
You are the general-purpose analysis subagent for a benchmark data task.

Focus only on the delegated objective. First identify its requested scope, filters,
units, and expected output. Use `write_todos` when the delegated work has multiple
steps. Inspect the relevant files under `/context/`, perform calculations with
`execute_python(code=...)`, and verify the result before returning.

Your report to the main agent must be concise and include:
- the result or finding;
- the source files, tables, or fields used;
- the calculation and filtering rules applied;
- assumptions, ambiguities, or unresolved issues.

Directory listing, shell commands, and persistent script files are unavailable.
Use `glob` for recursive discovery when needed. Python code should use virtual paths
such as `/context/data.csv` and `/scratch/output.json`. Do not attempt to submit the
final answer.
""".strip()


@dataclass(frozen=True, slots=True)
class DeepAgentConfig:
    max_steps: int = 16
    execute_timeout_seconds: int = 30
    max_output_bytes: int = 100_000

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if self.execute_timeout_seconds < 1:
            raise ValueError("execute_timeout_seconds must be at least 1.")
        if self.max_output_bytes < 1:
            raise ValueError("max_output_bytes must be at least 1.")


class BenchmarkDeepAgentState(DeepAgentState):
    answer: NotRequired[AnswerTable | None]
    analysis_plan: NotRequired[dict[str, Any]]
    todos: NotRequired[list[dict[str, str]]]


@tool("analyze_plan")
def _analyze_plan_tool(
    intent: str,
    output_spec: str,
    steps: list[str],
    delegation_candidates: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[BenchmarkDeepAgentState] | ToolMessage:
    """Record the question interpretation and execution plan before data access."""
    if not intent.strip() or not output_spec.strip():
        return ToolMessage(
            content="intent and output_spec must be non-empty.",
            name="analyze_plan",
            tool_call_id=tool_call_id,
            status="error",
        )
    normalized_steps = [step.strip() for step in steps if step.strip()]
    if not normalized_steps:
        return ToolMessage(
            content="steps must contain at least one non-empty plan step.",
            name="analyze_plan",
            tool_call_id=tool_call_id,
            status="error",
        )

    plan = {
        "intent": intent.strip(),
        "output_spec": output_spec.strip(),
        "steps": normalized_steps,
        "delegation_candidates": [
            item.strip() for item in delegation_candidates if item.strip()
        ],
    }
    return Command(
        update={
            "analysis_plan": plan,
            "messages": [
                ToolMessage(
                    content=json.dumps(plan, ensure_ascii=False),
                    name="analyze_plan",
                    tool_call_id=tool_call_id,
                    status="success",
                )
            ],
        }
    )


@tool("answer")
def _answer_tool(
    columns: list[str],
    rows: list[list[Any]],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[BenchmarkDeepAgentState] | ToolMessage:
    """Submit the final answer table and terminate the benchmark task."""
    if not columns or not all(isinstance(column, str) and column for column in columns):
        return ToolMessage(
            content="answer.columns must be a non-empty list of non-empty strings.",
            name="answer",
            tool_call_id=tool_call_id,
            status="error",
        )

    if not rows:
        return ToolMessage(
            content="answer.rows must contain at least one row.",
            name="answer",
            tool_call_id=tool_call_id,
            status="error",
        )

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if len(row) != len(columns):
            return ToolMessage(
                content="Each answer row must match the number of columns.",
                name="answer",
                tool_call_id=tool_call_id,
                status="error",
            )
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    content = json.dumps(
        {
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        ensure_ascii=False,
    )
    return Command(
        update={
            "answer": answer,
            "messages": [
                ToolMessage(
                    content=content,
                    name="answer",
                    tool_call_id=tool_call_id,
                    status="success",
                )
            ],
        }
    )


class AnswerMiddleware(AgentMiddleware[BenchmarkDeepAgentState, None, Any]):
    state_schema = BenchmarkDeepAgentState
    tools = [_answer_tool]

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: BenchmarkDeepAgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        del runtime
        if state.get("answer") is not None:
            return {"jump_to": "end"}
        return None


class PlanningMiddleware(AgentMiddleware[BenchmarkDeepAgentState, None, Any]):
    state_schema = BenchmarkDeepAgentState
    tools = [_analyze_plan_tool]

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        if request.state.get("analysis_plan") is None:
            planning_tools = [
                item for item in request.tools if _tool_name(item) == "analyze_plan"
            ]
            request = request.override(tools=planning_tools)
        elif not request.state.get("todos"):
            todo_tools = [
                item for item in request.tools if _tool_name(item) == "write_todos"
            ]
            request = request.override(tools=todo_tools)
        return handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_name = str(request.tool_call.get("name") or "")
        if tool_name != "analyze_plan" and request.state.get("analysis_plan") is None:
            return ToolMessage(
                content="Call analyze_plan successfully before using any other tool.",
                name=tool_name,
                tool_call_id=str(request.tool_call.get("id") or ""),
                status="error",
            )
        if (
            request.state.get("analysis_plan") is not None
            and not request.state.get("todos")
            and tool_name != "write_todos"
        ):
            return ToolMessage(
                content="Call write_todos successfully before using any other tool.",
                name=tool_name,
                tool_call_id=str(request.tool_call.get("id") or ""),
                status="error",
            )
        return handler(request)


def _build_shell_environment() -> dict[str, str]:
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


def _create_execute_python_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    @tool("execute_python")
    def execute_python(
        code: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> ToolMessage:
        """Execute Python source directly without a shell or persistent script file."""
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
        return ToolMessage(
            content=content,
            name="execute_python",
            tool_call_id=tool_call_id,
            status="success" if completed.returncode == 0 else "error",
        )

    return execute_python


def _tool_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("function", {}).get("name") or "")
    return str(getattr(value, "name", ""))


class HideUnavailableToolsMiddleware(AgentMiddleware[Any, None, Any]):
    hidden_tools = frozenset({"execute", "ls", "write_file", "edit_file"})

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        filtered_tools = [
            item for item in request.tools if _tool_name(item) not in self.hidden_tools
        ]
        return handler(request.override(tools=filtered_tools))


def _workspace_permissions() -> list[FilesystemPermission]:
    context_paths = ["/context/**"]
    return [
        FilesystemPermission(operations=["read"], paths=context_paths, mode="allow"),
        FilesystemPermission(operations=["write"], paths=context_paths, mode="deny"),
    ]


def _context_inventory(context_dir: Path) -> str:
    entries: list[str] = []
    for path in sorted(context_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(context_dir).as_posix()
        entries.append(f"- /context/{relative_path} ({path.stat().st_size} bytes)")
    return "\n".join(entries) if entries else "- <no context files>"


def _task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n\n"
        "Complete recursive context inventory:\n"
        f"{_context_inventory(task.context_dir)}\n\n"
        "Begin by identifying the requested output and forming a compact plan. Use the "
        "`general-purpose` subagent for complex or independent analysis steps. Do not list "
        "directories again. Inspect only relevant files, validate the result, and submit "
        "the final table with `answer`."
    )


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _serialize_message(message: BaseMessage) -> dict[str, Any]:
    return _json_safe(message.model_dump(mode="json", exclude_none=True))


def _serialize_tool_definition(value: BaseTool | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return _json_safe(value)
    args_schema: dict[str, Any] | None = None
    if value.args_schema is not None:
        try:
            args_schema = _json_safe(value.args_schema.model_json_schema())
        except (AttributeError, TypeError, ValueError):
            args_schema = None
    return {
        "name": value.name,
        "description": value.description,
        "args_schema": args_schema,
    }


def _serialize_tool_result(value: ToolMessage | Command[Any]) -> dict[str, Any]:
    if isinstance(value, ToolMessage):
        return _serialize_message(value)
    update = value.update
    if isinstance(update, dict):
        serialized_update = dict(update)
        messages = serialized_update.get("messages")
        if isinstance(messages, list):
            serialized_update["messages"] = [
                _serialize_message(message)
                if isinstance(message, BaseMessage)
                else _json_safe(message)
                for message in messages
            ]
    else:
        serialized_update = _json_safe(update)
    return {
        "type": "command",
        "graph": value.graph,
        "update": _json_safe(serialized_update),
        "goto": _json_safe(value.goto),
    }


@dataclass(slots=True)
class TraceEvent:
    sequence: int
    action: str
    scope: str
    llm_call_index: int | None
    thought: str
    action_input: dict[str, Any]
    tool_call_id: str | None
    raw_response: dict[str, Any]
    observation: dict[str, Any]
    ok: bool


class TraceEventCollector:
    def __init__(self) -> None:
        self._lock = Lock()
        self._events: list[TraceEvent] = []
        self._next_sequence = 1
        self._next_llm_call_index = 1
        self._initialized_scopes: set[str] = set()
        self._tool_call_llm_indexes: dict[str, int] = {}
        self._llm_events: dict[int, TraceEvent] = {}

    def next_llm_call_index(self) -> int:
        with self._lock:
            call_index = self._next_llm_call_index
            self._next_llm_call_index += 1
            return call_index

    def initialize_scope(
        self,
        *,
        scope: str,
        system_message: BaseMessage | None,
        messages: list[BaseMessage],
        llm_call_index: int,
    ) -> None:
        with self._lock:
            if scope in self._initialized_scopes:
                return
            self._initialized_scopes.add(scope)
        if system_message is not None:
            self.add(
                action="system_prompt",
                scope=scope,
                llm_call_index=llm_call_index,
                action_input={"message": _serialize_message(system_message)},
            )
        for message in messages:
            if isinstance(message, HumanMessage):
                self.add(
                    action="user_prompt",
                    scope=scope,
                    llm_call_index=llm_call_index,
                    action_input={"message": _serialize_message(message)},
                )
                break

    def add(
        self,
        *,
        action: str,
        scope: str,
        llm_call_index: int | None = None,
        thought: str = "",
        action_input: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
        raw_response: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        ok: bool = True,
    ) -> None:
        with self._lock:
            event = TraceEvent(
                sequence=self._next_sequence,
                action=action,
                scope=scope,
                llm_call_index=llm_call_index,
                thought=thought,
                action_input=action_input or {},
                tool_call_id=tool_call_id,
                raw_response=raw_response or {},
                observation=observation or {},
                ok=ok,
            )
            self._next_sequence += 1
            self._events.append(event)

    def register_tool_calls(
        self,
        messages: list[BaseMessage],
        *,
        llm_call_index: int,
    ) -> None:
        with self._lock:
            for message in messages:
                if not isinstance(message, AIMessage):
                    continue
                for tool_call in message.tool_calls:
                    tool_call_id = str(tool_call.get("id") or "")
                    if tool_call_id:
                        self._tool_call_llm_indexes[tool_call_id] = llm_call_index

    def start_llm_call(
        self,
        *,
        scope: str,
        llm_call_index: int,
        request_payload: dict[str, Any],
    ) -> None:
        with self._lock:
            event = TraceEvent(
                sequence=self._next_sequence,
                action="llm_pending",
                scope=scope,
                llm_call_index=llm_call_index,
                thought="",
                action_input={"request": request_payload},
                tool_call_id=None,
                raw_response={},
                observation={"status": "requesting", "tool_calls": []},
                ok=True,
            )
            self._next_sequence += 1
            self._events.append(event)
            self._llm_events[llm_call_index] = event

    def complete_llm_call(
        self,
        *,
        llm_call_index: int,
        response_payload: dict[str, Any],
        messages: list[BaseMessage],
        visible_text: str,
    ) -> None:
        tool_calls: list[dict[str, Any]] = []
        with self._lock:
            event = self._llm_events[llm_call_index]
            event.thought = visible_text
            event.raw_response = response_payload
            tool_names: list[str] = []
            for message in messages:
                if not isinstance(message, AIMessage):
                    continue
                for tool_call in message.tool_calls:
                    tool_name = str(tool_call.get("name") or "")
                    if tool_name:
                        tool_names.append(tool_name)
                    tool_call_id = str(tool_call.get("id") or "")
                    if tool_call_id:
                        self._tool_call_llm_indexes[tool_call_id] = llm_call_index
                    tool_calls.append(
                        {
                            "name": tool_name,
                            "tool_call_id": tool_call_id or None,
                            "args": _json_safe(tool_call.get("args")),
                            "status": "pending",
                            "ok": None,
                            "result": None,
                        }
                    )
            event.action = "+".join(tool_names) if tool_names else "llm_response"
            event.observation = {
                "status": "tools_pending" if tool_calls else "completed",
                "tool_calls": tool_calls,
            }

    def fail_llm_call(
        self,
        *,
        llm_call_index: int,
        error: BaseException,
    ) -> None:
        with self._lock:
            event = self._llm_events[llm_call_index]
            event.ok = False
            event.observation = {
                "status": "error",
                "error": str(error),
                "type": type(error).__name__,
                "tool_calls": [],
            }

    def complete_tool_call(
        self,
        *,
        tool_call_id: str,
        result: dict[str, Any],
        ok: bool,
    ) -> None:
        with self._lock:
            llm_call_index = self._tool_call_llm_indexes.get(tool_call_id)
            if llm_call_index is None:
                return
            event = self._llm_events[llm_call_index]
            tool_calls = event.observation.get("tool_calls", [])
            for tool_call in tool_calls:
                if tool_call.get("tool_call_id") != tool_call_id:
                    continue
                tool_call["status"] = "success" if ok else "error"
                tool_call["ok"] = ok
                tool_call["result"] = result
                break
            statuses = [item.get("status") for item in tool_calls]
            if statuses and all(status != "pending" for status in statuses):
                event.observation["status"] = "completed"
            event.ok = all(item.get("ok") is not False for item in tool_calls)

    def snapshot(self) -> list[StepRecord]:
        with self._lock:
            events = list(self._events)
        return [
            StepRecord(
                step_index=index,
                thought=event.thought,
                action=event.action,
                action_input={
                    "scope": event.scope,
                    "llm_call_index": event.llm_call_index,
                    **event.action_input,
                },
                tool_call_id=event.tool_call_id,
                raw_response=event.raw_response,
                observation=event.observation,
                ok=event.ok,
            )
            for index, event in enumerate(events, start=1)
        ]


class LlmTraceMiddleware(AgentMiddleware[Any, None, Any]):
    def __init__(self, collector: TraceEventCollector, *, scope: str) -> None:
        self.collector = collector
        self.scope = scope

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        call_index = self.collector.next_llm_call_index()
        self.collector.initialize_scope(
            scope=self.scope,
            system_message=request.system_message,
            messages=request.messages,
            llm_call_index=call_index,
        )
        request_payload = {
            "model": {
                "class": type(request.model).__name__,
                "identifier": str(
                    getattr(request.model, "model_name", None)
                    or getattr(request.model, "model", None)
                    or ""
                ),
            },
            "system_message": (
                _serialize_message(request.system_message)
                if request.system_message is not None
                else None
            ),
            "messages": [_serialize_message(message) for message in request.messages],
            "tools": [_serialize_tool_definition(item) for item in request.tools],
            "tool_choice": _json_safe(request.tool_choice),
            "model_settings": _json_safe(request.model_settings),
        }
        self.collector.start_llm_call(
            scope=self.scope,
            llm_call_index=call_index,
            request_payload=request_payload,
        )
        try:
            response = handler(request)
        except BaseException as exc:  # noqa: BLE001
            self.collector.fail_llm_call(
                llm_call_index=call_index,
                error=exc,
            )
            raise

        serialized_result = [_serialize_message(message) for message in response.result]
        visible_text = "\n".join(
            _visible_text(message)
            for message in response.result
            if isinstance(message, AIMessage) and _visible_text(message)
        )
        self.collector.complete_llm_call(
            llm_call_index=call_index,
            messages=response.result,
            visible_text=visible_text,
            response_payload={
                "messages": serialized_result,
                "structured_response": _json_safe(response.structured_response),
            },
        )
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_call_id = str(request.tool_call.get("id") or "")
        try:
            result = handler(request)
        except BaseException as exc:  # noqa: BLE001
            self.collector.complete_tool_call(
                tool_call_id=tool_call_id,
                result={"error": str(exc), "type": type(exc).__name__},
                ok=False,
            )
            raise

        serialized_result = _serialize_tool_result(result)
        ok = not (
            isinstance(result, ToolMessage)
            and getattr(result, "status", "success") == "error"
        )
        self.collector.complete_tool_call(
            tool_call_id=tool_call_id,
            result=serialized_result,
            ok=ok,
        )
        return result


def _visible_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    text_parts: list[str] = []
    for block in message.content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in {"text", "output_text"} and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
    return "\n".join(text_parts)


def _normalize_answer(value: Any) -> AnswerTable | None:
    if isinstance(value, AnswerTable):
        return value
    if isinstance(value, dict):
        columns = value.get("columns")
        rows = value.get("rows")
        if isinstance(columns, list) and isinstance(rows, list):
            return AnswerTable(
                columns=[str(column) for column in columns],
                rows=[list(row) for row in rows if isinstance(row, list)],
            )
    return None


def _messages_from_state(state: dict[str, Any]) -> list[BaseMessage]:
    return [message for message in state.get("messages", []) if isinstance(message, BaseMessage)]


def _partial_run_result(
    task_id: str,
    state: dict[str, Any],
    collector: TraceEventCollector,
) -> AgentRunResult:
    return AgentRunResult(
        task_id=task_id,
        answer=_normalize_answer(state.get("answer")),
        steps=collector.snapshot(),
        failure_reason=None,
    )


class DeepAgent:
    def __init__(
        self,
        *,
        model: BaseChatModel,
        config: DeepAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.config = config or DeepAgentConfig()
        self.system_prompt = system_prompt or DEEP_AGENT_SYSTEM_PROMPT

    def _create_graph(
        self,
        workspace: Path,
        collector: TraceEventCollector,
    ):
        backend = FilesystemBackend(
            root_dir=workspace,
            virtual_mode=True,
        )
        execute_python_tool = _create_execute_python_tool(workspace, self.config)
        return create_deep_agent(
            model=self.model,
            tools=[execute_python_tool],
            system_prompt=self.system_prompt,
            backend=backend,
            permissions=_workspace_permissions(),
            subagents=[
                {
                    "name": "general-purpose",
                    "description": (
                        "Handle a focused, complex data-analysis or verification subtask. "
                        "Returns findings with source files, calculation rules, assumptions, "
                        "and unresolved issues; it cannot submit the final answer."
                    ),
                    "system_prompt": SUBAGENT_SYSTEM_PROMPT,
                    "tools": [execute_python_tool],
                    "middleware": [
                        HideUnavailableToolsMiddleware(),
                        LlmTraceMiddleware(
                            collector,
                            scope="subagent:general-purpose",
                        ),
                    ],
                }
            ],
            middleware=[
                PlanningMiddleware(),
                AnswerMiddleware(),
                HideUnavailableToolsMiddleware(),
                ModelCallLimitMiddleware(
                    run_limit=self.config.max_steps,
                    exit_behavior="end",
                ),
                LlmTraceMiddleware(collector, scope="main"),
            ],
            state_schema=BenchmarkDeepAgentState,
            name="dabench_deep_agent",
        )

    def run(
        self,
        task: PublicTask,
        trace_callback: TraceCallback | None = None,
    ) -> AgentRunResult:
        result: dict[str, Any] = {}
        last_partial_signature: str | None = None
        collector = TraceEventCollector()
        with tempfile.TemporaryDirectory(prefix=f"dabench-{task.task_id}-") as temp_dir:
            workspace = Path(temp_dir)
            shutil.copytree(task.context_dir, workspace / "context")
            (workspace / "scratch").mkdir()
            graph = self._create_graph(workspace, collector)

            try:
                for state in graph.stream(
                    {"messages": [HumanMessage(content=_task_prompt(task))]},
                    config={"recursion_limit": max(100, self.config.max_steps * 8)},
                    stream_mode="values",
                ):
                    if not isinstance(state, dict):
                        continue
                    result = state
                    if trace_callback is None:
                        continue

                    partial_result = _partial_run_result(
                        task.task_id,
                        state,
                        collector,
                    )
                    if not partial_result.steps and partial_result.answer is None:
                        continue
                    partial_signature = json.dumps(
                        partial_result.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                    if partial_signature == last_partial_signature:
                        continue
                    trace_callback(partial_result, "running")
                    last_partial_signature = partial_signature
            except GraphRecursionError as exc:
                failed_result = AgentRunResult(
                    task_id=task.task_id,
                    answer=None,
                    steps=collector.snapshot(),
                    failure_reason=f"Deep Agent graph recursion limit exceeded: {exc}",
                )
                if trace_callback is not None:
                    trace_callback(failed_result, "failed")
                return failed_result

        messages = _messages_from_state(result)
        answer = _normalize_answer(result.get("answer"))
        failure_reason: str | None = None
        if answer is None:
            model_call_count = int(result.get("run_model_call_count", 0))
            limit_reached = any(
                isinstance(message, AIMessage)
                and _visible_text(message).startswith("Model call limits exceeded:")
                for message in messages
            )
            if limit_reached or model_call_count >= self.config.max_steps:
                failure_reason = (
                    f"Agent did not submit an answer within {self.config.max_steps} model calls."
                )
            else:
                failure_reason = "Agent completed without calling the answer tool."

        run_result = AgentRunResult(
            task_id=task.task_id,
            answer=answer,
            steps=collector.snapshot(),
            failure_reason=failure_reason,
        )
        if trace_callback is not None:
            trace_callback(run_result, "completed" if run_result.succeeded else "failed")
        return run_result
