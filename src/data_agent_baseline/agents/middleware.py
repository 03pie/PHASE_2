from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from data_agent_baseline.agents.deep_state import BenchmarkDeepAgentState
from data_agent_baseline.tools.analyze_plan import analyze_plan_tool
from data_agent_baseline.tools.answer import answer_tool

_SOURCE_DISCOVERY_TOOLS = frozenset({"execute_python", "grep", "read_file"})
_CONTEXT_PATH_PATTERN = re.compile(r"""["'](/context/[^"']+)["']""")
_MAX_DISCOVERY_CALLS_BEFORE_PLAN = 3


@dataclass(frozen=True, slots=True)
class _DiscoveryState:
    knowledge_present: bool
    knowledge_checked: bool
    knowledge_available: bool
    context_sources: frozenset[str]
    discovery_call_count: int
    needs_cross_validation: bool

    @property
    def knowledge_ready(self) -> bool:
        return not self.knowledge_present or self.knowledge_checked

    @property
    def context_ready(self) -> bool:
        required = 2 if not self.knowledge_available or self.needs_cross_validation else 1
        return len(self.context_sources) >= required

    def tool_policy(self) -> tuple[set[str], str | None]:
        """根据已掌握的信息决定下一轮可见工具及是否强制调用。"""

        if not self.knowledge_ready:
            return {"read_file"}, "read_file"
        if not self.context_ready:
            return set(_SOURCE_DISCOVERY_TOOLS), None
        if (
            self.needs_cross_validation
            or self.discovery_call_count >= _MAX_DISCOVERY_CALLS_BEFORE_PLAN
        ):
            return {"analyze_plan"}, "analyze_plan"
        return set(_SOURCE_DISCOVERY_TOOLS) | {"analyze_plan"}, None


def tool_name(value: Any) -> str:
    """兼容 LangChain 工具对象和字典形式的工具定义。"""

    if isinstance(value, dict):
        return str(value.get("name") or value.get("function", {}).get("name") or "")
    return str(getattr(value, "name", ""))


def _tool_error(request: ToolCallRequest, content: str) -> ToolMessage:
    return ToolMessage(
        content=content,
        name=str(request.tool_call.get("name") or ""),
        tool_call_id=str(request.tool_call.get("id") or ""),
        status="error",
    )


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
    )


def _context_sources(tool: str, arguments: dict[str, Any]) -> set[str]:
    """从成功的探索调用中提取实际检查过的上下文来源。"""

    if tool in {"read_file", "grep"}:
        path = str(
            arguments.get("file_path")
            or arguments.get("path")
            or ""
        ).replace("\\", "/")
        if path.startswith("/context/") and not path.lower().endswith("/knowledge.md"):
            return {path}
        return set()
    if tool == "execute_python":
        code = str(arguments.get("code") or "")
        return {
            path.replace("\\", "/")
            for path in _CONTEXT_PATH_PATTERN.findall(code)
            if not path.lower().endswith("/knowledge.md")
        }
    return set()


def _discovery_state(messages: list[BaseMessage]) -> _DiscoveryState:
    """汇总计划前已经检查的 knowledge 和独立数据来源。"""

    tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}
    knowledge_present = any(
        isinstance(message, HumanMessage)
        and "/context/knowledge.md" in _message_text(message)
        for message in messages
    )
    knowledge_checked = False
    knowledge_available = False
    context_sources: set[str] = set()
    discovery_call_count = 0
    needs_cross_validation = False

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                tool_call_id = str(tool_call.get("id") or "")
                if tool_call_id:
                    tool_calls[tool_call_id] = (
                        str(tool_call.get("name") or ""),
                        tool_call.get("args") or {},
                    )
            continue
        if not isinstance(message, ToolMessage):
            continue

        tool_call = tool_calls.get(str(message.tool_call_id or ""))
        if tool_call is None:
            continue
        current_tool_name, arguments = tool_call

        if current_tool_name == "read_file":
            file_path = str(
                arguments.get("file_path")
                or arguments.get("path")
                or ""
            ).replace("\\", "/")
            if file_path.lower().endswith("/knowledge.md"):
                knowledge_checked = True
                knowledge_available = getattr(message, "status", "success") != "error"
                continue

        if (
            current_tool_name == "analyze_plan"
            and getattr(message, "status", "success") == "error"
        ):
            needs_cross_validation = (
                str(arguments.get("knowledge_status") or "") != "authoritative"
            )
            continue

        if (
            current_tool_name in _SOURCE_DISCOVERY_TOOLS
            and getattr(message, "status", "success") != "error"
        ):
            discovery_call_count += 1
            context_sources.update(_context_sources(current_tool_name, arguments))

    return _DiscoveryState(
        knowledge_present=knowledge_present,
        knowledge_checked=knowledge_checked,
        knowledge_available=knowledge_available,
        context_sources=frozenset(context_sources),
        discovery_call_count=discovery_call_count,
        needs_cross_validation=needs_cross_validation,
    )


class AnswerMiddleware(AgentMiddleware[BenchmarkDeepAgentState, None, Any]):
    """注册最终答案工具，并在工作完成后强制结构化提交。"""

    state_schema = BenchmarkDeepAgentState
    tools = [answer_tool]

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        todos = request.state.get("todos") or []
        todos_completed = bool(todos) and all(
            todo.get("status") == "completed" for todo in todos
        )
        if request.state.get("analysis_plan") is not None and todos_completed:
            answer_tools = [
                item for item in request.tools if tool_name(item) == "answer"
            ]
            request = request.override(tools=answer_tools, tool_choice="answer")
        return handler(request)

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
    """强制执行 Discovery -> Plan -> Todos，并允许执行中修订计划。"""

    state_schema = BenchmarkDeepAgentState
    tools = [analyze_plan_tool]

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        discovery = _discovery_state(request.messages)
        if request.state.get("analysis_plan") is None:
            allowed_tools, tool_choice = discovery.tool_policy()
            request = request.override(
                tools=[
                    item
                    for item in request.tools
                    if tool_name(item) in allowed_tools
                ],
                tool_choice=tool_choice,
            )
        elif not request.state.get("todos"):
            todo_tools = [
                item for item in request.tools if tool_name(item) == "write_todos"
            ]
            request = request.override(tools=todo_tools, tool_choice="write_todos")
        return handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        current_tool_name = str(request.tool_call.get("name") or "")
        plan = request.state.get("analysis_plan")
        discovery = (
            _discovery_state(request.state["messages"])
            if plan is None or current_tool_name == "analyze_plan"
            else None
        )

        if plan is None and discovery is not None:
            if current_tool_name not in _SOURCE_DISCOVERY_TOOLS | {"analyze_plan"}:
                return _tool_error(
                    request,
                    "Only discovery tools are available before analyze_plan.",
                )
            requested_path = str(
                request.tool_call.get("args", {}).get("file_path")
                or request.tool_call.get("args", {}).get("path")
                or ""
            ).replace("\\", "/")
            if not discovery.knowledge_ready and (
                current_tool_name != "read_file"
                or not requested_path.lower().endswith("/knowledge.md")
            ):
                return _tool_error(
                    request,
                    "Read /context/knowledge.md before inspecting other sources.",
                )

        if current_tool_name == "analyze_plan":
            assert discovery is not None
            if not (discovery.knowledge_ready and discovery.context_ready):
                return _tool_error(
                    request,
                    (
                        "Complete discovery before analyze_plan: read knowledge.md when "
                        "present and inspect the minimum required independent data sources."
                    ),
                )
            arguments = request.tool_call.get("args", {})
            knowledge_status = str(arguments.get("knowledge_status") or "")
            requested_sources = {
                str(path).replace("\\", "/")
                for path in arguments.get("context_sources", [])
                if str(path).strip()
            }
            unobserved_sources = requested_sources - discovery.context_sources
            if unobserved_sources:
                return _tool_error(
                    request,
                    (
                        "context_sources must come from successful discovery calls. "
                        f"Unobserved sources: {sorted(unobserved_sources)}"
                    ),
                )
            if knowledge_status == "authoritative" and (
                not discovery.knowledge_present or not discovery.knowledge_available
            ):
                return _tool_error(
                    request,
                    (
                        "knowledge_status cannot be authoritative because knowledge.md "
                        "is unavailable or could not be read."
                    ),
                )
        if (
            plan is not None
            and not request.state.get("todos")
            and current_tool_name != "write_todos"
        ):
            return _tool_error(
                request,
                "Call write_todos successfully before using any other tool.",
            )
        return handler(request)


class HideUnavailableToolsMiddleware(AgentMiddleware[Any, None, Any]):
    """隐藏基准环境不允许模型调用的通用文件和 shell 工具。"""

    hidden_tools = frozenset({"execute", "ls", "write_file", "edit_file"})

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        filtered_tools = [
            item for item in request.tools if tool_name(item) not in self.hidden_tools
        ]
        return handler(request.override(tools=filtered_tools))


def workspace_permissions() -> list[FilesystemPermission]:
    """上下文只读；scratch 由后端默认规则管理。"""

    context_paths = ["/context/**"]
    return [
        FilesystemPermission(operations=["read"], paths=context_paths, mode="allow"),
        FilesystemPermission(operations=["write"], paths=context_paths, mode="deny"),
    ]
