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

_SOURCE_DISCOVERY_TOOLS = frozenset({"execute_python", "grep", "read_file"})
_CONTEXT_PATH_PATTERN = re.compile(r"""["'](/context/[^"']+)["']""")
_USER_TRANSFORMATION_REQUIREMENT_TYPES = {
    "filter": frozenset({"filter"}),
    "aggregate": frozenset({"calculation"}),
    "derive": frozenset({"calculation"}),
    "sort": frozenset({"ordering"}),
    "limit": frozenset({"limit"}),
    "deduplicate": frozenset({"deduplication"}),
    "reshape": frozenset({"reshape"}),
}
_KNOWLEDGE_TRANSFORMATION_RULE_TYPES = {
    "filter": frozenset({"filter"}),
    "aggregate": frozenset({"calculation"}),
    "derive": frozenset({"calculation"}),
    "sort": frozenset({"output"}),
    "limit": frozenset({"output"}),
    "deduplicate": frozenset({"output"}),
    "reshape": frozenset({"output"}),
}
_REVISION_FIELDS = (
    "intent",
    "output_spec",
    "evidence",
    "steps",
    "delegation_candidates",
)


@dataclass(frozen=True, slots=True)
class _DiscoveryState:
    knowledge_present: bool
    knowledge_checked: bool
    knowledge_available: bool
    knowledge_content: str
    context_sources: frozenset[str]
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
        return set(_SOURCE_DISCOVERY_TOOLS) | {"analyze_plan"}, None


def tool_name(value: Any) -> str:
    """兼容 LangChain 工具对象和字典形式的工具定义。"""

    if isinstance(value, dict):
        return str(value.get("name") or value.get("function", {}).get("name") or "")
    return str(getattr(value, "name", ""))


def _invalid_tool_name(response: ModelResponse[Any]) -> str | None:
    """识别只有格式错误工具调用、没有可执行调用的模型响应。"""

    for message in response.result:
        if not isinstance(message, AIMessage):
            continue
        if message.invalid_tool_calls and not message.tool_calls:
            return str(message.invalid_tool_calls[0].get("name") or "")
    return None


def _retry_invalid_tool_call(
    request: ModelRequest[None],
    handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    response: ModelResponse[Any],
) -> ModelResponse[Any]:
    """让模型纠正一次工具参数格式，避免图把解析失败误判为正常结束。"""

    invalid_name = _invalid_tool_name(response)
    if invalid_name is None:
        return response

    available_tool_names = {tool_name(item) for item in request.tools}
    retry_request = request.override(
        messages=[
            *request.messages,
            HumanMessage(
                content=(
                    f"The previous `{invalid_name or 'tool'}` call could not be "
                    "parsed. Reissue exactly one tool call with valid JSON arguments "
                    "that fully match the provided schema. Keep the intended task "
                    "semantics unchanged and do not answer in plain text."
                )
            ),
        ],
        tool_choice=(
            invalid_name
            if invalid_name and invalid_name in available_tool_names
            else request.tool_choice
        ),
    )
    return handler(retry_request)


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
    if not content:
        return ""
    return "\n".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
    )


def _normalized_quote_text(value: str) -> str:
    """仅忽略 Markdown 和标点差异，用于定位唯一的原始知识行。"""

    return " ".join(
        re.sub(r"[^\w\u4e00-\u9fff]+", " ", value, flags=re.UNICODE)
        .casefold()
        .split()
    )


def _canonical_knowledge_quote(
    quote: str,
    knowledge_content: str | None,
) -> str | None:
    """把格式被简化的引用还原为 knowledge 中唯一存在的原始行。"""

    if not knowledge_content:
        return None
    if quote and quote in knowledge_content:
        return quote
    normalized_quote = _normalized_quote_text(quote)
    if not normalized_quote:
        return None
    matches = [
        line.strip()
        for line in knowledge_content.splitlines()
        if normalized_quote in _normalized_quote_text(line)
    ]
    return matches[0] if len(matches) == 1 else None


def _canonicalize_plan_quotes(
    request: ToolCallRequest,
    discovery: _DiscoveryState,
) -> ToolCallRequest:
    """将可唯一定位的 knowledge 引用替换成真实原文后再校验和落盘。"""

    arguments = dict(request.tool_call.get("args") or {})
    evidence = dict(arguments.get("evidence") or {})
    normalized_rules: list[Any] = []
    changed = False
    for rule in evidence.get("knowledge_rules") or []:
        if not isinstance(rule, dict):
            normalized_rules.append(rule)
            continue
        normalized_rule = dict(rule)
        quote = str(rule.get("quote") or "").strip()
        canonical_quote = _canonical_knowledge_quote(
            quote,
            discovery.knowledge_content,
        )
        if canonical_quote is not None and canonical_quote != quote:
            normalized_rule["quote"] = canonical_quote
            changed = True
        normalized_rules.append(normalized_rule)
    if not changed:
        return request

    evidence["knowledge_rules"] = normalized_rules
    arguments["evidence"] = evidence
    return request.override(
        tool_call={
            **request.tool_call,
            "args": arguments,
        }
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
    knowledge_contents: list[str] = []
    context_sources: set[str] = set()
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
                if knowledge_available:
                    knowledge_contents.append(_message_text(message))
                continue

        if (
            current_tool_name in {"execute_python", "grep"}
            and getattr(message, "status", "success") != "error"
        ):
            inspected_text = str(
                arguments.get("code")
                or arguments.get("file_path")
                or arguments.get("path")
                or ""
            ).replace("\\", "/")
            if "/context/knowledge.md" in inspected_text:
                knowledge_checked = True
                knowledge_available = True
                observed_content = _message_text(message)
                if observed_content:
                    knowledge_contents.append(observed_content)

        if (
            current_tool_name == "analyze_plan"
            and getattr(message, "status", "success") == "error"
        ):
            evidence = arguments.get("evidence") or {}
            needs_cross_validation = (
                str(evidence.get("knowledge_status") or "") != "authoritative"
            )
            continue

        if (
            current_tool_name in _SOURCE_DISCOVERY_TOOLS
            and getattr(message, "status", "success") != "error"
        ):
            context_sources.update(_context_sources(current_tool_name, arguments))

    return _DiscoveryState(
        knowledge_present=knowledge_present,
        knowledge_checked=knowledge_checked,
        knowledge_available=knowledge_available,
        knowledge_content="\n".join(knowledge_contents),
        context_sources=frozenset(context_sources),
        needs_cross_validation=needs_cross_validation,
    )


def _plan_error(
    request: ToolCallRequest,
    content: str,
) -> ToolMessage:
    return _tool_error(request, f"Invalid analysis plan: {content}")


def _validate_plan_contract(
    request: ToolCallRequest,
    discovery: _DiscoveryState,
) -> ToolMessage | None:
    """用可核验的引用和状态约束计划，不推断自然语言语义。"""

    arguments = request.tool_call.get("args") or {}
    original_request = str(request.state.get("original_request") or "")
    if not original_request:
        return _plan_error(request, "original_request is missing from agent state.")

    intent = arguments.get("intent") or {}
    requirements = intent.get("requirements") or []
    requirement_types_by_quote = {
        str(item.get("quote") or "").strip(): str(
            item.get("requirement_type") or ""
        )
        for item in requirements
        if isinstance(item, dict)
    }
    if not requirement_types_by_quote or any(
        not quote or quote not in original_request
        for quote in requirement_types_by_quote
    ):
        return _plan_error(
            request,
            "every intent requirement quote must occur verbatim in original_request.",
        )

    evidence = arguments.get("evidence") or {}
    knowledge_status = str(evidence.get("knowledge_status") or "")
    context_sources = evidence.get("context_sources") or []
    requested_sources = {
        str(item.get("path") or "").replace("\\", "/")
        for item in context_sources
        if isinstance(item, dict) and str(item.get("path") or "").strip()
    }
    unobserved_sources = requested_sources - discovery.context_sources
    if unobserved_sources:
        return _plan_error(
            request,
            (
                "context sources must come from successful discovery calls; "
                f"unobserved sources: {sorted(unobserved_sources)}."
            ),
        )
    if knowledge_status == "authoritative" and (
        not discovery.knowledge_present or not discovery.knowledge_available
    ):
        return _plan_error(
            request,
            "knowledge cannot be authoritative because knowledge.md was unavailable.",
        )

    knowledge_rules = evidence.get("knowledge_rules") or []
    rule_quotes_by_type: dict[str, set[str]] = {}
    for rule in knowledge_rules:
        if not isinstance(rule, dict):
            continue
        source_path = str(rule.get("source_path") or "").replace("\\", "/")
        quote = str(rule.get("quote") or "").strip()
        rule_type = str(rule.get("rule_type") or "")
        if (
            source_path.lower() != "/context/knowledge.md"
            or not quote
            or quote not in discovery.knowledge_content
        ):
            return _plan_error(
                request,
                (
                    "every knowledge rule must quote text observed in "
                    "/context/knowledge.md."
                ),
            )
        rule_quotes_by_type.setdefault(rule_type, set()).add(quote)

    output_spec = arguments.get("output_spec") or {}
    transformations = output_spec.get("transformations") or []
    row_policy = str(output_spec.get("row_policy") or "")
    if transformations and row_policy != "transform":
        return _plan_error(
            request,
            "row_policy must be transform when transformations are present.",
        )
    if not transformations:
        if row_policy != "preserve":
            return _plan_error(
                request,
                "row_policy must be preserve when no transformation is authorized.",
            )
        if output_spec.get("ordering") != "source":
            return _plan_error(
                request,
                "preserve plans must keep source ordering.",
            )
        if output_spec.get("null_policy") != "preserve":
            return _plan_error(
                request,
                "preserve plans must keep source null values.",
            )
        if output_spec.get("sort_keys"):
            return _plan_error(
                request,
                "preserve plans cannot define sort keys.",
            )

    for transformation in transformations:
        if not isinstance(transformation, dict):
            return _plan_error(request, "transformations must be structured objects.")
        operation = str(transformation.get("operation") or "")
        authorization = transformation.get("authorization") or {}
        source = str(authorization.get("source") or "")
        quote = str(authorization.get("quote") or "").strip()
        if source == "user":
            requirement_type = requirement_types_by_quote.get(quote)
            allowed_types = _USER_TRANSFORMATION_REQUIREMENT_TYPES.get(
                operation, frozenset()
            )
            if requirement_type not in allowed_types:
                return _plan_error(
                    request,
                    (
                        f"user authorization for {operation!r} must cite an explicit "
                        f"requirement typed as one of {sorted(allowed_types)}."
                    ),
                )
        if source == "knowledge":
            allowed_rule_types = _KNOWLEDGE_TRANSFORMATION_RULE_TYPES.get(
                operation, frozenset()
            )
            authorized_quotes = set().union(
                *(
                    rule_quotes_by_type.get(rule_type, set())
                    for rule_type in allowed_rule_types
                )
            )
            if quote not in authorized_quotes:
                return _plan_error(
                    request,
                    (
                        f"knowledge authorization for {operation!r} must cite an "
                        f"observed rule typed as one of {sorted(allowed_rule_types)}."
                    ),
                )
        if source not in {"user", "knowledge"}:
            return _plan_error(
                request,
                "context evidence cannot authorize a transformation.",
            )

    revision = arguments.get("revision") or {}
    version = revision.get("version")
    reported_changes = {
        str(field).strip()
        for field in revision.get("changed_fields") or []
        if str(field).strip()
    }
    previous_plan = request.state.get("analysis_plan")
    if previous_plan is None:
        if version != 1:
            return _plan_error(request, "the initial plan must use revision version 1.")
        if reported_changes:
            return _plan_error(
                request,
                "the initial plan cannot report changed_fields.",
            )
        return None

    previous_version = (previous_plan.get("revision") or {}).get("version")
    if not isinstance(previous_version, int) or version != previous_version + 1:
        return _plan_error(
            request,
            "a revised plan must increment revision.version by exactly one.",
        )
    previous_requirements = (
        previous_plan.get("intent", {}).get("requirements") or []
    )
    if any(item not in requirements for item in previous_requirements):
        return _plan_error(
            request,
            "a revision cannot remove or rewrite existing user requirements.",
        )

    actual_changes = {
        field
        for field in _REVISION_FIELDS
        if previous_plan.get(field) != arguments.get(field)
    }
    if not actual_changes:
        return _plan_error(request, "a revision must make an actual plan change.")
    missing_changes = actual_changes - reported_changes
    if missing_changes:
        return _plan_error(
            request,
            f"revision.changed_fields omits: {sorted(missing_changes)}.",
        )
    if "evidence" in actual_changes and not revision.get("evidence_changes"):
        return _plan_error(
            request,
            "evidence changes must be described in revision.evidence_changes.",
        )
    return None


class AnswerMiddleware(AgentMiddleware[BenchmarkDeepAgentState, None, Any]):
    """计算结果进入状态后直接完成，不再要求模型复述完整表格。"""

    state_schema = BenchmarkDeepAgentState

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: BenchmarkDeepAgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        del runtime
        if state.get("answer") is not None:
            return {"jump_to": "end"}
        prepared_answer = state.get("prepared_answer")
        if prepared_answer is not None:
            return {
                "answer": prepared_answer,
                "jump_to": "end",
            }
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
        response = handler(request)
        return _retry_invalid_tool_call(request, handler, response)

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
            request = _canonicalize_plan_quotes(request, discovery)
            contract_error = _validate_plan_contract(request, discovery)
            if contract_error is not None:
                return contract_error
        if current_tool_name == "write_todos" and plan is not None:
            todos = request.tool_call.get("args", {}).get("todos") or []
            todo_contents = [
                str(todo.get("content") or "").strip()
                for todo in todos
                if isinstance(todo, dict)
            ]
            if todo_contents != plan.get("steps"):
                return _tool_error(
                    request,
                    "write_todos contents must exactly match analysis_plan.steps.",
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
