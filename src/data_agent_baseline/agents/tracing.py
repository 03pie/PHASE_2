from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from data_agent_baseline.agents.runtime import StepRecord


def json_safe(value: Any) -> Any:
    """将任意运行时对象转换成可写入 trace 的 JSON 数据。"""

    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    """完整保留 LangChain 消息的可序列化字段。"""

    return json_safe(message.model_dump(mode="json", exclude_none=True))


def serialize_tool_definition(value: BaseTool | dict[str, Any]) -> dict[str, Any]:
    """统一序列化对象形式和字典形式的工具定义。"""

    if isinstance(value, dict):
        return json_safe(value)
    args_schema: dict[str, Any] | None = None
    if value.tool_call_schema is not None:
        try:
            args_schema = json_safe(value.tool_call_schema.model_json_schema())
        except (AttributeError, TypeError, ValueError):
            args_schema = None
    return {
        "name": value.name,
        "description": value.description,
        "args_schema": args_schema,
    }


def serialize_tool_result(value: ToolMessage | Command[Any]) -> dict[str, Any]:
    """保留工具消息，或 Command 的图跳转与状态更新信息。"""

    if isinstance(value, ToolMessage):
        return serialize_message(value)
    update = value.update
    if isinstance(update, dict):
        serialized_update = dict(update)
        messages = serialized_update.get("messages")
        if isinstance(messages, list):
            serialized_update["messages"] = [
                serialize_message(message)
                if isinstance(message, BaseMessage)
                else json_safe(message)
                for message in messages
            ]
    else:
        serialized_update = json_safe(update)
    return {
        "type": "command",
        "graph": value.graph,
        "update": json_safe(serialized_update),
        "goto": json_safe(value.goto),
    }


@dataclass(slots=True)
class TraceEvent:
    """trace 内部事件；最终由 collector 转换成公共 StepRecord。"""

    action: str
    scope: str
    llm_call_index: int | None
    thought: str
    action_input: dict[str, Any]
    raw_response: dict[str, Any]
    observation: dict[str, Any]
    ok: bool


class TraceEventCollector:
    """收集并关联 LLM 调用与其产生的工具调用结果。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._events: list[TraceEvent] = []
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
        """每个代理作用域只记录一次系统提示词和首条用户消息。"""

        with self._lock:
            if scope in self._initialized_scopes:
                return
            self._initialized_scopes.add(scope)
        if system_message is not None:
            self.add(
                action="system_prompt",
                scope=scope,
                llm_call_index=llm_call_index,
                action_input={"message": serialize_message(system_message)},
            )
        for message in messages:
            if isinstance(message, HumanMessage):
                self.add(
                    action="user_prompt",
                    scope=scope,
                    llm_call_index=llm_call_index,
                    action_input={"message": serialize_message(message)},
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
        raw_response: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        ok: bool = True,
    ) -> None:
        with self._lock:
            event = TraceEvent(
                action=action,
                scope=scope,
                llm_call_index=llm_call_index,
                thought=thought,
                action_input=action_input or {},
                raw_response=raw_response or {},
                observation=observation or {},
                ok=ok,
            )
            self._events.append(event)

    def start_llm_call(
        self,
        *,
        scope: str,
        llm_call_index: int,
        request_payload: dict[str, Any],
    ) -> None:
        with self._lock:
            event = TraceEvent(
                action="llm_pending",
                scope=scope,
                llm_call_index=llm_call_index,
                thought="",
                action_input={"request": request_payload},
                raw_response={},
                observation={"status": "requesting", "tool_calls": []},
                ok=True,
            )
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
        """将模型声明的工具调用登记为 pending，等待工具中间件回填。"""

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
                            "args": json_safe(tool_call.get("args")),
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
        """返回不可变的公共 trace 快照，供运行结果和增量回调使用。"""

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
                tool_call_id=None,
                raw_response=event.raw_response,
                observation=event.observation,
                ok=event.ok,
            )
            for index, event in enumerate(events, start=1)
        ]


class LlmTraceMiddleware(AgentMiddleware[Any, None, Any]):
    """记录模型请求、模型响应，以及响应中工具调用的执行结果。"""

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
            "messages": [serialize_message(message) for message in request.messages],
            "tools": [serialize_tool_definition(item) for item in request.tools],
            "tool_choice": json_safe(request.tool_choice),
            "model_settings": json_safe(request.model_settings),
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

        serialized_result = [serialize_message(message) for message in response.result]
        response_text = "\n".join(
            visible_text(message)
            for message in response.result
            if isinstance(message, AIMessage) and visible_text(message)
        )
        self.collector.complete_llm_call(
            llm_call_index=call_index,
            messages=response.result,
            visible_text=response_text,
            response_payload={
                "messages": serialized_result,
                "structured_response": json_safe(response.structured_response),
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

        serialized_result = serialize_tool_result(result)
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


def visible_text(message: AIMessage) -> str:
    """兼容纯字符串和 OpenAI 内容块两种消息格式。"""

    if isinstance(message.content, str):
        return message.content
    text_parts: list[str] = []
    for block in message.content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in {"text", "output_text"} and isinstance(
            block.get("text"), str
        ):
            text_parts.append(block["text"])
    return "\n".join(text_parts)
