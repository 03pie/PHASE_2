from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from deepagents import HarnessProfile, create_deep_agent, register_harness_profile
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from data_agent_baseline.agents.deep_state import (
    BenchmarkDeepAgentState,
    DeepAgentConfig,
    TraceCallback,
)
from data_agent_baseline.agents.filesystem import Utf8FilesystemBackend
from data_agent_baseline.agents.middleware import (
    AnswerMiddleware,
    CustomSystemPromptMiddleware,
    DISABLED_BUILTIN_TOOLS,
    DisabledToolGuardMiddleware,
    PlanningMiddleware,
    workspace_permissions,
)
from data_agent_baseline.agents.question_structure import (
    fallback_question_structure,
    format_question_structure,
    structure_question,
)
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.agents.tracing import (
    LlmTraceMiddleware,
    ModelCallRateLimiter,
    TraceEventCollector,
    serialize_message,
    visible_text,
)
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.prompts.loader import (
    build_task_prompt,
    load_main_agent_prompt,
    load_subagent_prompt,
    read_knowledge_content,
)
from data_agent_baseline.tools.agent_tools.analyze_plan import analyze_plan_tool
from data_agent_baseline.tools.agent_tools.execute_sql import create_execute_sql_tool
from data_agent_baseline.tools.agent_tools.execute_python import (
    create_execute_python_tool,
)
from data_agent_baseline.tools.agent_tools.finalize_answer_candidate import (
    finalize_answer_candidate_tool,
)
from data_agent_baseline.tools.agent_tools.extract_narrative_records import (
    create_extract_narrative_records_tool,
)
from data_agent_baseline.tools.agent_tools.grep_file import create_grep_file_tool
from data_agent_baseline.tools.agent_tools.inspect_sqlite import (
    create_inspect_sqlite_tool,
)
from data_agent_baseline.tools.agent_tools.query_schema import create_query_schema_tool
from data_agent_baseline.tools.agent_tools.read_csv import create_read_csv_tool
from data_agent_baseline.tools.agent_tools.read_doc import create_read_doc_tool
from data_agent_baseline.tools.agent_tools.read_json import create_read_json_tool
from data_agent_baseline.tools.agent_tools.set_answer import set_answer_tool
from data_agent_baseline.tools.agent_tools.task import (
    format_task_prompt_entry,
    task_tool_description_overrides,
)
from data_agent_baseline.tools.agent_tools.write_todos import (
    WRITE_TODOS_EXCLUDED_MIDDLEWARE,
    build_write_todos_middleware,
    create_write_todos_tools,
)

_TOOL_DESCRIPTION_LIMIT = 260
_WORKSPACE_CLEANUP_RETRY_DELAYS_SECONDS = (0.05, 0.1, 0.2, 0.4, 0.8)


def _normalize_answer(value: Any) -> AnswerTable | None:
    """把图状态中的字典答案统一转换为 AnswerTable。"""

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
    return [
        message
        for message in state.get("messages", [])
        if isinstance(message, BaseMessage)
    ]


def _partial_run_result(
    task_id: str,
    state: dict[str, Any],
    collector: TraceEventCollector,
) -> AgentRunResult:
    """构造运行中的可落盘快照，失败原因只在终态确定。"""

    return AgentRunResult(
        task_id=task_id,
        answer=_normalize_answer(state.get("answer")),
        steps=collector.snapshot(),
        failure_reason=None,
    )


def _model_provider(model: BaseChatModel) -> str | None:
    try:
        params = model._get_ls_params()
    except (AttributeError, TypeError, NotImplementedError):
        return None
    provider = params.get("ls_provider")
    return provider if isinstance(provider, str) and provider else None


def _register_benchmark_harness_profile(model: BaseChatModel) -> None:
    provider = _model_provider(model)
    if provider is None:
        return
    register_harness_profile(
        provider,
        HarnessProfile(
            base_system_prompt="",
            tool_description_overrides=task_tool_description_overrides(),
            excluded_tools=DISABLED_BUILTIN_TOOLS,
            excluded_middleware=WRITE_TODOS_EXCLUDED_MIDDLEWARE,
            extra_middleware=build_write_todos_middleware,
        ),
    )


def _remove_workspace_with_retry(path: Path) -> None:
    for delay_seconds in (*_WORKSPACE_CLEANUP_RETRY_DELAYS_SECONDS, 0.0):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
    shutil.rmtree(path, ignore_errors=True)


def _compact_text(value: Any, *, limit: int = _TOOL_DESCRIPTION_LIMIT) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _tool_name(tool: Any) -> str:
    if isinstance(tool, Mapping):
        return str(tool.get("name") or tool.get("function", {}).get("name") or "")
    return str(getattr(tool, "name", ""))


def _tool_description(tool: Any) -> str:
    if isinstance(tool, Mapping):
        function = tool.get("function") if isinstance(tool.get("function"), Mapping) else {}
        return str(tool.get("description") or function.get("description") or "")
    return str(getattr(tool, "description", "") or "")


def _tool_argument_names(tool: Any) -> list[str]:
    args = getattr(tool, "args", None)
    if isinstance(args, Mapping):
        return [str(name) for name in args if not str(name).startswith("_")]
    if isinstance(tool, Mapping):
        function = tool.get("function") if isinstance(tool.get("function"), Mapping) else {}
        parameters = function.get("parameters")
        if isinstance(parameters, Mapping):
            properties = parameters.get("properties")
            if isinstance(properties, Mapping):
                return [str(name) for name in properties]
    return []


def _format_tool_entry(tool: Any) -> str | None:
    name = _tool_name(tool)
    if not name:
        return None
    args = _tool_argument_names(tool)
    signature = f"`{name}`({', '.join(args)})" if args else f"`{name}`"
    description = _compact_text(_tool_description(tool)) or "Use according to its tool schema."
    return f"- {signature}: {description}"


def _format_tool_entries(tools: Sequence[Any]) -> list[str]:
    entries: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        name = _tool_name(tool)
        if not name or name in seen:
            continue
        seen.add(name)
        entry = _format_tool_entry(tool)
        if entry is not None:
            entries.append(entry)
    return entries


def _build_tool_descriptions(
    *,
    tools: Sequence[Any],
    include_plan_tools: bool,
    include_todo_tool: bool,
    subagents: Sequence[Mapping[str, Any]] = (),
) -> str:
    dynamic_tools: list[Any] = list(tools)
    if include_plan_tools:
        dynamic_tools.append(analyze_plan_tool)
    if include_todo_tool:
        dynamic_tools.extend(create_write_todos_tools())

    entries = _format_tool_entries(dynamic_tools)
    task_entry = format_task_prompt_entry(subagents)
    if task_entry is not None:
        entries.append(task_entry)

    return "\n".join(entries)


def _inject_tool_descriptions(prompt: str, tool_descriptions: str) -> str:
    placeholder = "{tool_descriptions}"
    if placeholder in prompt:
        return prompt.replace(placeholder, tool_descriptions.strip())
    return f"{prompt.strip()}\n\n## 可用工具\n\n{tool_descriptions.strip()}"


class DeepAgent:
    """负责组装 DeepAgents 图，并在隔离工作区内执行单个基准任务。"""

    def __init__(
        self,
        *,
        model: BaseChatModel,
        config: DeepAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.config = config or DeepAgentConfig()
        self.system_prompt = system_prompt or load_main_agent_prompt()

    def _create_graph(
        self,
        workspace: Path,
        collector: TraceEventCollector,
    ) -> Any:
        _register_benchmark_harness_profile(self.model)
        backend = Utf8FilesystemBackend(
            root_dir=workspace,
            virtual_mode=True,
        )
        execute_python_tool = create_execute_python_tool(workspace, self.config)
        data_tools = [
            create_read_csv_tool(workspace, self.config),
            create_read_json_tool(workspace, self.config),
            create_read_doc_tool(workspace, self.config),
            create_inspect_sqlite_tool(workspace, self.config),
            create_execute_sql_tool(workspace, self.config),
            create_grep_file_tool(workspace, self.config),
            create_query_schema_tool(workspace, self.config),
            create_extract_narrative_records_tool(workspace, self.config),
        ]
        rate_limiter = ModelCallRateLimiter(self.config.model_call_interval_seconds)
        execution_tools = [*data_tools, execute_python_tool]
        main_tools = [*execution_tools, set_answer_tool, finalize_answer_candidate_tool]
        subagent_description = (
            "Handle a focused, complex data-analysis or verification subtask. "
            "Returns findings with source files, calculation rules, assumptions, "
            "and unresolved issues; it cannot submit the final answer."
        )
        subagent_prompt = _inject_tool_descriptions(
            load_subagent_prompt(),
            _build_tool_descriptions(
                tools=execution_tools,
                include_plan_tools=False,
                include_todo_tool=True,
            ),
        )
        subagents: list[dict[str, Any]] = [
            {
                "name": "general-purpose",
                "description": subagent_description,
                "system_prompt": subagent_prompt,
                "tools": execution_tools,
                "middleware": [
                    CustomSystemPromptMiddleware(subagent_prompt),
                    DisabledToolGuardMiddleware(),
                    LlmTraceMiddleware(
                        collector,
                        scope="subagent:general-purpose",
                        rate_limiter=rate_limiter,
                    ),
                ],
            }
        ]
        main_prompt = _inject_tool_descriptions(
            self.system_prompt,
            _build_tool_descriptions(
                tools=main_tools,
                include_plan_tools=True,
                include_todo_tool=True,
                subagents=subagents,
            ),
        )
        return create_deep_agent(
            model=self.model,
            tools=main_tools,
            system_prompt=None,
            backend=backend,
            permissions=workspace_permissions(),
            subagents=subagents,
            middleware=[
                CustomSystemPromptMiddleware(main_prompt),
                LlmTraceMiddleware(
                    collector,
                    scope="main",
                    rate_limiter=rate_limiter,
                ),
                PlanningMiddleware(),
                AnswerMiddleware(),
                DisabledToolGuardMiddleware(),
                ModelCallLimitMiddleware(
                    run_limit=self.config.max_steps,
                    exit_behavior="end",
                ),
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
        question_structure, question_structure_enforced = self._structure_question(
            task,
            collector,
        )

        # 每个任务复制到独立临时目录，保证上下文只读且任务之间互不污染。
        workspace = Path(tempfile.mkdtemp(prefix=f"dabench-{task.task_id}-"))
        try:
            shutil.copytree(task.context_dir, workspace / "context")
            (workspace / "scratch").mkdir()
            graph = self._create_graph(workspace, collector)

            try:
                for state in graph.stream(
                    {
                        "original_request": task.question,
                        "question_structure": question_structure,
                        "question_structure_enforced": question_structure_enforced,
                        "knowledge_content": read_knowledge_content(task.context_dir),
                        "messages": [
                            HumanMessage(
                                content=build_task_prompt(
                                    task,
                                    question_structure=format_question_structure(
                                        question_structure
                                    ),
                                )
                            )
                        ],
                    },
                    config={"recursion_limit": max(100, self.config.max_steps * 8)},
                    stream_mode="values",
                ):
                    if not isinstance(state, dict):
                        continue
                    result = state
                    if trace_callback is None:
                        continue

                    # 状态可能重复推送，仅在快照实际变化时触发增量落盘。
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
            except Exception as exc:
                failed_result = AgentRunResult(
                    task_id=task.task_id,
                    answer=None,
                    steps=collector.snapshot(),
                    failure_reason=(
                        f"Deep Agent failed with {type(exc).__name__}: {exc}"
                    ),
                )
                if trace_callback is not None:
                    trace_callback(failed_result, "failed")
                return failed_result
        finally:
            _remove_workspace_with_retry(workspace)

        messages = _messages_from_state(result)
        answer = _normalize_answer(result.get("answer"))
        failure_reason: str | None = None
        if answer is None:
            # 区分模型调用额度耗尽和正常结束但未提交答案，便于 trace 排障。
            model_call_count = int(result.get("run_model_call_count", 0))
            limit_reached = any(
                isinstance(message, AIMessage)
                and visible_text(message).startswith("Model call limits exceeded:")
                for message in messages
            )
            if limit_reached or model_call_count >= self.config.max_steps:
                failure_reason = (
                    f"Agent did not prepare an answer within "
                    f"{self.config.max_steps} model calls."
                )
            else:
                failure_reason = "Agent completed without preparing an answer."

        run_result = AgentRunResult(
            task_id=task.task_id,
            answer=answer,
            steps=collector.snapshot(),
            failure_reason=failure_reason,
        )
        if trace_callback is not None:
            trace_callback(run_result, "completed" if run_result.succeeded else "failed")
        return run_result

    def _structure_question(
        self,
        task: PublicTask,
        collector: TraceEventCollector,
    ) -> tuple[dict[str, Any], bool]:
        if not self.config.question_structure_enabled:
            return (
                fallback_question_structure(
                    task.question,
                    "Question structuring node is disabled.",
                ),
                False,
            )
        try:
            structure, response = structure_question(self.model, task.question)
            collector.add(
                action="question_structure",
                scope="question_structure",
                action_input={"question": task.question},
                raw_response={"message": serialize_message(response)},
                observation={"status": "completed", "structure": structure},
                ok=True,
            )
            return structure, True
        except Exception as exc:
            structure = fallback_question_structure(
                task.question,
                f"Question structuring failed: {type(exc).__name__}: {exc}",
            )
            collector.add(
                action="question_structure",
                scope="question_structure",
                action_input={"question": task.question},
                observation={
                    "status": "error",
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "structure": structure,
                },
                ok=False,
            )
            return structure, False
