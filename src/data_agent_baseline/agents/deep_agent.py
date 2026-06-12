from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from data_agent_baseline.agents.deep_state import (
    BenchmarkDeepAgentState,
    DeepAgentConfig,
    TraceCallback,
)
from data_agent_baseline.agents.middleware import (
    AnswerMiddleware,
    HideUnavailableToolsMiddleware,
    PlanningMiddleware,
    workspace_permissions,
)
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.agents.tracing import (
    LlmTraceMiddleware,
    TraceEventCollector,
    visible_text,
)
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.prompts.loader import (
    build_task_prompt,
    load_main_agent_prompt,
    load_subagent_prompt,
)
from data_agent_baseline.tools.execute_python import create_execute_python_tool


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
        backend = FilesystemBackend(
            root_dir=workspace,
            virtual_mode=True,
        )
        execute_python_tool = create_execute_python_tool(workspace, self.config)
        return create_deep_agent(
            model=self.model,
            tools=[execute_python_tool],
            system_prompt=self.system_prompt,
            backend=backend,
            permissions=workspace_permissions(),
            subagents=[
                {
                    "name": "general-purpose",
                    "description": (
                        "Handle a focused, complex data-analysis or verification subtask. "
                        "Returns findings with source files, calculation rules, assumptions, "
                        "and unresolved issues; it cannot submit the final answer."
                    ),
                    "system_prompt": load_subagent_prompt(),
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

        # 每个任务复制到独立临时目录，保证上下文只读且任务之间互不污染。
        with tempfile.TemporaryDirectory(prefix=f"dabench-{task.task_id}-") as temp_dir:
            workspace = Path(temp_dir)
            shutil.copytree(task.context_dir, workspace / "context")
            (workspace / "scratch").mkdir()
            graph = self._create_graph(workspace, collector)

            try:
                for state in graph.stream(
                    {"messages": [HumanMessage(content=build_task_prompt(task))]},
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
                    f"Agent did not submit an answer within "
                    f"{self.config.max_steps} model calls."
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
