from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NotRequired

from deepagents.graph import DeepAgentState

from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.schema import AnswerTable

# 运行过程中的增量 trace 回调，runner 用它持续落盘执行状态。
TraceCallback = Callable[[AgentRunResult, str], None]


@dataclass(frozen=True, slots=True)
class DeepAgentConfig:
    """DeepAgent 的运行限制配置。"""

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
    """在 DeepAgents 默认状态上扩展基准任务专用字段。"""

    answer: NotRequired[AnswerTable | None]
    analysis_plan: NotRequired[dict[str, Any]]
    todos: NotRequired[list[dict[str, str]]]
