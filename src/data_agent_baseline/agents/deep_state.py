from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, NotRequired

from deepagents.graph import DeepAgentState

from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.schema import AnswerTable

# 运行过程中的增量 trace 回调，runner 用它持续落盘执行状态。
TraceCallback = Callable[[AgentRunResult, str], None]


def merge_observed_sources_state(
    left: list[dict[str, Any]] | None,
    right: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge concurrently observed source metadata by normalized source path."""

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for sources in (left, right):
        if not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, dict):
                continue
            path = str(source.get("path") or "").replace("\\", "/")
            if not path:
                continue
            if path not in merged:
                order.append(path)
                merged[path] = {"path": path}
            merged[path].update(source)
            merged[path]["path"] = path
    return [merged[path] for path in order if path in merged]


@dataclass(frozen=True, slots=True)
class DeepAgentConfig:
    """DeepAgent 的运行限制配置。"""

    max_steps: int = 16
    execute_timeout_seconds: int = 30
    max_output_bytes: int = 100_000
    model_call_interval_seconds: float = 0.0
    question_structure_enabled: bool = False

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if self.execute_timeout_seconds < 1:
            raise ValueError("execute_timeout_seconds must be at least 1.")
        if self.max_output_bytes < 1:
            raise ValueError("max_output_bytes must be at least 1.")
        if self.model_call_interval_seconds < 0:
            raise ValueError("model_call_interval_seconds cannot be negative.")


class BenchmarkDeepAgentState(DeepAgentState):
    """在 DeepAgents 默认状态上扩展基准任务专用字段。"""

    original_request: NotRequired[str]
    question_structure: NotRequired[dict[str, Any]]
    question_structure_enforced: NotRequired[bool]
    knowledge_content: NotRequired[str]
    answer: NotRequired[AnswerTable | None]
    prepared_answer: NotRequired[AnswerTable | None]
    answer_candidate: NotRequired[dict[str, Any]]
    analysis_plan: NotRequired[dict[str, Any]]
    observed_sources: NotRequired[
        Annotated[list[dict[str, Any]], merge_observed_sources_state]
    ]
    todos: NotRequired[list[dict[str, str]]]
