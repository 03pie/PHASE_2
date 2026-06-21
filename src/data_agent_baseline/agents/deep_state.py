from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from data_agent_baseline.agents.runtime import AgentRunResult

TraceCallback = Callable[[AgentRunResult, str], None]


@dataclass(frozen=True, slots=True)
class DeepAgentConfig:
    """Runtime limits shared by the benchmark runner and evidence agent."""

    max_steps: int = 16
    execute_timeout_seconds: int = 30
    max_output_bytes: int = 100_000
    model_call_interval_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if self.execute_timeout_seconds < 1:
            raise ValueError("execute_timeout_seconds must be at least 1.")
        if self.max_output_bytes < 1:
            raise ValueError("max_output_bytes must be at least 1.")
        if self.model_call_interval_seconds < 0:
            raise ValueError("model_call_interval_seconds cannot be negative.")
