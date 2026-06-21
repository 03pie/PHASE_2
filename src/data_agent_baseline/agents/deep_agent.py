from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from data_agent_baseline.agents.deep_state import DeepAgentConfig, TraceCallback
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.evidence_agent.controller import EvidenceAgentController


class DeepAgent:
    """Benchmark-compatible wrapper around the knowledge-guided evidence agent."""

    def __init__(
        self,
        *,
        model: BaseChatModel,
        config: DeepAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.config = config or DeepAgentConfig()
        self.system_prompt = system_prompt
        self.controller = EvidenceAgentController(model=model, config=self.config)

    def run(
        self,
        task: PublicTask,
        trace_callback: TraceCallback | None = None,
    ) -> AgentRunResult:
        return self.controller.run(task, trace_callback=trace_callback)
