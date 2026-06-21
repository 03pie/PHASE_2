from __future__ import annotations

from typing import Any

from data_agent_baseline.agents.runtime import StepRecord


class EvidenceTrace:
    """Small trace collector for the deterministic evidence state machine."""

    def __init__(self) -> None:
        self._steps: list[StepRecord] = []

    def add(
        self,
        *,
        action: str,
        action_input: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        thought: str = "",
        raw_response: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
        ok: bool = True,
    ) -> None:
        self._steps.append(
            StepRecord(
                step_index=len(self._steps) + 1,
                thought=thought,
                action=action,
                action_input=action_input or {},
                tool_call_id=tool_call_id,
                raw_response=raw_response or {},
                observation=observation or {},
                ok=ok,
            )
        )

    def snapshot(self) -> list[StepRecord]:
        return list(self._steps)
