from __future__ import annotations

import json
from typing import Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from data_agent_baseline.agents.deep_state import BenchmarkDeepAgentState


def _error(content: str, tool_call_id: str) -> ToolMessage:
    return ToolMessage(
        content=content,
        name="analyze_plan",
        tool_call_id=tool_call_id,
        status="error",
    )


def _clean(items: list[str]) -> list[str]:
    return [item.strip() for item in items if item.strip()]


@tool("analyze_plan")
def analyze_plan_tool(
    intent: str,
    output_spec: str,
    knowledge_status: Literal[
        "authoritative",
        "unavailable",
        "invalid",
        "insufficient",
    ],
    knowledge_findings: list[str],
    knowledge_issue: str,
    context_sources: list[str],
    context_evidence: list[str],
    cross_validated_inference: str,
    uncertainties: list[str],
    steps: list[str],
    delegation_candidates: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[BenchmarkDeepAgentState] | ToolMessage:
    """Create or revise a plan after inspecting knowledge and relevant task data."""

    if not intent.strip() or not output_spec.strip():
        return _error("intent and output_spec must be non-empty.", tool_call_id)

    normalized_knowledge = _clean(knowledge_findings)
    normalized_sources = _clean(context_sources)
    normalized_context = _clean(context_evidence)
    if not normalized_sources:
        return _error(
            "context_sources must identify the inspected source paths.",
            tool_call_id,
        )
    if not normalized_context:
        return _error(
            "context_evidence must cite observed schema or data findings.",
            tool_call_id,
        )

    normalized_issue = knowledge_issue.strip()
    normalized_inference = cross_validated_inference.strip()
    if knowledge_status == "authoritative":
        if not normalized_knowledge:
            return _error(
                (
                    "knowledge_findings must cite the binding rules when "
                    "knowledge_status is authoritative."
                ),
                tool_call_id,
            )
        if normalized_issue or normalized_inference:
            return _error(
                (
                    "Authoritative knowledge cannot be replaced by an inferred rule; "
                    "knowledge_issue and cross_validated_inference must be empty."
                ),
                tool_call_id,
            )
    else:
        if not normalized_issue:
            return _error(
                (
                    "knowledge_issue must explain why knowledge is unavailable, "
                    "invalid, or insufficient."
                ),
                tool_call_id,
            )
        if len(set(normalized_sources)) < 2:
            return _error(
                (
                    "Non-authoritative knowledge requires at least two distinct "
                    "context_sources for cross-validation."
                ),
                tool_call_id,
            )
        if not normalized_inference:
            return _error(
                (
                    "cross_validated_inference is required when knowledge is "
                    "non-authoritative."
                ),
                tool_call_id,
            )

    normalized_steps = _clean(steps)
    if not normalized_steps:
        return _error(
            "steps must contain at least one non-empty plan step.",
            tool_call_id,
        )

    plan = {
        "intent": intent.strip(),
        "output_spec": output_spec.strip(),
        "knowledge_status": knowledge_status,
        "knowledge_findings": normalized_knowledge,
        "knowledge_issue": normalized_issue,
        "context_sources": normalized_sources,
        "context_evidence": normalized_context,
        "cross_validated_inference": normalized_inference,
        "uncertainties": _clean(uncertainties),
        "steps": normalized_steps,
        "delegation_candidates": _clean(delegation_candidates),
    }
    return Command(
        update={
            "analysis_plan": plan,
            "messages": [
                ToolMessage(
                    content=json.dumps(plan, ensure_ascii=False),
                    name="analyze_plan",
                    tool_call_id=tool_call_id,
                    status="success",
                )
            ],
        }
    )
