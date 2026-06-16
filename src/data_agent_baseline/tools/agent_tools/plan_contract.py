from __future__ import annotations

import json
from typing import Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import TypedDict

from data_agent_baseline.agents.deep_state import BenchmarkDeepAgentState
from data_agent_baseline.prompts.loader import load_tool_prompt

ContractIntentMode = Literal["lookup_records", "transform_records"]
ContractOperation = Literal[
    "project",
    "select_existing_records",
    "filter",
    "aggregate",
    "derive",
    "sort",
    "limit",
    "deduplicate",
    "reshape",
]
ContractNullPolicy = Literal["preserve", "drop", "fill"]
ScopeStrength = Literal["none", "weak", "strong"]


class ScopePolicy(TypedDict):
    user_scope_hint: str
    scope_strength: ScopeStrength
    existing_scope_field: str
    derived_scope_allowed: bool


class PlanContract(TypedDict):
    contract_id: str
    intent_mode: ContractIntentMode
    priority_policy: Literal["knowledge > observed_data > user_colloquial_hint"]
    requested_measures: list[str]
    source_grain: str
    required_identity_fields: list[str]
    allowed_operations: list[ContractOperation]
    forbidden_operations: list[ContractOperation]
    null_policy: ContractNullPolicy
    scope_policy: ScopePolicy
    source_row_count: int | None
    source_paths: list[str]
    rationale: list[str]


def _error(content: str, tool_call_id: str) -> ToolMessage:
    return ToolMessage(
        content=content,
        name="establish_plan_contract",
        tool_call_id=tool_call_id,
        status="error",
    )


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _clean_texts(values: list[str]) -> list[str]:
    return [text for value in values if (text := _clean_text(value))]


def _dedupe_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in _clean_texts(values):
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


@tool(
    "establish_plan_contract",
    description=load_tool_prompt("establish_plan_contract"),
)
def establish_plan_contract_tool(
    contract: PlanContract,
    original_request: Annotated[str, InjectedState("original_request")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[BenchmarkDeepAgentState] | ToolMessage:
    """Run the establish_plan_contract tool."""

    contract_id = _clean_text(contract["contract_id"])
    if not contract_id:
        return _error("contract.contract_id is required.", tool_call_id)

    requested_measures = _dedupe_texts(contract["requested_measures"])
    if not requested_measures:
        return _error("contract.requested_measures must not be empty.", tool_call_id)

    source_paths = [
        path.replace("\\", "/")
        for path in _dedupe_texts(contract["source_paths"])
    ]
    if not source_paths:
        return _error("contract.source_paths must name inspected context sources.", tool_call_id)

    source_row_count = contract["source_row_count"]
    if source_row_count is not None and source_row_count < 0:
        return _error("contract.source_row_count cannot be negative.", tool_call_id)

    scope_policy = contract["scope_policy"]
    normalized_contract = {
        "contract_id": contract_id,
        "original_request": original_request,
        "intent_mode": contract["intent_mode"],
        "priority_policy": contract["priority_policy"],
        "requested_measures": requested_measures,
        "source_grain": _clean_text(contract["source_grain"]),
        "required_identity_fields": _dedupe_texts(contract["required_identity_fields"]),
        "allowed_operations": list(dict.fromkeys(contract["allowed_operations"])),
        "forbidden_operations": list(dict.fromkeys(contract["forbidden_operations"])),
        "null_policy": contract["null_policy"],
        "scope_policy": {
            "user_scope_hint": _clean_text(scope_policy["user_scope_hint"]),
            "scope_strength": scope_policy["scope_strength"],
            "existing_scope_field": _clean_text(scope_policy["existing_scope_field"]),
            "derived_scope_allowed": bool(scope_policy["derived_scope_allowed"]),
        },
        "source_row_count": source_row_count,
        "source_paths": source_paths,
        "rationale": _clean_texts(contract["rationale"]),
    }
    if not normalized_contract["source_grain"]:
        return _error("contract.source_grain is required.", tool_call_id)
    if not normalized_contract["rationale"]:
        return _error("contract.rationale must explain the contract.", tool_call_id)

    return Command(
        update={
            "plan_contract": normalized_contract,
            "messages": [
                ToolMessage(
                    content=json.dumps(normalized_contract, ensure_ascii=False),
                    name="establish_plan_contract",
                    tool_call_id=tool_call_id,
                    status="success",
                )
            ],
        }
    )
