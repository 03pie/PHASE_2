from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict

from data_agent_baseline.agents.deep_state import BenchmarkDeepAgentState
from data_agent_baseline.prompts.loader import load_tool_prompt

KnowledgeStatus = Literal[
    "authoritative",
    "unavailable",
    "invalid",
    "insufficient",
]
RequirementType = Literal[
    "entity",
    "measure",
    "filter",
    "time_range",
    "grouping",
    "ordering",
    "limit",
    "output",
    "output_column",
    "calculation",
    "deduplication",
    "reshape",
]
KnowledgeRuleType = Literal["semantic", "filter", "calculation", "output"]
OutputColumnRole = Literal[
    "measure",
    "calculation",
    "output_column",
    "record_key",
    "entity_key",
    "time_key",
]
TransformationOperation = Literal[
    "filter",
    "aggregate",
    "derive",
    "sort",
    "limit",
    "deduplicate",
    "reshape",
]


class IntentRequirement(TypedDict):
    statement: str
    requirement_type: RequirementType
    quote: str


class IntentSpec(TypedDict):
    requirements: list[IntentRequirement]
    unresolved: list[str]


class OutputColumn(TypedDict):
    name: str
    source_fields: list[str]
    role: NotRequired[OutputColumnRole]


class TransformationAuthorization(TypedDict):
    source: Literal["user", "knowledge"]
    quote: str


class TransformationSpec(TypedDict):
    operation: TransformationOperation
    description: str
    authorization: TransformationAuthorization


class SortKey(TypedDict):
    field: str
    direction: Literal["ascending", "descending"]


class OutputSpec(TypedDict):
    columns: list[OutputColumn]
    row_grain: str
    row_policy: Literal["preserve", "transform"]
    transformations: list[TransformationSpec]
    ordering: Literal["source", "specified", "unspecified"]
    sort_keys: list[SortKey]
    null_policy: Literal["preserve", "drop", "fill"]
    expected_row_count: int | None


class KnowledgeRule(TypedDict):
    rule_type: KnowledgeRuleType
    quote: str
    source_path: str


class ContextSource(TypedDict):
    path: str
    observations: list[str]


class EvidenceSpec(TypedDict):
    knowledge_status: KnowledgeStatus
    knowledge_rules: list[KnowledgeRule]
    knowledge_issue: str
    context_sources: list[ContextSource]
    cross_validated_inference: str


class RevisionSpec(TypedDict):
    version: int
    reason: str
    evidence_changes: list[str]
    changed_fields: list[str]


def _error(content: str, tool_call_id: str) -> ToolMessage:
    return ToolMessage(
        content=content,
        name="analyze_plan",
        tool_call_id=tool_call_id,
        status="error",
    )


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _clean_texts(items: list[str]) -> list[str]:
    return [text for item in items if (text := _clean_text(item))]


@tool("analyze_plan", description=load_tool_prompt("analyze_plan"))
def analyze_plan_tool(
    intent: IntentSpec,
    output_spec: OutputSpec,
    evidence: EvidenceSpec,
    revision: RevisionSpec,
    steps: list[str],
    original_request: Annotated[str, InjectedState("original_request")],
    tool_call_id: Annotated[str, InjectedToolCallId],
    schema_version: Literal["1.0"] = "1.0",
    delegation_candidates: list[str] | None = None,
) -> Command[BenchmarkDeepAgentState] | ToolMessage:
    """Create or revise a traceable plan after inspecting knowledge and task data.

    In preserve mode, columns may include minimal source-row context keys such as
    dates, periods, locations, or row IDs. Mark these as record_key, entity_key,
    or time_key when they explain the preserved source row rather than adding an
    analytical output target. The role is descriptive and must be supported by
    the isolated question structure; it does not make arbitrary helper columns
    valid.

    When knowledge is not authoritative, unavailable, or insufficient, pass an
    empty knowledge_rules list and document the issue plus the evidence-based
    inference in the corresponding evidence fields.
    """

    requirements = [
        {
            "statement": _clean_text(item["statement"]),
            "requirement_type": item["requirement_type"],
            "quote": _clean_text(item["quote"]),
        }
        for item in intent["requirements"]
    ]
    if not requirements or any(
        not item["statement"] or not item["quote"] for item in requirements
    ):
        return _error(
            "intent.requirements must contain statements with user quotes.",
            tool_call_id,
        )

    columns = [
        {
            "name": _clean_text(item["name"]),
            "source_fields": _clean_texts(item["source_fields"]),
            **(
                {"role": item["role"]}
                if "role" in item and _clean_text(item["role"])
                else {}
            ),
        }
        for item in output_spec["columns"]
    ]
    column_names = [item["name"] for item in columns]
    if not columns or any(not name for name in column_names):
        return _error(
            "output_spec.columns must contain non-empty output names.",
            tool_call_id,
        )
    if len(set(column_names)) != len(column_names):
        return _error("output_spec column names must be unique.", tool_call_id)

    expected_row_count = output_spec["expected_row_count"]
    if expected_row_count is not None and expected_row_count < 0:
        return _error(
            "output_spec.expected_row_count cannot be negative.",
            tool_call_id,
        )

    transformations = []
    for item in output_spec["transformations"]:
        if not isinstance(item, Mapping):
            return _error(
                "Each transformation must be an object.",
                tool_call_id,
            )
        authorization = item.get("authorization")
        if not isinstance(authorization, Mapping):
            return _error(
                "Each transformation authorization must be an object.",
                tool_call_id,
            )
        transformations.append(
            {
                "operation": item.get("operation"),
                "description": _clean_text(item.get("description")),
                "authorization": {
                    "source": authorization.get("source"),
                    "quote": _clean_text(authorization.get("quote")),
                },
            }
        )
    if any(
        not item["description"] or not item["authorization"]["quote"]
        for item in transformations
    ):
        return _error(
            "Each transformation requires a description and authorization quote.",
            tool_call_id,
        )

    context_sources = [
        {
            "path": _clean_text(item["path"]).replace("\\", "/"),
            "observations": _clean_texts(item["observations"]),
        }
        for item in evidence["context_sources"]
    ]
    if not context_sources or any(
        not item["path"] or not item["observations"] for item in context_sources
    ):
        return _error(
            "evidence.context_sources must include inspected paths and observations.",
            tool_call_id,
        )

    knowledge_rules = [
        {
            "rule_type": item["rule_type"],
            "quote": _clean_text(item["quote"]),
            "source_path": _clean_text(item["source_path"]).replace("\\", "/"),
        }
        for item in evidence["knowledge_rules"]
    ]
    knowledge_issue = _clean_text(evidence["knowledge_issue"])
    cross_validated_inference = _clean_text(
        evidence["cross_validated_inference"]
    )
    knowledge_status = evidence["knowledge_status"]
    if knowledge_status == "authoritative":
        if not knowledge_rules:
            return _error(
                "Authoritative knowledge requires at least one quoted rule.",
                tool_call_id,
            )
        knowledge_issue = ""
        cross_validated_inference = ""
    else:
        if not knowledge_issue:
            return _error(
                "Non-authoritative knowledge requires an explicit knowledge_issue.",
                tool_call_id,
            )
        if not cross_validated_inference:
            return _error(
                (
                    "Non-authoritative knowledge requires a "
                    "cross_validated_inference."
                ),
                tool_call_id,
            )

    normalized_steps = _clean_texts(steps)
    if not normalized_steps:
        return _error("steps must contain at least one plan step.", tool_call_id)
    if revision["version"] < 1 or not _clean_text(revision["reason"]):
        return _error(
            "revision.version must be positive and revision.reason is required.",
            tool_call_id,
        )

    plan = {
        "schema_version": schema_version,
        "original_request": original_request,
        "intent": {
            "requirements": requirements,
            "unresolved": _clean_texts(intent["unresolved"]),
        },
        "output_spec": {
            "columns": columns,
            "row_grain": _clean_text(output_spec["row_grain"]),
            "row_policy": output_spec["row_policy"],
            "transformations": transformations,
            "ordering": output_spec["ordering"],
            "sort_keys": [
                {
                    "field": _clean_text(item["field"]),
                    "direction": item["direction"],
                }
                for item in output_spec["sort_keys"]
            ],
            "null_policy": output_spec["null_policy"],
            "expected_row_count": expected_row_count,
        },
        "evidence": {
            "knowledge_status": knowledge_status,
            "knowledge_rules": knowledge_rules,
            "knowledge_issue": knowledge_issue,
            "context_sources": context_sources,
            "cross_validated_inference": cross_validated_inference,
        },
        "revision": {
            "version": revision["version"],
            "reason": _clean_text(revision["reason"]),
            "evidence_changes": _clean_texts(revision["evidence_changes"]),
            "changed_fields": _clean_texts(revision["changed_fields"]),
        },
        "steps": normalized_steps,
        "delegation_candidates": _clean_texts(delegation_candidates or []),
    }
    return Command(
        update={
            "analysis_plan": plan,
            "todos": [],
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
