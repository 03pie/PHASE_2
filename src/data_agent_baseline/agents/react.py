from __future__ import annotations

import ast
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, NotRequired

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.graph import DeepAgentState
from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from data_agent_baseline.agents.knowledge_schema import format_knowledge_schema_prompt
from data_agent_baseline.agents.prompts import DEEP_AGENT_SYSTEM_PROMPT, SUBAGENT_SYSTEM_PROMPT
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask

TraceCallback = Callable[[AgentRunResult, str], None]


@dataclass(frozen=True, slots=True)
class DeepAgentConfig:
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
    answer: NotRequired[AnswerTable | None]
    analysis_plan: NotRequired[dict[str, Any]]
    original_question: NotRequired[str]
    planning_blocked: NotRequired[str]
    plan_revisions: NotRequired[list[dict[str, Any]]]
    todos: NotRequired[list[dict[str, str]]]


RAW_OPERATION_TYPES = {"column_extract", "filter_rows", "lookup"}
AGGREGATE_OPERATION_TYPES = {"aggregate", "rank"}
SUPPORTED_OPERATION_TYPES = RAW_OPERATION_TYPES | AGGREGATE_OPERATION_TYPES | {
    "compute",
    "unknown",
}
StringListInput = list[str] | str | None
EvidenceMapInput = dict[str, str] | str | None
QuestionAuditInput = list[dict[str, Any]] | str | None
AnswerRowsInput = list[Any] | str | None
MIN_FINAL_INTENT_CONFIDENCE = 0.8
MAX_REPEATED_PLANNING_ERRORS = 4
IMMEDIATE_BLOCK_PLANNING_ERRORS = {"schema_source_rebinding"}


def _normalize_string_list(values: StringListInput) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        stripped = values.strip()
        if not stripped:
            return []
        if re.fullmatch(r"none(?:\s*[-鈥?].*)?|null|n/?a", stripped, flags=re.IGNORECASE):
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(decoded, list):
            values = decoded
        else:
            return [stripped]
    return [value.strip() for value in values if isinstance(value, str) and value.strip()]


def _normalize_identifier_list(values: StringListInput) -> list[str]:
    normalized = _normalize_string_list(values)
    if len(normalized) != 1 or not isinstance(values, str):
        return normalized

    candidate = normalized[0]
    if "," not in candidate and "\uff0c" not in candidate:
        return normalized
    parts = [part.strip() for part in re.split(r"[,\uff0c]", candidate) if part.strip()]
    if len(parts) < 2 or not all(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", part) for part in parts
    ):
        return normalized
    return parts


def _normalize_evidence_map(values: EvidenceMapInput) -> dict[str, str]:
    if values is None:
        return {}
    if isinstance(values, str):
        stripped = values.strip()
        if not stripped:
            return {}
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return {}
        if not isinstance(decoded, dict):
            return {}
        values = decoded
    return {
        str(field).strip(): evidence.strip()
        for field, evidence in values.items()
        if str(field).strip() and isinstance(evidence, str) and evidence.strip()
    }


def _normalize_question_audit(values: QuestionAuditInput) -> list[dict[str, str]]:
    if values is None:
        return []
    if isinstance(values, str):
        stripped = values.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [
                {
                    "text": stripped,
                    "role": "ambiguous",
                    "plan_impact": "model-supplied audit text was not JSON.",
                }
            ]
        if not isinstance(decoded, list):
            return []
        values = decoded
    normalized: list[dict[str, str]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("span") or "").strip()
        if not text:
            continue
        role = str(item.get("role") or "ambiguous").strip() or "ambiguous"
        plan_impact = str(
            item.get("plan_impact")
            or item.get("impact")
            or item.get("reason")
            or ""
        ).strip()
        normalized.append(
            {
                "text": text,
                "role": role,
                "plan_impact": plan_impact,
            }
        )
    return normalized


def _contract_evidence_items(contract: dict[str, Any]) -> list[tuple[str, str]]:
    role_names = {
        "requested_outputs": "requested_output",
        "scope_evidence": "scope",
        "request_mode_evidence": "request_action",
        "filter_evidence": "filter",
        "grouping_evidence": "time_or_sequence",
        "transformation_evidence": "transformation",
    }
    items: list[tuple[str, str]] = []
    for key, role in role_names.items():
        for evidence in _normalize_string_list(contract.get(key)):
            items.append((evidence, role))
    for evidence in _normalize_evidence_map(
        contract.get("derived_field_bindings")
    ).values():
        items.append((evidence, "derived_output"))
    return items


def _auto_question_audit(
    *,
    question: str,
    contract: dict[str, Any],
    supplied_audit: QuestionAuditInput,
) -> dict[str, Any]:
    supplied_segments = _normalize_question_audit(supplied_audit)
    accepted_supplied: list[dict[str, str]] = []
    for segment in supplied_segments:
        span = _question_span_for_evidence(question, segment["text"])
        if span is None:
            continue
        accepted_supplied.append({**segment, "text": span})
    if accepted_supplied:
        covered = "".join(segment["text"] for segment in accepted_supplied)
        return {
            "original_question": question,
            "segments": accepted_supplied,
            "coverage": "model_supplied",
            "coverage_complete": all(
                char.isspace() or char in covered for char in question
            ),
        }

    spans: list[tuple[int, int, str, str]] = []
    occupied = [False] * len(question)
    for evidence, role in sorted(
        _contract_evidence_items(contract),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if not evidence:
            continue
        start = question.find(evidence)
        while start >= 0:
            end = start + len(evidence)
            if not any(occupied[start:end]):
                spans.append((start, end, evidence, role))
                for index in range(start, end):
                    occupied[index] = True
                break
            start = question.find(evidence, start + 1)

    spans.sort(key=lambda item: item[0])
    segments: list[dict[str, str]] = []
    cursor = 0
    for start, end, text, role in spans:
        if cursor < start:
            unused = question[cursor:start]
            if unused:
                segments.append(
                    {
                        "text": unused,
                        "role": "unused_confirmed",
                        "plan_impact": (
                            "No contract evidence was derived from this wording."
                        ),
                    }
                )
        segments.append(
            {
                "text": text,
                "role": role,
                "plan_impact": "Authorizes the corresponding request contract role.",
            }
        )
        cursor = end
    if cursor < len(question):
        unused = question[cursor:]
        if unused:
            segments.append(
                {
                    "text": unused,
                    "role": "unused_confirmed",
                    "plan_impact": "No contract evidence was derived from this wording.",
                }
            )
    if not segments and question:
        segments.append(
            {
                "text": question,
                "role": "ambiguous",
                "plan_impact": "No explicit request contract evidence was supplied.",
            }
        )
    return {
        "original_question": question,
        "segments": segments,
        "coverage": "auto_from_contract",
        "coverage_complete": True,
    }


def _normalize_plan_operation(operation_type: str) -> str:
    operation = operation_type.strip().lower().replace("-", "_").replace(" ", "_")
    if not operation:
        return "unknown"
    aliases = {
        "extract": "column_extract",
        "query": "column_extract",
        "raw_extract": "column_extract",
        "select_column": "column_extract",
        "filter": "filter_rows",
        "groupby": "aggregate",
        "group_by": "aggregate",
        "aggregation": "aggregate",
    }
    return aliases.get(operation, operation)


def _infer_plan_operation(
    operation_type: str,
    *,
    group_by: StringListInput,
    aggregation: str | None,
    transformation_evidence: StringListInput,
) -> str:
    operation = _normalize_plan_operation(operation_type)
    if operation != "unknown":
        return operation
    if _normalize_identifier_list(group_by) or (
        isinstance(aggregation, str) and aggregation.strip()
    ):
        return "aggregate"
    if _normalize_string_list(transformation_evidence):
        return "compute"
    return "column_extract"


def _planned_output_columns(args: dict[str, Any]) -> list[str]:
    bound_columns = list(_normalize_evidence_map(args.get("field_bindings")))
    derived_columns = list(_normalize_evidence_map(args.get("derived_field_bindings")))
    if bound_columns or derived_columns:
        return [*bound_columns, *derived_columns]
    if bound_columns:
        return bound_columns
    output_columns = _normalize_identifier_list(args.get("output_columns"))
    if output_columns:
        return output_columns
    target_fields = _normalize_identifier_list(args.get("target_fields"))
    if target_fields:
        return target_fields
    return list(_normalize_evidence_map(args.get("field_bindings")))


def _normalized_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", value.strip().lower())


def _evidence_key(value: str) -> str:
    return "".join(value.split()).casefold()


def _question_span_for_evidence(question: str, evidence: str) -> str | None:
    evidence_units = [
        char.casefold() for char in evidence if not char.isspace()
    ]
    if not evidence_units:
        return None

    question_units = [
        (char.casefold(), index)
        for index, char in enumerate(question)
        if not char.isspace()
    ]
    limit = len(question_units) - len(evidence_units) + 1
    for start in range(max(limit, 0)):
        for offset, evidence_char in enumerate(evidence_units):
            if question_units[start + offset][0] != evidence_char:
                break
        else:
            begin = question_units[start][1]
            end = question_units[start + len(evidence_units) - 1][1] + 1
            return question[begin:end]
    return None


def _evidence_is_in_question(question: str, evidence: str) -> bool:
    return _question_span_for_evidence(question, evidence) is not None


def _canonical_question_evidence(question: str, evidence: str) -> str:
    return _question_span_for_evidence(question, evidence) or evidence


def _canonical_question_evidence_list(
    question: str,
    values: StringListInput,
) -> list[str]:
    return [
        _canonical_question_evidence(question, evidence)
        for evidence in _normalize_string_list(values)
    ]


def _canonical_question_evidence_map(
    question: str,
    values: EvidenceMapInput,
) -> dict[str, str]:
    return {
        field: _canonical_question_evidence(question, evidence)
        for field, evidence in _normalize_evidence_map(values).items()
    }


def _canonical_request_contract(
    question: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    canonical = dict(contract)
    for key in (
        "requested_outputs",
        "scope_evidence",
        "request_mode_evidence",
        "filter_evidence",
        "grouping_evidence",
        "transformation_evidence",
    ):
        canonical[key] = _canonical_question_evidence_list(question, contract.get(key))
    canonical["field_bindings"] = _canonical_question_evidence_map(
        question,
        contract.get("field_bindings"),
    )
    canonical["derived_field_bindings"] = _canonical_question_evidence_map(
        question,
        contract.get("derived_field_bindings"),
    )
    return canonical


def _identifier_candidates(*values: StringListInput) -> set[str]:
    candidates: set[str] = set()
    for value in values:
        for item in _normalize_identifier_list(value):
            normalized = _normalized_field_name(item)
            if normalized:
                candidates.add(normalized)
    return candidates


def _normalize_plan_args_for_contract(
    args: dict[str, Any],
    *,
    question: str,
) -> dict[str, Any]:
    normalized_args = dict(args)
    bindings = _normalize_evidence_map(args.get("field_bindings"))
    if not bindings:
        return normalized_args

    identifier_candidates = _identifier_candidates(
        args.get("target_fields"),
        args.get("output_columns"),
    )
    if not identifier_candidates:
        return normalized_args

    reversed_bindings: dict[str, str] = {}
    for maybe_evidence, maybe_field in bindings.items():
        if not _evidence_is_in_question(question, maybe_evidence):
            return normalized_args
        normalized_field = _normalized_field_name(maybe_field)
        if normalized_field not in identifier_candidates:
            return normalized_args
        reversed_bindings[maybe_field] = maybe_evidence

    if len(reversed_bindings) != len(bindings):
        return normalized_args
    normalized_args["field_bindings"] = reversed_bindings
    return normalized_args


def _role_spans_overlap(left: str, right: str) -> bool:
    left_key = _evidence_key(left)
    right_key = _evidence_key(right)
    if left_key in right_key or right_key in left_key:
        return True

    left_words = set(re.findall(r"[a-z0-9_]{3,}", left_key))
    right_words = set(re.findall(r"[a-z0-9_]{3,}", right_key))
    if left_words & right_words:
        return True

    left_cjk = "".join(re.findall(r"[\u3400-\u9fff]", left_key))
    right_cjk = "".join(re.findall(r"[\u3400-\u9fff]", right_key))
    return any(
        left_cjk[index : index + 2] in right_cjk
        for index in range(max(len(left_cjk) - 1, 0))
    )


def _request_contract_from_values(
    *,
    requested_outputs: StringListInput,
    scope_evidence: StringListInput,
    request_mode_evidence: StringListInput,
    field_bindings: EvidenceMapInput,
    derived_field_bindings: EvidenceMapInput,
    filter_evidence: StringListInput,
    grouping_evidence: StringListInput,
    transformation_evidence: StringListInput,
) -> dict[str, Any]:
    return {
        "requested_outputs": _normalize_string_list(requested_outputs),
        "scope_evidence": _normalize_string_list(scope_evidence),
        "request_mode_evidence": _normalize_string_list(request_mode_evidence),
        "field_bindings": _normalize_evidence_map(field_bindings),
        "derived_field_bindings": _normalize_evidence_map(derived_field_bindings),
        "filter_evidence": _normalize_string_list(filter_evidence),
        "grouping_evidence": _normalize_string_list(grouping_evidence),
        "transformation_evidence": _normalize_string_list(transformation_evidence),
    }


def _request_contract_from_args(args: dict[str, Any]) -> dict[str, Any]:
    return _request_contract_from_values(
        requested_outputs=args.get("requested_outputs"),
        scope_evidence=args.get("scope_evidence"),
        request_mode_evidence=args.get("request_mode_evidence"),
        field_bindings=args.get("field_bindings"),
        derived_field_bindings=args.get("derived_field_bindings"),
        filter_evidence=args.get("filter_evidence"),
        grouping_evidence=args.get("grouping_evidence"),
        transformation_evidence=args.get("transformation_evidence"),
    )


def _contract_evidence_sets(contract: dict[str, Any]) -> dict[str, set[str]]:
    evidence_sets = {
        key: {_evidence_key(item) for item in _normalize_string_list(contract.get(key))}
        for key in (
            "requested_outputs",
            "scope_evidence",
            "request_mode_evidence",
            "filter_evidence",
            "grouping_evidence",
            "transformation_evidence",
        )
    }
    evidence_sets["derived_field_bindings"] = {
        _evidence_key(item)
        for item in _normalize_evidence_map(
            contract.get("derived_field_bindings")
        ).values()
    }
    return evidence_sets


def _source_binding_rewrite_error(
    *,
    args: dict[str, Any],
    previous_plan: dict[str, Any] | None,
) -> str | None:
    if not isinstance(previous_plan, dict):
        return None

    previous_tables = {
        _normalized_table_name(table)
        for table in _normalize_identifier_list(previous_plan.get("target_tables"))
    }
    previous_tables.discard("")
    revised_tables = {
        _normalized_table_name(table)
        for table in _normalize_identifier_list(args.get("target_tables"))
    }
    revised_tables.discard("")
    if previous_tables and not previous_tables.issubset(revised_tables):
        return (
            "schema source binding rejected: revise_plan cannot replace or remove "
            "target_tables established by analyze_plan. If the schema-defined data "
            "source is unavailable, keep the locked source binding, lower "
            "intent_confidence, record the conflict, and stop instead of switching "
            "to a different semantic data source; "
            f"missing_tables={sorted(previous_tables - revised_tables)}, "
            f"previous_tables={_normalize_identifier_list(previous_plan.get('target_tables'))}, "
            f"revised_tables={_normalize_identifier_list(args.get('target_tables'))}"
        )

    previous_fields = {
        _normalized_field_name(field)
        for field in _normalize_identifier_list(previous_plan.get("target_fields"))
    }
    previous_fields.discard("")
    revised_fields = {
        _normalized_field_name(field)
        for field in _normalize_identifier_list(args.get("target_fields"))
    }
    revised_fields.discard("")
    if previous_fields and not previous_fields.issubset(revised_fields):
        return (
            "schema source binding rejected: revise_plan cannot replace or remove "
            "target_fields established by analyze_plan. If the schema-defined field "
            "is unavailable, keep the locked field binding, lower intent_confidence, "
            "record the conflict, and stop instead of switching to a different "
            "semantic field; "
            f"missing_fields={sorted(previous_fields - revised_fields)}, "
            f"previous_fields={_normalize_identifier_list(previous_plan.get('target_fields'))}, "
            f"revised_fields={_normalize_identifier_list(args.get('target_fields'))}"
        )
    return None



def _validate_request_contract(
    *,
    question: str,
    args: dict[str, Any],
    previous_plan: dict[str, Any] | None,
) -> str | None:
    args = _normalize_plan_args_for_contract(args, question=question)
    operation = _infer_plan_operation(
        str(args.get("operation_type") or ""),
        group_by=args.get("group_by"),
        aggregation=args.get("aggregation"),
        transformation_evidence=args.get("transformation_evidence"),
    )
    output_columns = _planned_output_columns(args)
    filters = _normalize_string_list(args.get("filters"))
    group_by = _normalize_identifier_list(args.get("group_by"))
    aggregation = str(args.get("aggregation") or "").strip()
    contract = _canonical_request_contract(
        question,
        _request_contract_from_args(args),
    )
    source_error = _source_binding_rewrite_error(
        args=args,
        previous_plan=previous_plan,
    )
    if source_error is not None:
        return source_error

    all_evidence = [
        *contract["requested_outputs"],
        *contract["scope_evidence"],
        *contract["request_mode_evidence"],
        *_normalize_evidence_map(contract.get("derived_field_bindings")).values(),
        *contract["filter_evidence"],
        *contract["grouping_evidence"],
        *contract["transformation_evidence"],
    ]
    fabricated = [
        evidence
        for evidence in all_evidence
        if not _evidence_is_in_question(question, evidence)
    ]
    if fabricated:
        return (
            "request contract contains evidence that is not a verbatim span of the "
            f"original question: {fabricated}"
        )

    requested_keys = {_evidence_key(item) for item in contract["requested_outputs"]}
    mode_keys = {_evidence_key(item) for item in contract["request_mode_evidence"]}
    if (contract["requested_outputs"] or contract["field_bindings"]) and not mode_keys:
        return (
            "request_mode_evidence must quote the wording that asks to retrieve, "
            "display, compare, summarize, or otherwise act on the data"
        )
    bindings = contract["field_bindings"]
    mode_bound_fields = {
        field: evidence
        for field, evidence in bindings.items()
        if any(
            _role_spans_overlap(evidence, mode)
            for mode in contract["request_mode_evidence"]
        )
    }
    if mode_bound_fields:
        return (
            "output fields cannot bind to wording that describes how the user wants "
            "the data retrieved or displayed. Remove these fields from target_fields, "
            f"output_columns, and field_bindings: {mode_bound_fields}"
        )

    derived_bindings = contract["derived_field_bindings"]
    binding_fields = {_normalized_field_name(field) for field in bindings}
    derived_fields = {_normalized_field_name(field) for field in derived_bindings}
    expected_fields = {_normalized_field_name(field) for field in output_columns}
    if output_columns and not contract["requested_outputs"] and not derived_bindings:
        return (
            "requested_outputs or derived_field_bindings must quote the user wording "
            "that authorizes each output concept"
        )
    covered_fields = binding_fields | derived_fields
    if covered_fields != expected_fields:
        missing = sorted(expected_fields - covered_fields)
        extra = sorted(covered_fields - expected_fields)
        return (
            "field_bindings and derived_field_bindings must cover exactly the planned "
            "output columns; "
            f"missing={missing}, extra={extra}"
        )

    allowed_binding_evidence = requested_keys | {
        _evidence_key(item) for item in contract["grouping_evidence"]
    }
    unsupported_bindings = {
        field: evidence
        for field, evidence in bindings.items()
        if _evidence_key(evidence) not in allowed_binding_evidence
    }
    if unsupported_bindings:
        return (
            "each output field must bind to a requested output concept or explicit "
            f"grouping dimension, never to scope wording: {unsupported_bindings}"
        )
    derived_allowed_evidence = (
        requested_keys
        | {_evidence_key(item) for item in contract["request_mode_evidence"]}
        | {_evidence_key(item) for item in contract["grouping_evidence"]}
        | {_evidence_key(item) for item in contract["transformation_evidence"]}
    )
    unsupported_derived_bindings = {
        field: evidence
        for field, evidence in derived_bindings.items()
        if _evidence_key(evidence) not in derived_allowed_evidence
    }
    if unsupported_derived_bindings:
        return (
            "derived_field_bindings must bind derived output columns to explicit "
            "request-mode, grouping, output, or transformation wording from the "
            f"original question: {unsupported_derived_bindings}"
        )

    if filters and not contract["filter_evidence"]:
        return "filters require verbatim filter_evidence from the original question"
    if group_by and not contract["grouping_evidence"]:
        return "group_by requires verbatim grouping_evidence from the original question"
    if (
        operation in AGGREGATE_OPERATION_TYPES
        or operation == "compute"
        or aggregation
    ) and not contract["transformation_evidence"] and not derived_bindings:
        return (
            "derived operations require verbatim transformation_evidence or explicit "
            "derived_field_bindings from the original question"
        )
    if operation in RAW_OPERATION_TYPES and (group_by or aggregation):
        return "raw extraction cannot contain group_by or aggregation"
    if (
        operation in RAW_OPERATION_TYPES
        and args.get("preserve_raw_rows") is False
        and not contract["filter_evidence"]
    ):
        return (
            "changing source rows requires verbatim filter_evidence from the original "
            "question"
        )

    previous_contract = (
        previous_plan.get("request_contract")
        if isinstance(previous_plan, dict)
        else None
    )
    if isinstance(previous_contract, dict) and any(
        _normalize_string_list(previous_contract.get(role))
        for role in (
            "requested_outputs",
            "scope_evidence",
            "request_mode_evidence",
            "filter_evidence",
            "grouping_evidence",
            "transformation_evidence",
            "derived_field_bindings",
        )
    ):
        previous_sets = _contract_evidence_sets(previous_contract)
        revised_sets = _contract_evidence_sets(contract)
        expanded_roles = {
            role: sorted(revised_sets[role] - previous_sets[role])
            for role in revised_sets
            if revised_sets[role] - previous_sets[role]
        }
        if expanded_roles:
            return (
                "revise_plan cannot broaden the locked request contract with new output, "
                f"scope, filter, grouping, or transformation evidence: {expanded_roles}"
            )
        previous_outputs = previous_sets["requested_outputs"]
        if any(
            _evidence_key(evidence) not in previous_outputs
            for evidence in bindings.values()
            if _evidence_key(evidence) not in revised_sets["grouping_evidence"]
        ):
            return (
                "revise_plan may rebind source fields, but every value binding must use "
                "an output concept locked by analyze_plan"
            )
    return None


def _normalize_intent_confidence(value: float | int | str | None) -> float | None:
    try:
        confidence = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    if confidence is None or not 0.0 <= confidence <= 1.0:
        return None
    return confidence


def _plan_intent_confidence(plan: Any) -> float:
    if not isinstance(plan, dict):
        return 0.0
    confidence = _normalize_intent_confidence(plan.get("intent_confidence"))
    return confidence if confidence is not None else 0.0


def _has_observed_data_evidence(state: BenchmarkDeepAgentState) -> bool:
    evidence_tools = {
        "execute_python",
        "glob",
        "grep",
        "ls",
        "read_file",
        "task",
    }
    return any(
        isinstance(message, ToolMessage)
        and str(message.name or "") in evidence_tools
        and str(message.status or "success") != "error"
        for message in state.get("messages", [])
    )


def _plan_submission_readiness_error(plan: Any) -> str | None:
    if not isinstance(plan, dict):
        return "analysis_plan is missing."
    contract = plan.get("request_contract")
    if not isinstance(contract, dict):
        return None

    has_contract = any(
        _normalize_string_list(contract.get(key))
        for key in (
            "requested_outputs",
            "scope_evidence",
            "request_mode_evidence",
            "filter_evidence",
            "grouping_evidence",
            "transformation_evidence",
        )
    ) or bool(_normalize_evidence_map(contract.get("field_bindings"))) or bool(
        _normalize_evidence_map(contract.get("derived_field_bindings"))
    )
    if not has_contract:
        return None

    output_columns = _normalize_identifier_list(plan.get("output_columns"))
    field_bindings = _normalize_evidence_map(
        (plan.get("request_contract") or {}).get("field_bindings")
    )
    derived_bindings = _normalize_evidence_map(
        (plan.get("request_contract") or {}).get("derived_field_bindings")
    )
    requested_outputs = _normalize_string_list(contract.get("requested_outputs"))
    if (requested_outputs or field_bindings or derived_bindings) and not output_columns:
        return (
            "latest plan is not executable: requested outputs exist but output_columns "
            "is empty. Inspect schema/data evidence and call revise_plan with the "
            "actual output columns and bindings before answer."
        )

    filters = _normalize_string_list(plan.get("filters"))
    if _normalize_string_list(contract.get("filter_evidence")) and not filters:
        return (
            "latest plan is not executable: filter_evidence exists but filters is "
            "empty. Revise the plan with the verified executable filter before answer."
        )

    group_by = _normalize_identifier_list(plan.get("group_by"))
    if _normalize_string_list(contract.get("grouping_evidence")) and not group_by:
        return (
            "latest plan is not executable: grouping_evidence exists but group_by is "
            "empty. Revise the plan with the verified grouping field before answer."
        )

    operation = str(plan.get("operation_type") or "")
    aggregation = str(plan.get("aggregation") or "").strip()
    if (
        _normalize_string_list(contract.get("transformation_evidence"))
        or derived_bindings
    ) and not aggregation and operation not in AGGREGATE_OPERATION_TYPES | {"compute"}:
        return (
            "latest plan is not executable: derived output evidence exists but the "
            "plan has no executable aggregation or compute operation. Revise the plan "
            "with the verified calculation before answer."
        )
    return None


def _todos_are_completed(todos: Any) -> bool:
    if not isinstance(todos, list) or not todos:
        return False
    for todo in todos:
        if not isinstance(todo, dict):
            return False
        if str(todo.get("status") or "").lower() != "completed":
            return False
    return True


def _response_has_tool_calls(response: ModelResponse[Any]) -> bool:
    return any(
        isinstance(message, AIMessage) and bool(message.tool_calls)
        for message in response.result
    )


def _planning_error_code(error: str) -> str:
    if "not a verbatim span" in error:
        return "fabricated_evidence"
    if "request_mode_evidence" in error:
        return "missing_request_mode_evidence"
    if "output fields cannot bind" in error:
        return "mode_bound_output"
    if "field_bindings must cover" in error:
        return "binding_output_mismatch"
    if "each output field must bind" in error:
        return "unsupported_binding_evidence"
    if "filters require" in error:
        return "filter_without_evidence"
    if "group_by requires" in error:
        return "group_by_without_evidence"
    if "derived operations require" in error:
        return "derived_without_evidence"
    if "raw extraction cannot" in error:
        return "raw_operation_with_grouping"
    if "changing source rows requires" in error:
        return "raw_row_change_without_evidence"
    if "cannot broaden" in error:
        return "revise_contract_broadened"
    if "may rebind source fields" in error:
        return "revise_binding_not_locked"
    if "schema source binding rejected" in error:
        return "schema_source_rebinding"
    return _normalized_field_name(error[:80]) or "unknown_planning_error"


def _planning_error_count(
    messages: list[BaseMessage],
    *,
    tool_name: str,
    code: str,
) -> int:
    prefix = "Request contract rejected: "
    count = 0
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if str(message.name or "") != tool_name:
            continue
        if str(message.status or "") != "error":
            continue
        content = str(message.content or "")
        if not content.startswith(prefix):
            continue
        previous_error = content.removeprefix(prefix).rstrip(".")
        if _planning_error_code(previous_error) == code:
            count += 1
    return count


def _format_contract_error(
    *,
    question: str,
    contract_error: str,
    error_code: str,
) -> str:
    guidance = [
        f"Request contract rejected: {contract_error}.",
        f"Original question exact text: {question}",
    ]
    if error_code == "fabricated_evidence":
        guidance.append(
            "Every evidence item must be copied as an exact contiguous substring "
            "from the original question after whitespace is ignored. Do not "
            "rewrite, translate, normalize units, or add explanatory words."
        )
    elif error_code == "mode_bound_output":
        guidance.append(
            "Do not put computed/statistical output columns in field_bindings. "
            "Keep field_bindings for direct source or grouping outputs; put "
            "computed outputs such as count, ratio, or totals in "
            "derived_field_bindings, bound to exact original question wording."
        )
    elif error_code == "binding_output_mismatch":
        guidance.append(
            "Each output column must be covered exactly once by either "
            "field_bindings for direct outputs or derived_field_bindings for "
            "computed/statistical outputs."
        )
    elif error_code == "derived_without_evidence":
        guidance.append(
            "Derived operations require either transformation_evidence copied "
            "exactly from the original question or explicit derived_field_bindings "
            "for every computed/statistical output column."
        )
    elif error_code == "schema_source_rebinding":
        guidance.append(
            "Use the knowledge schema as the semantic baseline. Do not replace a "
            "schema-defined table or field with a different data source only because "
            "the locked source is missing. Keep the locked binding, lower confidence, "
            "record the conflict, and do not submit an alternate calculation."
        )
    return " ".join(guidance)


def _planning_block_command(
    *,
    tool_name: str,
    tool_call_id: str,
    code: str,
    contract_error: str,
    formatted_error: str,
    repeated_count: int,
) -> Command[BenchmarkDeepAgentState]:
    reason = (
        f"Repeated {tool_name} contract rejection ({code}) after "
        f"{repeated_count} attempts: {contract_error}. {formatted_error} "
        "The agent must stop "
        "instead of retrying the same invalid plan; inspect the original "
        "question audit and observed evidence before another run."
    )
    return Command(
        update={
            "planning_blocked": reason,
            "messages": [
                ToolMessage(
                    content=reason,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    status="error",
                )
            ],
        }
    )


def _build_analysis_plan(
    *,
    question: str,
    intent: str,
    output_spec: str,
    steps: StringListInput,
    delegation_candidates: StringListInput,
    operation_type: str,
    target_tables: StringListInput,
    target_fields: StringListInput,
    filters: StringListInput,
    group_by: StringListInput,
    aggregation: str | None,
    preserve_raw_rows: bool | None,
    output_columns: StringListInput,
    ambiguities: StringListInput,
    requested_outputs: StringListInput,
    scope_evidence: StringListInput,
    request_mode_evidence: StringListInput,
    field_bindings: EvidenceMapInput,
    derived_field_bindings: EvidenceMapInput,
    filter_evidence: StringListInput,
    grouping_evidence: StringListInput,
    transformation_evidence: StringListInput,
    question_audit: QuestionAuditInput,
    verification_questions: StringListInput,
    intent_confidence: float | int | str | None,
    confidence_reason: str,
) -> dict[str, Any] | str:
    if not intent.strip() or not output_spec.strip():
        return "intent and output_spec must be non-empty."
    normalized_steps = _normalize_string_list(steps)
    if not normalized_steps:
        return "steps must contain at least one non-empty plan step."
    normalized_confidence = _normalize_intent_confidence(intent_confidence)
    if normalized_confidence is None:
        return "intent_confidence must be a number between 0 and 1."
    if not confidence_reason.strip():
        return "confidence_reason must explain the current intent confidence."

    operation = _infer_plan_operation(
        operation_type,
        group_by=group_by,
        aggregation=aggregation,
        transformation_evidence=transformation_evidence,
    )
    normalized_target_fields = _normalize_identifier_list(target_fields)
    normalized_group_by = _normalize_identifier_list(group_by)
    normalized_output_columns = _normalize_identifier_list(output_columns)
    normalized_binding_args = _normalize_plan_args_for_contract(
        {
            "field_bindings": field_bindings,
            "target_fields": target_fields,
            "output_columns": output_columns,
        },
        question=question,
    )
    normalized_bindings = _normalize_evidence_map(
        normalized_binding_args.get("field_bindings")
    )
    normalized_bindings = _canonical_question_evidence_map(
        question,
        normalized_bindings,
    )
    normalized_derived_bindings = _normalize_evidence_map(derived_field_bindings)
    normalized_derived_bindings = _canonical_question_evidence_map(
        question,
        normalized_derived_bindings,
    )
    if normalized_bindings or normalized_derived_bindings:
        normalized_output_columns = [
            *list(normalized_bindings),
            *list(normalized_derived_bindings),
        ]
        if operation in RAW_OPERATION_TYPES:
            normalized_target_fields = list(normalized_bindings)
    elif not normalized_output_columns:
        normalized_output_columns = list(normalized_target_fields)
    if not normalized_output_columns:
        normalized_output_columns = list(normalized_bindings)
    normalized_aggregation = aggregation.strip() if isinstance(aggregation, str) else ""
    planning_warnings: list[str] = []
    if operation not in SUPPORTED_OPERATION_TYPES:
        planning_warnings.append(
            "operation_type was not recognized; treating it as unknown rather than "
            "blocking the run."
        )
        operation = "unknown"
    if operation in RAW_OPERATION_TYPES and (normalized_group_by or normalized_aggregation):
        planning_warnings.append(
            "Raw-record operation includes group_by or aggregation; verify the user "
            "explicitly requested a derived summary before aggregating."
        )
    if operation in AGGREGATE_OPERATION_TYPES:
        if not normalized_group_by and operation == "aggregate":
            planning_warnings.append(
                "Aggregate plan has no group_by; verify whether aggregation is really intended."
            )
        if not normalized_aggregation:
            planning_warnings.append(
                "Aggregate/rank plan has no aggregation rule; verify whether a raw record "
                "extract is more faithful."
            )

    request_contract = _canonical_request_contract(
        question,
        _request_contract_from_values(
            requested_outputs=requested_outputs,
            scope_evidence=scope_evidence,
            request_mode_evidence=request_mode_evidence,
            field_bindings=normalized_bindings,
            derived_field_bindings=normalized_derived_bindings,
            filter_evidence=filter_evidence,
            grouping_evidence=grouping_evidence,
            transformation_evidence=transformation_evidence,
        ),
    )
    return {
        "intent": intent.strip(),
        "output_spec": output_spec.strip(),
        "intent_confidence": normalized_confidence,
        "confidence_reason": confidence_reason.strip(),
        "operation_type": operation,
        "target_tables": _normalize_identifier_list(target_tables),
        "target_fields": normalized_target_fields,
        "filters": _normalize_string_list(filters),
        "group_by": normalized_group_by,
        "aggregation": normalized_aggregation or None,
        "preserve_raw_rows": (
            preserve_raw_rows if preserve_raw_rows is not None else operation in RAW_OPERATION_TYPES
        ),
        "output_columns": normalized_output_columns,
        "request_contract": request_contract,
        "question_audit": _auto_question_audit(
            question=question,
            contract=request_contract,
            supplied_audit=question_audit,
        ),
        "evidence_bindings": {
            "target_tables": _normalize_identifier_list(target_tables),
            "target_fields": normalized_target_fields,
            "field_bindings": normalized_bindings,
            "derived_field_bindings": normalized_derived_bindings,
            "operation_type": operation,
            "verified": False,
        },
        "verification_questions": _normalize_string_list(verification_questions),
        "ambiguities": _normalize_string_list(ambiguities),
        "planning_warnings": planning_warnings,
        "steps": normalized_steps,
        "delegation_candidates": _normalize_string_list(delegation_candidates),
    }


@tool("analyze_plan")
def _analyze_plan_tool(
    intent: str,
    output_spec: str,
    steps: StringListInput,
    delegation_candidates: StringListInput,
    intent_confidence: float,
    confidence_reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[BenchmarkDeepAgentState, InjectedState],
    requested_outputs: StringListInput = None,
    request_mode_evidence: StringListInput = None,
    field_bindings: EvidenceMapInput = None,
    derived_field_bindings: EvidenceMapInput = None,
    operation_type: str = "unknown",
    target_tables: StringListInput = None,
    target_fields: StringListInput = None,
    filters: StringListInput = None,
    group_by: StringListInput = None,
    aggregation: str | None = None,
    preserve_raw_rows: bool | None = None,
    output_columns: StringListInput = None,
    ambiguities: StringListInput = None,
    scope_evidence: StringListInput = None,
    filter_evidence: StringListInput = None,
    grouping_evidence: StringListInput = None,
    transformation_evidence: StringListInput = None,
    question_audit: QuestionAuditInput = None,
    verification_questions: StringListInput = None,
) -> Command[BenchmarkDeepAgentState] | ToolMessage:
    """Record the question interpretation and execution plan before data access.

    Build a request contract from verbatim spans of the original question.
    requested_outputs contains only value concepts the user wants returned.
    request_mode_evidence quotes wording that asks to retrieve, display,
    compare, summarize, or otherwise act on data; scope_evidence quotes
    contextual scope. Neither role authorizes output columns. Bind every
    output column to a requested output concept (or an explicit grouping
    dimension) with field_bindings. Bind computed/statistical output columns
    such as counts or ratios with derived_field_bindings. Filters, grouping,
    and derived transformations each require evidence in their dedicated slot.
    """
    original_question = str(state.get("original_question") or "")
    plan = _build_analysis_plan(
        question=original_question,
        intent=intent,
        output_spec=output_spec,
        steps=steps,
        delegation_candidates=delegation_candidates,
        operation_type=operation_type,
        target_tables=target_tables,
        target_fields=target_fields,
        filters=filters,
        group_by=group_by,
        aggregation=aggregation,
        preserve_raw_rows=preserve_raw_rows,
        output_columns=output_columns,
        ambiguities=ambiguities,
        requested_outputs=requested_outputs,
        scope_evidence=scope_evidence,
        request_mode_evidence=request_mode_evidence,
        field_bindings=field_bindings,
        derived_field_bindings=derived_field_bindings,
        filter_evidence=filter_evidence,
        grouping_evidence=grouping_evidence,
        transformation_evidence=transformation_evidence,
        question_audit=question_audit,
        verification_questions=verification_questions,
        intent_confidence=intent_confidence,
        confidence_reason=confidence_reason,
    )
    if isinstance(plan, str):
        return ToolMessage(
            content=plan,
            name="analyze_plan",
            tool_call_id=tool_call_id,
            status="error",
        )
    confidence_update = {
        "previous": None,
        "new": plan["intent_confidence"],
        "delta": None,
        "reason": plan["confidence_reason"],
        "original_question": original_question,
        "data_evidence": [],
    }
    plan["confidence_history"] = [confidence_update]
    plan["confidence_basis"] = confidence_update
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


@tool("revise_plan")
def _revise_plan_tool(
    revision_reason: str,
    evidence: StringListInput,
    intent: str,
    output_spec: str,
    steps: StringListInput,
    delegation_candidates: StringListInput,
    intent_confidence: float,
    confidence_reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[BenchmarkDeepAgentState, InjectedState],
    requested_outputs: StringListInput = None,
    request_mode_evidence: StringListInput = None,
    field_bindings: EvidenceMapInput = None,
    derived_field_bindings: EvidenceMapInput = None,
    operation_type: str = "unknown",
    target_tables: StringListInput = None,
    target_fields: StringListInput = None,
    filters: StringListInput = None,
    group_by: StringListInput = None,
    aggregation: str | None = None,
    preserve_raw_rows: bool | None = None,
    output_columns: StringListInput = None,
    ambiguities: StringListInput = None,
    scope_evidence: StringListInput = None,
    filter_evidence: StringListInput = None,
    grouping_evidence: StringListInput = None,
    transformation_evidence: StringListInput = None,
    question_audit: QuestionAuditInput = None,
    verification_questions: StringListInput = None,
    conflict_points: StringListInput = None,
    question_evidence: StringListInput = None,
    superseded_plan_reason: str | None = None,
) -> Command[BenchmarkDeepAgentState] | ToolMessage:
    """Revise the current interpretation after inspecting data evidence."""
    normalized_reason = revision_reason.strip()
    normalized_evidence = _normalize_string_list(evidence)
    if not normalized_reason or not normalized_evidence:
        return ToolMessage(
            content="revision_reason and evidence must be non-empty.",
            name="revise_plan",
            tool_call_id=tool_call_id,
            status="error",
        )
    original_question = str(state.get("original_question") or "")
    normalized_conflicts = _normalize_string_list(conflict_points)
    normalized_question_evidence = _normalize_string_list(question_evidence)
    fabricated_question_evidence = [
        item
        for item in normalized_question_evidence
        if not _evidence_is_in_question(original_question, item)
    ]
    if fabricated_question_evidence:
        return ToolMessage(
            content=(
                "question_evidence must quote verbatim spans from the original "
                f"question: {fabricated_question_evidence}."
            ),
            name="revise_plan",
            tool_call_id=tool_call_id,
            status="error",
        )
    normalized_question_evidence = _canonical_question_evidence_list(
        original_question,
        normalized_question_evidence,
    )
    plan = _build_analysis_plan(
        question=original_question,
        intent=intent,
        output_spec=output_spec,
        steps=steps,
        delegation_candidates=delegation_candidates,
        operation_type=operation_type,
        target_tables=target_tables,
        target_fields=target_fields,
        filters=filters,
        group_by=group_by,
        aggregation=aggregation,
        preserve_raw_rows=preserve_raw_rows,
        output_columns=output_columns,
        ambiguities=ambiguities,
        requested_outputs=requested_outputs,
        scope_evidence=scope_evidence,
        request_mode_evidence=request_mode_evidence,
        field_bindings=field_bindings,
        derived_field_bindings=derived_field_bindings,
        filter_evidence=filter_evidence,
        grouping_evidence=grouping_evidence,
        transformation_evidence=transformation_evidence,
        question_audit=question_audit,
        verification_questions=verification_questions,
        intent_confidence=intent_confidence,
        confidence_reason=confidence_reason,
    )
    if isinstance(plan, str):
        return ToolMessage(
            content=plan,
            name="revise_plan",
            tool_call_id=tool_call_id,
            status="error",
        )
    previous_plan = state.get("analysis_plan")
    previous_confidence = _plan_intent_confidence(previous_plan)
    confidence_update = {
        "previous": previous_confidence,
        "new": plan["intent_confidence"],
        "delta": plan["intent_confidence"] - previous_confidence,
        "reason": plan["confidence_reason"],
        "original_question": str(state.get("original_question") or ""),
        "data_evidence": normalized_evidence,
    }
    previous_history = (
        list(previous_plan.get("confidence_history") or [])
        if isinstance(previous_plan, dict)
        else []
    )
    plan["confidence_history"] = [*previous_history, confidence_update]
    plan["confidence_basis"] = confidence_update
    revision = {
        "revision_reason": normalized_reason,
        "evidence": normalized_evidence,
        "conflict_points": normalized_conflicts,
        "question_evidence": normalized_question_evidence,
        "superseded_plan_reason": (
            superseded_plan_reason.strip()
            if isinstance(superseded_plan_reason, str)
            else ""
        ),
        "confidence_update": confidence_update,
        "revised_plan": plan,
    }
    return Command(
        update={
            "analysis_plan": plan,
            "plan_revisions": [
                *list(state.get("plan_revisions") or []),
                revision,
            ],
            "todos": [],
            "messages": [
                ToolMessage(
                    content=json.dumps(revision, ensure_ascii=False),
                    name="revise_plan",
                    tool_call_id=tool_call_id,
                    status="success",
                )
            ],
        }
    )


def _answer_error(tool_call_id: str, content: str) -> ToolMessage:
    return ToolMessage(
        content=content,
        name="answer",
        tool_call_id=tool_call_id,
        status="error",
    )


def _validate_answer_table(
    columns: list[str],
    rows: list[list[Any]],
    *,
    tool_call_id: str,
) -> AnswerTable | ToolMessage:
    if not columns or not all(isinstance(column, str) and column for column in columns):
        return _answer_error(
            tool_call_id,
            "answer.columns must be a non-empty list of non-empty strings.",
        )
    if not rows:
        return _answer_error(
            tool_call_id,
            "answer.rows must contain at least one row.",
        )

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if len(row) != len(columns):
            return _answer_error(
                tool_call_id,
                "Each answer row must match the number of columns.",
            )
        normalized_rows.append(list(row))
    return AnswerTable(columns=list(columns), rows=normalized_rows)


def _resolve_virtual_file(
    workspace: Path,
    virtual_path: str,
    *,
    allowed_roots: tuple[str, ...],
) -> Path:
    normalized = virtual_path.replace("\\", "/")
    for virtual_root in allowed_roots:
        prefix = f"/{virtual_root}"
        if normalized != prefix and not normalized.startswith(f"{prefix}/"):
            continue
        relative_path = normalized.removeprefix(prefix).lstrip("/")
        root = (workspace / virtual_root).resolve()
        resolved = (root / Path(relative_path)).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Virtual path escapes /{virtual_root}: {virtual_path}")
        return resolved
    roots = ", ".join(f"/{root}/..." for root in allowed_roots)
    raise ValueError(f"Path must use one of these virtual roots: {roots}")


def _load_source_projection(
    workspace: Path,
    source_path: str,
    source_columns: list[str],
) -> AnswerTable:
    if not source_columns or not all(
        isinstance(column, str) and column for column in source_columns
    ):
        raise ValueError("source_columns must be a non-empty list of column names.")
    resolved_path = _resolve_virtual_file(
        workspace,
        source_path,
        allowed_roots=("context",),
    )
    if not resolved_path.is_file():
        raise ValueError(f"Source file does not exist: {source_path}")

    suffix = resolved_path.suffix.lower()
    records: list[dict[str, Any]]
    if suffix == ".json":
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            raw_records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
            raw_records = payload["records"]
        else:
            raise ValueError(
                "JSON source must be a list of objects or contain a top-level records list."
            )
        if not all(isinstance(record, dict) for record in raw_records):
            raise ValueError("JSON source records must all be objects.")
        records = [dict(record) for record in raw_records]
    elif suffix == ".csv":
        with resolved_path.open(newline="", encoding="utf-8-sig") as handle:
            records = [dict(record) for record in csv.DictReader(handle)]
    else:
        raise ValueError("source_path currently supports JSON and CSV files.")

    available_columns = {
        str(column)
        for record in records
        for column in record
    }
    missing_columns = [
        column for column in source_columns if column not in available_columns
    ]
    if missing_columns:
        raise ValueError(f"Source columns not found: {missing_columns}")
    rows = [[record.get(column) for column in source_columns] for record in records]
    return AnswerTable(columns=list(source_columns), rows=rows)


def _normalize_answer_rows(
    values: AnswerRowsInput,
    columns: list[str],
) -> list[list[Any]]:
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except json.JSONDecodeError as exc:
            raise ValueError(f"rows is not valid JSON: {exc}") from exc
    if not isinstance(values, list):
        raise ValueError("rows must be a list or a JSON-encoded list.")

    normalized_rows: list[list[Any]] = []
    for row in values:
        if isinstance(row, dict):
            normalized_rows.append([row.get(column) for column in columns])
        elif isinstance(row, list):
            normalized_rows.append(list(row))
        else:
            raise ValueError("Each answer row must be a list or an object.")
    return normalized_rows


def _load_staged_answer(
    workspace: Path,
    answer_path: str,
    fallback_columns: list[str],
) -> AnswerTable:
    resolved_path = _resolve_virtual_file(
        workspace,
        answer_path,
        allowed_roots=("scratch",),
    )
    if not resolved_path.is_file():
        raise ValueError(f"Staged answer file does not exist: {answer_path}")
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        columns = _normalize_string_list(payload.get("columns")) or fallback_columns
        rows = payload.get("rows")
    elif isinstance(payload, list):
        columns = fallback_columns
        rows = payload
    else:
        raise ValueError(
            "Staged answer JSON must be an object with columns/rows or a row list."
        )
    if not columns:
        raise ValueError("Staged answer columns are missing.")
    return AnswerTable(
        columns=list(columns),
        rows=_normalize_answer_rows(rows, columns),
    )


def _create_answer_tool(workspace: Path) -> BaseTool:
    @tool("answer")
    def answer(
        tool_call_id: Annotated[str, InjectedToolCallId],
        columns: StringListInput = None,
        rows: AnswerRowsInput = None,
        source_path: str | None = None,
        source_columns: StringListInput = None,
        answer_path: str | None = None,
    ) -> Command[BenchmarkDeepAgentState] | ToolMessage:
        """Submit the final answer table.

        For an unchanged projection of columns from a JSON or CSV source, use
        source_path plus source_columns. This preserves source row order and NULLs
        without copying every row into the tool call. For a large computed result,
        write {"columns": [...], "rows": [...]} to /scratch and use answer_path.
        Otherwise pass columns and rows directly. Use exactly one mode.
        """
        normalized_columns = _normalize_string_list(columns)
        normalized_source_columns = _normalize_string_list(source_columns)
        try:
            if answer_path is not None:
                if rows is not None or source_path is not None or normalized_source_columns:
                    raise ValueError(
                        "answer_path cannot be combined with rows or source projection."
                    )
                candidate = _load_staged_answer(
                    workspace,
                    answer_path,
                    normalized_columns,
                )
            elif source_path is not None:
                if rows is not None:
                    raise ValueError("source_path cannot be combined with rows.")
                selected_columns = normalized_source_columns or normalized_columns
                if not selected_columns:
                    raise ValueError(
                        "source_columns or columns is required with source_path."
                    )
                candidate = _load_source_projection(
                    workspace,
                    source_path,
                    selected_columns,
                )
            else:
                if normalized_source_columns:
                    raise ValueError("source_columns requires source_path.")
                candidate = AnswerTable(
                    columns=normalized_columns,
                    rows=_normalize_answer_rows(rows, normalized_columns),
                )
        except (OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
            return _answer_error(tool_call_id, f"Unable to load answer data: {exc}")

        validated = _validate_answer_table(
            candidate.columns,
            candidate.rows,
            tool_call_id=tool_call_id,
        )
        if isinstance(validated, ToolMessage):
            return validated
        answer_table = validated

        content = json.dumps(
            {
                "status": "submitted",
                "column_count": len(answer_table.columns),
                "row_count": len(answer_table.rows),
            },
            ensure_ascii=False,
        )
        return Command(
            update={
                "answer": answer_table,
                "messages": [
                    ToolMessage(
                        content=content,
                        name="answer",
                        tool_call_id=tool_call_id,
                        status="success",
                    )
                ],
            }
        )

    return answer


@dataclass(frozen=True, slots=True)
class AutoAnswerCandidate:
    answer: AnswerTable
    mode: str
    source_path: str | None
    source_columns: list[str]


def _load_auto_staged_answer(
    workspace: Path,
    state: BenchmarkDeepAgentState,
) -> AutoAnswerCandidate | None:
    plan = state.get("analysis_plan")
    if not _todos_are_completed(state.get("todos")):
        return None
    if _plan_intent_confidence(plan) < MIN_FINAL_INTENT_CONFIDENCE:
        return None
    staged_path = workspace / "scratch" / "answer.json"
    if not staged_path.is_file():
        return None

    expected_columns = (
        _normalize_string_list(plan.get("output_columns"))
        if isinstance(plan, dict)
        else []
    )
    try:
        candidate = _load_staged_answer(
            workspace,
            "/scratch/answer.json",
            expected_columns,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    validated = _validate_answer_table(
        candidate.columns,
        candidate.rows,
        tool_call_id="auto-staged-answer",
    )
    if not isinstance(validated, AnswerTable):
        return None
    validated = _restore_source_column_names(workspace, validated, plan)
    if expected_columns:
        actual_names = [_normalized_field_name(column) for column in validated.columns]
        expected_names = [_normalized_field_name(column) for column in expected_columns]
        if actual_names != expected_names:
            return None
    return AutoAnswerCandidate(
        answer=validated,
        mode="staged_answer",
        source_path="/scratch/answer.json",
        source_columns=expected_columns or list(validated.columns),
    )


def _normalized_table_name(value: str) -> str:
    return _normalized_field_name(Path(value).stem)


def _read_source_column_names(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            return []
        record = next((item for item in records if isinstance(item, dict)), None)
        return list(record) if isinstance(record, dict) else []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle).fieldnames or [])
    return []


def _load_auto_source_projection(
    workspace: Path,
    state: BenchmarkDeepAgentState,
) -> AutoAnswerCandidate | None:
    plan = state.get("analysis_plan")
    if not isinstance(plan, dict):
        return None
    if not _todos_are_completed(state.get("todos")):
        return None
    if _plan_intent_confidence(plan) < MIN_FINAL_INTENT_CONFIDENCE:
        return None
    if _normalize_plan_operation(str(plan.get("operation_type") or "")) != "column_extract":
        return None
    if plan.get("preserve_raw_rows") is not True:
        return None
    if (
        _normalize_string_list(plan.get("filters"))
        or _normalize_string_list(plan.get("group_by"))
        or plan.get("aggregation")
    ):
        return None

    target_tables = {
        _normalized_table_name(table)
        for table in _normalize_string_list(plan.get("target_tables"))
    }
    target_tables.discard("")
    expected_columns = _normalize_string_list(plan.get("output_columns"))
    if len(target_tables) != 1 or not expected_columns:
        return None
    request_contract = plan.get("request_contract")
    if not isinstance(request_contract, dict):
        return None
    bound_columns = {
        _normalized_field_name(column)
        for column in _normalize_evidence_map(
            request_contract.get("field_bindings")
        )
    }
    if bound_columns != {
        _normalized_field_name(column) for column in expected_columns
    }:
        return None

    source_paths = [
        path
        for path in (workspace / "context").rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".json"}
        and _normalized_table_name(path.name) in target_tables
    ]
    if len(source_paths) != 1:
        return None
    source_path = source_paths[0]
    try:
        available_columns = _read_source_column_names(source_path)
    except (OSError, ValueError, json.JSONDecodeError, csv.Error):
        return None

    source_names: dict[str, set[str]] = {}
    for column in available_columns:
        source_names.setdefault(_normalized_field_name(column), set()).add(column)
    resolved_columns: list[str] = []
    for column in expected_columns:
        candidates = source_names.get(_normalized_field_name(column), set())
        if len(candidates) != 1:
            return None
        resolved_columns.append(next(iter(candidates)))

    virtual_path = "/context/" + source_path.relative_to(
        workspace / "context"
    ).as_posix()
    try:
        answer = _load_source_projection(workspace, virtual_path, resolved_columns)
    except (OSError, ValueError, json.JSONDecodeError, csv.Error):
        return None
    return AutoAnswerCandidate(
        answer=answer,
        mode="source_projection",
        source_path=virtual_path,
        source_columns=resolved_columns,
    )


def _restore_source_column_names(
    workspace: Path,
    answer: AnswerTable,
    plan: dict[str, Any] | None,
) -> AnswerTable:
    if not isinstance(plan, dict):
        return answer
    target_tables = {
        _normalized_table_name(table)
        for table in _normalize_string_list(plan.get("target_tables"))
    }
    target_tables.discard("")
    if not target_tables:
        return answer

    source_names: dict[str, set[str]] = {}
    context_root = workspace / "context"
    for path in context_root.rglob("*"):
        if not path.is_file() or _normalized_table_name(path.name) not in target_tables:
            continue
        try:
            columns = _read_source_column_names(path)
        except (OSError, ValueError, json.JSONDecodeError, csv.Error):
            continue
        for column in columns:
            source_names.setdefault(_normalized_field_name(column), set()).add(column)

    restored_columns = []
    for column in answer.columns:
        candidates = source_names.get(_normalized_field_name(column), set())
        restored_columns.append(next(iter(candidates)) if len(candidates) == 1 else column)
    if restored_columns == answer.columns:
        return answer
    return AnswerTable(columns=restored_columns, rows=answer.rows)


class AnswerMiddleware(AgentMiddleware[BenchmarkDeepAgentState, None, Any]):
    state_schema = BenchmarkDeepAgentState

    def __init__(
        self,
        answer_tool: BaseTool,
        workspace: Path,
        collector: Any | None = None,
    ) -> None:
        super().__init__()
        self.tools = [answer_tool]
        self.workspace = workspace
        self.collector = collector

    def _record_auto_answer(
        self,
        candidate: AutoAnswerCandidate,
        state: BenchmarkDeepAgentState,
    ) -> None:
        if self.collector is None:
            return
        plan = state.get("analysis_plan")
        expected_columns = (
            _normalize_string_list(plan.get("output_columns"))
            if isinstance(plan, dict)
            else []
        )
        verification_id = f"auto-plan-verification-{candidate.mode.replace('_', '-')}"
        self.collector.add(
            action="plan_verification",
            scope="main",
            action_input={
                "auto": True,
                "mode": candidate.mode,
                "latest_plan": {
                    "operation_type": (
                        str(plan.get("operation_type") or "")
                        if isinstance(plan, dict)
                        else ""
                    ),
                    "output_columns": expected_columns,
                    "preserve_raw_rows": (
                        plan.get("preserve_raw_rows")
                        if isinstance(plan, dict)
                        else None
                    ),
                },
                "todos_completed": _todos_are_completed(state.get("todos")),
                "source_path": candidate.source_path,
                "source_columns": candidate.source_columns,
            },
            tool_call_id=verification_id,
            observation={
                "status": "completed",
                "tool_calls": [
                    {
                        "name": "plan_verification",
                        "tool_call_id": verification_id,
                        "args": {
                            "auto": True,
                            "mode": candidate.mode,
                            "source_path": candidate.source_path,
                            "source_columns": candidate.source_columns,
                            "expected_columns": expected_columns,
                        },
                        "status": "success",
                        "ok": True,
                        "result": {
                            "content": json.dumps(
                                {
                                    "status": "verified",
                                    "mode": candidate.mode,
                                    "source_path": candidate.source_path,
                                    "source_columns": candidate.source_columns,
                                    "expected_columns": expected_columns,
                                    "actual_columns": list(candidate.answer.columns),
                                    "row_count": len(candidate.answer.rows),
                                },
                                ensure_ascii=False,
                            ),
                            "name": "plan_verification",
                            "tool_call_id": verification_id,
                            "status": "success",
                            "type": "tool",
                        },
                    }
                ],
                "checks": {
                    "intent_confidence": _plan_intent_confidence(plan),
                    "expected_columns": expected_columns,
                    "actual_columns": list(candidate.answer.columns),
                    "column_count": len(candidate.answer.columns),
                    "row_count": len(candidate.answer.rows),
                    "todos_completed": _todos_are_completed(state.get("todos")),
                    "source_path": candidate.source_path,
                    "source_columns": candidate.source_columns,
                },
            },
            ok=True,
        )
        tool_call_id = f"auto-{candidate.mode.replace('_', '-')}"
        result = {
            "content": json.dumps(
                {
                    "status": "submitted",
                    "mode": candidate.mode,
                    "source_path": candidate.source_path,
                    "source_columns": candidate.source_columns,
                    "column_count": len(candidate.answer.columns),
                    "row_count": len(candidate.answer.rows),
                },
                ensure_ascii=False,
            ),
            "name": "answer",
            "tool_call_id": tool_call_id,
            "status": "success",
            "type": "tool",
        }
        self.collector.add(
            action="answer",
            scope="main",
            action_input={
                "auto": True,
                "mode": candidate.mode,
                "source_path": candidate.source_path,
                "source_columns": candidate.source_columns,
            },
            tool_call_id=tool_call_id,
            observation={
                "status": "completed",
                "tool_calls": [
                    {
                        "name": "answer",
                        "tool_call_id": tool_call_id,
                        "args": {
                            "auto": True,
                            "mode": candidate.mode,
                            "source_path": candidate.source_path,
                            "source_columns": candidate.source_columns,
                        },
                        "status": "success",
                        "ok": True,
                        "result": result,
                    }
                ],
            },
            ok=True,
        )

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: BenchmarkDeepAgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        del runtime
        if state.get("answer") is not None:
            return {"jump_to": "end"}
        source_candidate = _load_auto_source_projection(self.workspace, state)
        if source_candidate is not None:
            self._record_auto_answer(source_candidate, state)
            return {
                "answer": source_candidate.answer,
                "jump_to": "end",
            }
        staged_candidate = _load_auto_staged_answer(self.workspace, state)
        if staged_candidate is not None:
            self._record_auto_answer(staged_candidate, state)
            return {
                "answer": staged_candidate.answer,
                "jump_to": "end",
            }
        return None


class PlanningMiddleware(AgentMiddleware[BenchmarkDeepAgentState, None, Any]):
    state_schema = BenchmarkDeepAgentState
    tools = [_analyze_plan_tool, _revise_plan_tool]

    def __init__(self, collector: Any | None = None) -> None:
        super().__init__()
        self.collector = collector

    def _short_circuit_tool_call(
        self,
        *,
        tool_call_id: str,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        if self.collector is not None:
            self.collector.complete_tool_call(
                tool_call_id=tool_call_id,
                result=_serialize_tool_result(result),
                ok=_tool_result_ok(result),
            )
        return result

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: BenchmarkDeepAgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        del runtime
        if state.get("planning_blocked"):
            return {"jump_to": "end"}
        return None

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        original_tools = request.tools
        if request.state.get("analysis_plan") is None:
            planning_tools = [
                item for item in request.tools if _tool_name(item) == "analyze_plan"
            ]
            request = request.override(tools=planning_tools, tool_choice="analyze_plan")
        elif not request.state.get("todos"):
            todo_tools = [
                item for item in request.tools if _tool_name(item) == "write_todos"
            ]
            request = request.override(tools=todo_tools, tool_choice="write_todos")
        elif request.state.get("answer") is None and _todos_are_completed(
            request.state.get("todos")
        ) and (
            readiness_error := _plan_submission_readiness_error(
                request.state.get("analysis_plan")
            )
        ):
            revision_tools = [
                item for item in request.tools if _tool_name(item) == "revise_plan"
            ]
            if revision_tools and _has_observed_data_evidence(request.state):
                request = request.override(
                    messages=[
                        *request.messages,
                        HumanMessage(
                            content=(
                                f"{readiness_error} The current completed todos cannot "
                                "lead to submission. Compare the original question, "
                                "current plan, and observed evidence; call revise_plan "
                                "with the corrected executable contract, then call "
                                "write_todos again."
                            )
                        ),
                    ],
                    tools=revision_tools,
                    tool_choice="revise_plan",
                )
            else:
                request = request.override(
                    messages=[
                        *request.messages,
                        HumanMessage(
                            content=(
                                f"{readiness_error} Do not call answer yet. Inspect the "
                                "relevant data/schema or revise the plan once evidence is "
                                "available."
                            )
                        ),
                    ],
                    tools=[
                        item for item in request.tools if _tool_name(item) != "answer"
                    ],
                )
        elif request.state.get("answer") is None and _todos_are_completed(
            request.state.get("todos")
        ) and _plan_intent_confidence(
            request.state.get("analysis_plan")
        ) >= MIN_FINAL_INTENT_CONFIDENCE:
            answer_tools = [item for item in request.tools if _tool_name(item) == "answer"]
            request = request.override(tools=answer_tools, tool_choice="answer")
        elif (
            request.state.get("answer") is None
            and _plan_intent_confidence(request.state.get("analysis_plan"))
            < MIN_FINAL_INTENT_CONFIDENCE
        ):
            request = request.override(
                tools=[
                    item for item in request.tools if _tool_name(item) != "answer"
                ]
            )
        response = handler(request)
        if request.state.get("answer") is not None or _response_has_tool_calls(response):
            return response
        required_tool = (
            request.tool_choice
            if isinstance(request.tool_choice, str)
            and request.tool_choice in {"analyze_plan", "write_todos"}
            else None
        )
        previous_required_error = required_tool is not None and any(
            isinstance(message, ToolMessage)
            and str(message.name or "") == required_tool
            and str(message.status or "") == "error"
            for message in request.messages
        )
        if required_tool is not None and previous_required_error:
            retry_request = request.override(
                messages=[
                    *request.messages,
                    *response.result,
                    HumanMessage(
                        content=(
                            f"The required {required_tool} call is still incomplete. "
                            "Use the latest tool error to correct only the invalid "
                            "arguments, then call the required tool again."
                        )
                    ),
                ]
            )
            return handler(retry_request)
        if request.tool_choice == "answer":
            return response

        if not any(_tool_name(item) == "answer" for item in request.tools):
            return response
        answer_tools = [item for item in original_tools if _tool_name(item) == "answer"]
        if not answer_tools:
            return response
        retry_messages = [
            *request.messages,
            *response.result,
            HumanMessage(
                content=(
                    "The previous response did not call the required answer tool. "
                    "Submit the final benchmark table now by calling answer exactly once; "
                    "do not reply with plain text."
                )
            ),
        ]
        retry_request = request.override(
            messages=retry_messages,
            tools=answer_tools,
            tool_choice="answer",
        )
        return handler(retry_request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_name = str(request.tool_call.get("name") or "")
        tool_call_id = str(request.tool_call.get("id") or "")
        if tool_name != "analyze_plan" and request.state.get("analysis_plan") is None:
            return self._short_circuit_tool_call(
                tool_call_id=tool_call_id,
                result=ToolMessage(
                    content="Call analyze_plan successfully before using any other tool.",
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    status="error",
                ),
            )
        if (
            request.state.get("analysis_plan") is not None
            and not request.state.get("todos")
            and tool_name != "write_todos"
        ):
            return self._short_circuit_tool_call(
                tool_call_id=tool_call_id,
                result=ToolMessage(
                    content="Call write_todos successfully before using any other tool.",
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    status="error",
                ),
            )
        if tool_name in {"analyze_plan", "revise_plan"}:
            args = request.tool_call.get("args")
            if isinstance(args, dict):
                if tool_name == "revise_plan" and not _has_observed_data_evidence(
                    request.state
                ):
                    return self._short_circuit_tool_call(
                        tool_call_id=tool_call_id,
                        result=ToolMessage(
                            content=(
                                "revise_plan requires evidence from a successful data, "
                                "schema, execution, or delegated-analysis tool call. "
                                "Inspect the data before updating intent confidence."
                            ),
                            name=tool_name,
                            tool_call_id=tool_call_id,
                            status="error",
                        ),
                    )
                question = str(request.state.get("original_question") or "")
                contract_error = _validate_request_contract(
                    question=question,
                    args=args,
                    previous_plan=(
                        request.state.get("analysis_plan")
                        if tool_name == "revise_plan"
                        else None
                    ),
                )
                if contract_error is not None:
                    error_code = _planning_error_code(contract_error)
                    formatted_error = _format_contract_error(
                        question=question,
                        contract_error=contract_error,
                        error_code=error_code,
                    )
                    repeated_count = (
                        _planning_error_count(
                            request.state.get("messages", []),
                            tool_name=tool_name,
                            code=error_code,
                        )
                        + 1
                    )
                    if (
                        error_code in IMMEDIATE_BLOCK_PLANNING_ERRORS
                        or repeated_count >= MAX_REPEATED_PLANNING_ERRORS
                    ):
                        return self._short_circuit_tool_call(
                            tool_call_id=tool_call_id,
                            result=_planning_block_command(
                                tool_name=tool_name,
                                tool_call_id=tool_call_id,
                                code=error_code,
                                contract_error=contract_error,
                                formatted_error=formatted_error,
                                repeated_count=repeated_count,
                            ),
                        )
                    return self._short_circuit_tool_call(
                        tool_call_id=tool_call_id,
                        result=ToolMessage(
                            content=formatted_error,
                            name=tool_name,
                            tool_call_id=tool_call_id,
                            status="error",
                        ),
                    )
        result = handler(request)
        if tool_name != "answer" or not isinstance(result, Command):
            return result

        update = result.update
        answer = update.get("answer") if isinstance(update, dict) else None
        plan = request.state.get("analysis_plan")
        confidence = _plan_intent_confidence(plan)
        if confidence < MIN_FINAL_INTENT_CONFIDENCE:
            return self._short_circuit_tool_call(
                tool_call_id=tool_call_id,
                result=_answer_error(
                    tool_call_id,
                    "intent confidence is below the submission threshold "
                    f"({confidence:.2f} < {MIN_FINAL_INTENT_CONFIDENCE:.2f}); inspect "
                    "schema/data and call revise_plan with evidence from the data and "
                    "a reassessment of the complete original question.",
                ),
            )
        readiness_error = _plan_submission_readiness_error(plan)
        if readiness_error is not None:
            return self._short_circuit_tool_call(
                tool_call_id=tool_call_id,
                result=_answer_error(
                    tool_call_id,
                    f"{readiness_error} Do not submit a guessed or substitute result; "
                    "call revise_plan with observed evidence, then write_todos and "
                    "execute the revised plan.",
                ),
            )
        expected_columns = (
            _normalize_string_list(plan.get("output_columns"))
            if isinstance(plan, dict)
            else []
        )
        if isinstance(answer, AnswerTable) and expected_columns:
            actual_names = [_normalized_field_name(column) for column in answer.columns]
            expected_names = [_normalized_field_name(column) for column in expected_columns]
            if actual_names != expected_names:
                return self._short_circuit_tool_call(
                    tool_call_id=tool_call_id,
                    result=_answer_error(
                        tool_call_id,
                        "answer columns do not match the latest plan: "
                        f"expected {expected_columns}, received {answer.columns}.",
                    ),
                )
        return result

# 闄愬埗Python鎵ц鐜鍙橀噺锛岀‘淇漊TF-8缂栫爜锛屽苟鏋勫缓瀹夊叏鐨勭幆澧冧緵瀛愯繘绋嬩娇鐢ㄣ€?
# 鍙厑璁竝ython鑴氭湰鎵ц鐩稿叧鐨勭幆澧冨彉閲忥紝閬垮厤娉勯湶鏁忔劅淇℃伅鎴栧奖鍝嶇郴缁熷畨鍏ㄣ€?
def _build_shell_environment() -> dict[str, str]:
    allowed_names = {
        "COMSPEC", #
        "LANG", #
        "LC_ALL",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP", #
        "TMPDIR",
        "WINDIR",
    }
    environment = {
        name: value for name, value in os.environ.items() if name.upper() in allowed_names
    }
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _format_process_output(
    stdout: bytes,
    stderr: bytes,
    *,
    exit_code: int | None,
    max_output_bytes: int,
) -> str:
    sections: list[bytes] = []
    if stdout:
        sections.append(b"[stdout]\n" + stdout)
    if stderr:
        sections.append(b"[stderr]\n" + stderr)
    output = b"\n".join(sections) or b"<no output>"
    truncated = len(output) > max_output_bytes
    output = output[:max_output_bytes]
    text = output.decode("utf-8", errors="replace")
    if truncated:
        text += f"\n[output truncated at {max_output_bytes} bytes]"
    if exit_code is not None:
        text += f"\n\nExit code: {exit_code}"
    return text


class _VirtualPathRewriter(ast.NodeTransformer):
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not isinstance(node.value, str):
            return node
        mapped = self._map_path(node.value)
        if mapped == node.value:
            return node
        return ast.copy_location(ast.Constant(value=mapped), node)

    def _map_path(self, value: str) -> str:
        for virtual_root in ("context", "scratch"):
            prefix = f"/{virtual_root}"
            if value != prefix and not value.startswith(f"{prefix}/"):
                continue
            relative_path = value.removeprefix(prefix).lstrip("/")
            root = (self.workspace / virtual_root).resolve()
            mapped = (root / Path(relative_path)).resolve()
            if not mapped.is_relative_to(root):
                raise ValueError(f"Virtual path escapes /{virtual_root}: {value}")
            return str(mapped)
        return value


def _rewrite_virtual_python_paths(code: str, workspace: Path) -> str:
    tree = ast.parse(code, filename="<execute_python>", mode="exec")
    rewritten = _VirtualPathRewriter(workspace).visit(tree)
    ast.fix_missing_locations(rewritten)
    return ast.unparse(rewritten)


def _create_execute_python_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    @tool("execute_python")
    def execute_python(
        code: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> ToolMessage:
        """Execute Python source directly without a shell or persistent script file."""
        if not code.strip():
            return ToolMessage(
                content="code must be a non-empty Python source string.",
                name="execute_python",
                tool_call_id=tool_call_id,
                status="error",
            )

        try:
            executable_code = _rewrite_virtual_python_paths(code, workspace)
        except (SyntaxError, ValueError) as exc:
            return ToolMessage(
                content=f"Invalid Python source or virtual path: {exc}",
                name="execute_python",
                tool_call_id=tool_call_id,
                status="error",
            )

        try:
            completed = subprocess.run(
                [sys.executable, "-X", "utf8", "-I", "-B", "-"],
                cwd=workspace,
                env=_build_shell_environment(),
                input=executable_code.encode("utf-8"),
                capture_output=True,
                check=False,
                timeout=config.execute_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            content = _format_process_output(
                exc.stdout or b"",
                exc.stderr or b"",
                exit_code=None,
                max_output_bytes=config.max_output_bytes,
            )
            return ToolMessage(
                content=(
                    f"{content}\n\nPython script timed out after "
                    f"{config.execute_timeout_seconds} seconds."
                ),
                name="execute_python",
                tool_call_id=tool_call_id,
                status="error",
            )
        except OSError as exc:
            return ToolMessage(
                content=f"Failed to start Python: {exc}",
                name="execute_python",
                tool_call_id=tool_call_id,
                status="error",
            )

        content = _format_process_output(
            completed.stdout,
            completed.stderr,
            exit_code=completed.returncode,
            max_output_bytes=config.max_output_bytes,
        )
        return ToolMessage(
            content=content,
            name="execute_python",
            tool_call_id=tool_call_id,
            status="success" if completed.returncode == 0 else "error",
        )

    return execute_python


def _tool_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("function", {}).get("name") or "")
    return str(getattr(value, "name", ""))


class HideUnavailableToolsMiddleware(AgentMiddleware[Any, None, Any]):
    hidden_tools = frozenset({"execute", "ls", "write_file", "edit_file"})

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        filtered_tools = [
            item for item in request.tools if _tool_name(item) not in self.hidden_tools
        ]
        return handler(request.override(tools=filtered_tools))


def _workspace_permissions() -> list[FilesystemPermission]:
    context_paths = ["/context/**"]
    return [
        FilesystemPermission(operations=["read"], paths=context_paths, mode="allow"),
        FilesystemPermission(operations=["write"], paths=context_paths, mode="deny"),
    ]


def _context_inventory(context_dir: Path) -> str:
    entries: list[str] = []
    for path in sorted(context_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(context_dir).as_posix()
        entries.append(f"- /context/{relative_path} ({path.stat().st_size} bytes)")
    return "\n".join(entries) if entries else "- <no context files>"


def _task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n\n"
        "Complete recursive context inventory:\n"
        f"{_context_inventory(task.context_dir)}\n\n"
        f"{format_knowledge_schema_prompt(task.context_dir)}\n\n"
        "Begin by identifying the requested output and forming a compact plan. Use the "
        "`general-purpose` subagent for complex or independent analysis steps. Do not list "
        "directories again. Use the knowledge schema as the semantic reference when it is "
        "usable. If a schema-defined table or field required by the current plan is absent "
        "from /context/, treat that as a blocking conflict: call `revise_plan` to record "
        "the missing schema source, lower confidence, and stop rather than substituting a "
        "different table, file, field, unit, or calculation with a different semantic "
        "definition. When the schema is missing or empty, inspect the relevant data and "
        "keep the final answer internally consistent with /context/. "
        "After each key discovery, compare the current plan with the original question; if "
        "the data shows the plan is over-aggregated, under-specified, or otherwise misaligned, "
        "call `revise_plan`, then call `write_todos` again to align execution with the revised "
        "plan. Validate the latest plan and submit the final table with `answer`."
    )


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _serialize_message(message: BaseMessage) -> dict[str, Any]:
    return _json_safe(message.model_dump(mode="json", exclude_none=True))


def _serialize_tool_result(value: ToolMessage | Command[Any]) -> dict[str, Any]:
    if isinstance(value, ToolMessage):
        return _serialize_message(value)
    update = value.update
    if isinstance(update, dict):
        serialized_update = dict(update)
        messages = serialized_update.get("messages")
        if isinstance(messages, list):
            serialized_update["messages"] = [
                _serialize_message(message)
                if isinstance(message, BaseMessage)
                else _json_safe(message)
                for message in messages
            ]
    else:
        serialized_update = _json_safe(update)
    return {
        "type": "command",
        "graph": value.graph,
        "update": _json_safe(serialized_update),
        "goto": _json_safe(value.goto),
    }


def _tool_result_ok(value: ToolMessage | Command[Any]) -> bool:
    if isinstance(value, ToolMessage):
        return getattr(value, "status", "success") != "error"
    update = value.update
    if not isinstance(update, dict):
        return True
    messages = update.get("messages")
    if not isinstance(messages, list):
        return True
    return not any(
        isinstance(message, ToolMessage)
        and getattr(message, "status", "success") == "error"
        for message in messages
    )


@dataclass(slots=True)
class TraceEvent:
    sequence: int
    action: str
    scope: str
    llm_call_index: int | None
    thought: str
    action_input: dict[str, Any]
    tool_call_id: str | None
    raw_response: dict[str, Any]
    observation: dict[str, Any]
    ok: bool


class TraceEventCollector:
    def __init__(self) -> None:
        self._lock = Lock()
        self._events: list[TraceEvent] = []
        self._next_sequence = 1
        self._next_llm_call_index = 1
        self._initialized_scopes: set[str] = set()
        self._tool_call_llm_indexes: dict[str, int] = {}
        self._llm_events: dict[int, TraceEvent] = {}

    def next_llm_call_index(self) -> int:
        with self._lock:
            call_index = self._next_llm_call_index
            self._next_llm_call_index += 1
            return call_index

    def initialize_scope(
        self,
        *,
        scope: str,
        system_message: BaseMessage | None,
        messages: list[BaseMessage],
        llm_call_index: int,
    ) -> None:
        with self._lock:
            if scope in self._initialized_scopes:
                return
            self._initialized_scopes.add(scope)
        if system_message is not None:
            self.add(
                action="system_prompt",
                scope=scope,
                llm_call_index=llm_call_index,
                action_input={"message": _serialize_message(system_message)},
            )
        for message in messages:
            if isinstance(message, HumanMessage):
                self.add(
                    action="user_prompt",
                    scope=scope,
                    llm_call_index=llm_call_index,
                    action_input={"message": _serialize_message(message)},
                )
                break

    def add(
        self,
        *,
        action: str,
        scope: str,
        llm_call_index: int | None = None,
        thought: str = "",
        action_input: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
        raw_response: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        ok: bool = True,
    ) -> None:
        with self._lock:
            event = TraceEvent(
                sequence=self._next_sequence,
                action=action,
                scope=scope,
                llm_call_index=llm_call_index,
                thought=thought,
                action_input=action_input or {},
                tool_call_id=tool_call_id,
                raw_response=raw_response or {},
                observation=observation or {},
                ok=ok,
            )
            self._next_sequence += 1
            self._events.append(event)

    def register_tool_calls(
        self,
        messages: list[BaseMessage],
        *,
        llm_call_index: int,
    ) -> None:
        with self._lock:
            for message in messages:
                if not isinstance(message, AIMessage):
                    continue
                for tool_call in message.tool_calls:
                    tool_call_id = str(tool_call.get("id") or "")
                    if tool_call_id:
                        self._tool_call_llm_indexes[tool_call_id] = llm_call_index

    def start_llm_call(
        self,
        *,
        scope: str,
        llm_call_index: int,
        request_summary: dict[str, Any],
    ) -> None:
        with self._lock:
            event = TraceEvent(
                sequence=self._next_sequence,
                action="llm_pending",
                scope=scope,
                llm_call_index=llm_call_index,
                thought="",
                action_input={"llm": request_summary},
                tool_call_id=None,
                raw_response={},
                observation={"status": "requesting", "tool_calls": []},
                ok=True,
            )
            self._next_sequence += 1
            self._events.append(event)
            self._llm_events[llm_call_index] = event

    def complete_llm_call(
        self,
        *,
        llm_call_index: int,
        response_payload: dict[str, Any],
        messages: list[BaseMessage],
        visible_text: str,
    ) -> None:
        tool_calls: list[dict[str, Any]] = []
        with self._lock:
            event = self._llm_events[llm_call_index]
            event.thought = visible_text
            event.raw_response = response_payload
            tool_names: list[str] = []
            for message in messages:
                if not isinstance(message, AIMessage):
                    continue
                for tool_call in message.tool_calls:
                    tool_name = str(tool_call.get("name") or "")
                    if tool_name:
                        tool_names.append(tool_name)
                    tool_call_id = str(tool_call.get("id") or "")
                    if tool_call_id:
                        self._tool_call_llm_indexes[tool_call_id] = llm_call_index
                    tool_calls.append(
                        {
                            "name": tool_name,
                            "tool_call_id": tool_call_id or None,
                            "args": _json_safe(tool_call.get("args")),
                            "status": "pending",
                            "ok": None,
                            "result": None,
                        }
                    )
            event.action = "+".join(tool_names) if tool_names else "llm_response"
            event.observation = {
                "status": "tools_pending" if tool_calls else "completed",
                "tool_calls": tool_calls,
            }

    def fail_llm_call(
        self,
        *,
        llm_call_index: int,
        error: BaseException,
    ) -> None:
        with self._lock:
            event = self._llm_events[llm_call_index]
            event.ok = False
            event.observation = {
                "status": "error",
                "error": str(error),
                "type": type(error).__name__,
                "tool_calls": [],
            }

    def complete_tool_call(
        self,
        *,
        tool_call_id: str,
        result: dict[str, Any],
        ok: bool,
    ) -> None:
        with self._lock:
            llm_call_index = self._tool_call_llm_indexes.get(tool_call_id)
            if llm_call_index is None:
                return
            event = self._llm_events[llm_call_index]
            tool_calls = event.observation.get("tool_calls", [])
            for tool_call in tool_calls:
                if tool_call.get("tool_call_id") != tool_call_id:
                    continue
                tool_call["status"] = "success" if ok else "error"
                tool_call["ok"] = ok
                tool_call["result"] = result
                break
            statuses = [item.get("status") for item in tool_calls]
            if statuses and all(status != "pending" for status in statuses):
                event.observation["status"] = "completed"
            event.ok = all(item.get("ok") is not False for item in tool_calls)

    def snapshot(self) -> list[StepRecord]:
        with self._lock:
            events = list(self._events)
        return [
            StepRecord(
                step_index=index,
                thought=event.thought,
                action=event.action,
                action_input={
                    "scope": event.scope,
                    "llm_call_index": event.llm_call_index,
                    **event.action_input,
                },
                tool_call_id=event.tool_call_id,
                raw_response=event.raw_response,
                observation=event.observation,
                ok=event.ok,
            )
            for index, event in enumerate(events, start=1)
        ]


class LlmTraceMiddleware(AgentMiddleware[Any, None, Any]):
    def __init__(self, collector: TraceEventCollector, *, scope: str) -> None:
        self.collector = collector
        self.scope = scope

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        call_index = self.collector.next_llm_call_index()
        self.collector.initialize_scope(
            scope=self.scope,
            system_message=request.system_message,
            messages=request.messages,
            llm_call_index=call_index,
        )
        request_summary = {
            "model": {
                "class": type(request.model).__name__,
                "identifier": str(
                    getattr(request.model, "model_name", None)
                    or getattr(request.model, "model", None)
                    or ""
                ),
            },
            "message_count": len(request.messages),
            "has_system_message": request.system_message is not None,
            "tool_names": [_tool_name(item) for item in request.tools],
            "tool_choice": _json_safe(request.tool_choice),
        }
        self.collector.start_llm_call(
            scope=self.scope,
            llm_call_index=call_index,
            request_summary=request_summary,
        )
        try:
            response = handler(request)
        except BaseException as exc:  # noqa: BLE001
            self.collector.fail_llm_call(
                llm_call_index=call_index,
                error=exc,
            )
            raise

        serialized_result = [_serialize_message(message) for message in response.result]
        visible_text = "\n".join(
            _visible_text(message)
            for message in response.result
            if isinstance(message, AIMessage) and _visible_text(message)
        )
        self.collector.complete_llm_call(
            llm_call_index=call_index,
            messages=response.result,
            visible_text=visible_text,
            response_payload={
                "messages": serialized_result,
                "structured_response": _json_safe(response.structured_response),
            },
        )
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_call_id = str(request.tool_call.get("id") or "")
        try:
            result = handler(request)
        except BaseException as exc:  # noqa: BLE001
            self.collector.complete_tool_call(
                tool_call_id=tool_call_id,
                result={"error": str(exc), "type": type(exc).__name__},
                ok=False,
            )
            raise

        serialized_result = _serialize_tool_result(result)
        ok = _tool_result_ok(result)
        self.collector.complete_tool_call(
            tool_call_id=tool_call_id,
            result=serialized_result,
            ok=ok,
        )
        return result


def _visible_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    text_parts: list[str] = []
    for block in message.content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in {"text", "output_text"} and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
    return "\n".join(text_parts)


def _normalize_answer(value: Any) -> AnswerTable | None:
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
    return [message for message in state.get("messages", []) if isinstance(message, BaseMessage)]


def _partial_run_result(
    task_id: str,
    state: dict[str, Any],
    collector: TraceEventCollector,
) -> AgentRunResult:
    return AgentRunResult(
        task_id=task_id,
        answer=_normalize_answer(state.get("answer")),
        steps=collector.snapshot(),
        failure_reason=None,
    )


class DeepAgent:
    def __init__(
        self,
        *,
        model: BaseChatModel,
        config: DeepAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.config = config or DeepAgentConfig()
        self.system_prompt = system_prompt or DEEP_AGENT_SYSTEM_PROMPT

    def _create_graph(
        self,
        workspace: Path,
        collector: TraceEventCollector,
    ):
        backend = FilesystemBackend(
            root_dir=workspace,
            virtual_mode=True,
        )
        execute_python_tool = _create_execute_python_tool(workspace, self.config)
        answer_tool = _create_answer_tool(workspace)
        return create_deep_agent(
            model=self.model,
            tools=[execute_python_tool],
            system_prompt=self.system_prompt,
            backend=backend,
            permissions=_workspace_permissions(),
            subagents=[
                {
                    "name": "general-purpose",
                    "description": (
                        "Handle a focused, complex data-analysis or verification subtask. "
                        "Returns findings with source files, calculation rules, assumptions, "
                        "and unresolved issues; it cannot submit the final answer."
                    ),
                    "system_prompt": SUBAGENT_SYSTEM_PROMPT,
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
                PlanningMiddleware(collector),
                AnswerMiddleware(answer_tool, workspace, collector),
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
        with tempfile.TemporaryDirectory(prefix=f"dabench-{task.task_id}-") as temp_dir:
            workspace = Path(temp_dir)
            shutil.copytree(task.context_dir, workspace / "context")
            (workspace / "scratch").mkdir()
            graph = self._create_graph(workspace, collector)

            try:
                for state in graph.stream(
                    {
                        "messages": [HumanMessage(content=_task_prompt(task))],
                        "original_question": task.question,
                    },
                    config={"recursion_limit": max(100, self.config.max_steps * 8)},
                    stream_mode="values",
                ):
                    if not isinstance(state, dict):
                        continue
                    result = state
                    if trace_callback is None:
                        continue

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
            planning_blocked = str(result.get("planning_blocked") or "").strip()
            if planning_blocked:
                failure_reason = planning_blocked
            else:
                model_call_count = int(result.get("run_model_call_count", 0))
                limit_reached = any(
                    isinstance(message, AIMessage)
                    and _visible_text(message).startswith("Model call limits exceeded:")
                    for message in messages
                )
                if limit_reached or model_call_count >= self.config.max_steps:
                    failure_reason = (
                        f"Agent did not submit an answer within {self.config.max_steps} model calls."
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
