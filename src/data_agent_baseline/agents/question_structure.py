from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

QUESTION_STRUCTURE_SCHEMA: dict[str, Any] = {
    "schema_version": "1.0",
    "original_question": "string",
    "targets": [
        {
            "quote": "exact substring from original_question",
            "name": "normalized target name",
            "target_type": "measure | entity | record_set | unknown",
            "description": "what the user wants returned",
        }
    ],
    "target_constraints": [
        {
            "quote": "exact substring from original_question",
            "constraint_type": (
                "scope | filter | time_range | geography | entity | grouping | "
                "ordering | limit | output_shape"
            ),
            "value": "normalized value",
            "explicitness": "explicit | ambiguous",
        }
    ],
    "conditions": {
        "filters": [],
        "time_ranges": [],
        "groupings": [],
        "orderings": [],
        "limits": [],
        "calculations": [],
        "output_columns": [],
    },
    "intent_operators": [
        {
            "quote": "exact substring from original_question",
            "operation": "aggregate | sort | limit | derive",
            "operator_type": "distribution | average | selector | calculation",
        }
    ],
    "output": {
        "row_grain_hint": "source_records | aggregated_records | unspecified",
        "requested_columns": [],
        "preserve_source_rows": "true | false | unknown",
    },
    "ambiguities": ["string"],
}

_SYSTEM_PROMPT = f"""
You are an isolated Question Structuring node for a data benchmark agent.

Use only the original user question. Do not use dataset knowledge, file names,
schemas, prior messages, or domain facts. Your job is to produce a conservative
structured representation of the user's wording.

Rules:
- Every quote must be an exact substring of original_question.
- Put only explicit wording into target_constraints and conditions.
- Every condition item must be an object with quote, value, condition_type, and
  explicitness. If you cannot cite an exact substring, set quote to null and
  keep the wording only in value.
- If wording may imply something but does not explicitly request it, put that in
  ambiguities instead of conditions.
- Do not invent aggregation, filtering, sorting, row grain, or output columns.
- Return JSON only, with no Markdown.

Schema:
{json.dumps(QUESTION_STRUCTURE_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text"))
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(parts)
    return str(content)


def _fallback_structure(question: str, *, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "original_question": question,
        "targets": [
            {
                "quote": question,
                "name": question,
                "target_type": "unknown",
                "description": "Interpret the original question conservatively.",
            }
        ],
        "target_constraints": [],
        "conditions": {
            "filters": [],
            "time_ranges": [],
            "groupings": [],
            "orderings": [],
            "limits": [],
            "calculations": [],
            "output_columns": [],
        },
        "output": {
            "row_grain_hint": "unspecified",
            "requested_columns": [],
            "preserve_source_rows": "unknown",
        },
        "intent_operators": [],
        "ambiguities": [reason],
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_PATTERN.search(text)
        if match is None:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("question structure response must be a JSON object")
    return value


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


_CONDITION_TYPE_BY_KEY = {
    "filters": "filter",
    "time_ranges": "time_range",
    "groupings": "grouping",
    "orderings": "ordering",
    "limits": "limit",
    "calculations": "calculation",
    "output_columns": "output_column",
}

_OPERATIONS_BY_REQUIREMENT_TYPE = {
    "entity": {"filter"},
    "filter": {"filter"},
    "scope": {"filter"},
    "time_range": {"filter"},
    "value": {"filter"},
    "grouping": set(),
    "ordering": {"sort"},
    "selector": {"limit", "sort"},
    "limit": {"limit"},
    "calculation": {"aggregate", "derive"},
    "deduplication": {"deduplicate"},
    "reshape": {"reshape"},
}

_REQUIREMENT_TYPE_BY_TARGET_CONSTRAINT = {
    "entity": "entity",
    "filter": "filter",
    "geography": "scope",
    "scope": "scope",
    "source_scope": "scope",
    "table_scope": "scope",
    "time_range": "time_range",
    "value": "value",
    "grouping": "grouping",
    "ordering": "ordering",
    "limit": "limit",
    "selector": "selector",
    "output_shape": "reshape",
    "calculation": "calculation",
    "aggregate_min": "calculation",
    "aggregate_max": "calculation",
    "aggregate_sum": "calculation",
    "aggregate_avg": "calculation",
    "aggregate_average": "calculation",
    "aggregate_count": "calculation",
}

def _exact_substring(value: Any, question: str) -> str | None:
    text = str(value or "").strip()
    if not text or text not in question:
        return None
    if re.fullmatch(r"[A-Za-z0-9]+", text):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_-]){re.escape(text)}(?![A-Za-z0-9_-])"
        )
        if pattern.search(question) is None:
            return None
    return text


def _condition_value(item: Any) -> str:
    if isinstance(item, Mapping):
        for key in ("value", "description", "statement", "quote"):
            text = str(item.get(key) or "").strip()
            if text:
                return text
        return ""
    return str(item or "").strip()


def _normalize_condition_item(item: Any, *, key: str, question: str) -> dict[str, Any]:
    condition_type = _CONDITION_TYPE_BY_KEY.get(key, key)
    if isinstance(item, Mapping):
        quote = _exact_substring(item.get("quote"), question)
        value = _condition_value(item)
        explicitness = str(item.get("explicitness") or "")
        if explicitness not in {"explicit", "ambiguous"}:
            explicitness = "explicit" if quote else "unquoted_hint"
        return {
            "quote": quote,
            "value": value,
            "condition_type": str(item.get("condition_type") or condition_type),
            "explicitness": explicitness,
        }
    return {
        "quote": None,
        "value": str(item or "").strip(),
        "condition_type": condition_type,
        "explicitness": "unquoted_hint",
    }


def _condition_explicit_quote(item: Mapping[str, Any], question: str) -> str | None:
    quote = _exact_substring(item.get("quote"), question)
    explicitness = str(item.get("explicitness") or "")
    if not quote or explicitness not in {"", "explicit"}:
        return None
    return quote


def _constraint_requirement_type(item: Mapping[str, Any]) -> str | None:
    constraint_type = str(item.get("constraint_type") or "")
    return _REQUIREMENT_TYPE_BY_TARGET_CONSTRAINT.get(constraint_type)


def _authorized_operations_by_quote(structure: Mapping[str, Any], question: str) -> dict[str, set[str]]:
    operations_by_quote: dict[str, set[str]] = {}
    conditions = _dict_value(structure.get("conditions"))
    for container_name, requirement_type in _CONDITION_TYPE_BY_KEY.items():
        operations = _OPERATIONS_BY_REQUIREMENT_TYPE.get(requirement_type, set())
        if not operations:
            continue
        for item in _list_value(conditions.get(container_name)):
            if not isinstance(item, Mapping):
                continue
            quote = _condition_explicit_quote(item, question)
            if quote:
                operations_by_quote.setdefault(quote, set()).update(operations)

    for item in _list_value(structure.get("target_constraints")):
        if not isinstance(item, Mapping):
            continue
        quote = _exact_substring(item.get("quote"), question)
        if not quote or str(item.get("explicitness") or "") != "explicit":
            continue
        requirement_type = _constraint_requirement_type(item)
        if requirement_type is None:
            continue
        operations_by_quote.setdefault(quote, set()).update(
            _OPERATIONS_BY_REQUIREMENT_TYPE.get(requirement_type, set())
        )
    return operations_by_quote


def _target_requests_source_records(targets: list[Any], question: str) -> bool:
    del question
    if any(
        isinstance(target, Mapping)
        and str(target.get("target_type") or "") == "record_set"
        for target in targets
    ):
        return True
    return False


def _ensure_exact_quotes(items: list[Any], question: str) -> list[Any]:
    normalized_items: list[Any] = []
    for item in items:
        if not isinstance(item, Mapping):
            normalized_items.append(item)
            continue
        item_dict = dict(item)
        quote = item_dict.get("quote")
        if quote and not _exact_substring(quote, question):
            item_dict["quote"] = None
        normalized_items.append(item_dict)
    return normalized_items


def _normalize_structure(raw: dict[str, Any], question: str) -> dict[str, Any]:
    normalized = {
        "schema_version": str(raw.get("schema_version") or "1.0"),
        "original_question": str(raw.get("original_question") or question),
        "targets": _ensure_exact_quotes(_list_value(raw.get("targets")), question),
        "target_constraints": _ensure_exact_quotes(
            _list_value(raw.get("target_constraints")),
            question,
        ),
        "conditions": _dict_value(raw.get("conditions")),
        "intent_operators": _list_value(raw.get("intent_operators")),
        "output": _dict_value(raw.get("output")),
        "ambiguities": _list_value(raw.get("ambiguities")),
    }
    if not normalized["targets"]:
        normalized["targets"] = _fallback_structure(
            question,
            reason="No target was extracted by the question structuring node.",
        )["targets"]

    conditions = normalized["conditions"]
    for key in [
        "filters",
        "time_ranges",
        "groupings",
        "orderings",
        "limits",
        "calculations",
        "output_columns",
    ]:
        conditions[key] = [
            _normalize_condition_item(item, key=key, question=question)
            for item in _list_value(conditions.get(key))
            if isinstance(item, Mapping) or str(item or "").strip()
        ]

    valid_operations = {"aggregate", "derive", "filter", "limit", "sort"}
    authorized_operations = _authorized_operations_by_quote(normalized, question)
    existing_operators = [
        item
        for item in normalized["intent_operators"]
        if isinstance(item, Mapping)
        and _exact_substring(item.get("quote"), question)
        and str(item.get("operation") or "").strip()
        and str(item.get("operation") or "") in valid_operations
        and (
            str(item.get("operation") or "")
            in authorized_operations.get(str(item.get("quote") or "").strip(), set())
        )
    ]
    normalized["intent_operators"] = [dict(item) for item in existing_operators]

    output = normalized["output"]
    output["row_grain_hint"] = str(output.get("row_grain_hint") or "unspecified")
    output["requested_columns"] = _list_value(output.get("requested_columns"))
    output["preserve_source_rows"] = str(output.get("preserve_source_rows") or "unknown")
    if output["row_grain_hint"] not in {
        "source_records",
        "aggregated_records",
        "unspecified",
    }:
        output["row_grain_hint"] = "unspecified"
    if output["preserve_source_rows"] not in {"true", "false", "unknown"}:
        output["preserve_source_rows"] = "unknown"
    if (
        output["row_grain_hint"] == "aggregated_records"
        and not conditions["calculations"]
        and not conditions["groupings"]
    ):
        output["row_grain_hint"] = "source_records"
        output["preserve_source_rows"] = "true"
    if (
        _target_requests_source_records(normalized["targets"], question)
        and not conditions["calculations"]
        and not conditions["groupings"]
        and not conditions["orderings"]
        and not conditions["limits"]
    ):
        output["row_grain_hint"] = "source_records"
        output["preserve_source_rows"] = "true"
    return normalized


def format_question_structure(structure: dict[str, Any]) -> str:
    return json.dumps(structure, ensure_ascii=False, indent=2, sort_keys=True)


def structure_question(model: BaseChatModel, question: str) -> tuple[dict[str, Any], BaseMessage]:
    """Run an isolated question-only model call and return normalized JSON."""

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"original_question: {question}"),
    ]
    response = model.invoke(messages)
    if not isinstance(response, BaseMessage):
        response = AIMessage(content=str(response))
    payload = _extract_json_object(_message_text(response))
    return _normalize_structure(payload, question), response


def fallback_question_structure(question: str, reason: str) -> dict[str, Any]:
    return _fallback_structure(question, reason=reason)
