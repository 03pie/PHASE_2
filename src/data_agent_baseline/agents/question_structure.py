from __future__ import annotations

import json
import re
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


def _normalize_structure(raw: dict[str, Any], question: str) -> dict[str, Any]:
    normalized = {
        "schema_version": str(raw.get("schema_version") or "1.0"),
        "original_question": str(raw.get("original_question") or question),
        "targets": _list_value(raw.get("targets")),
        "target_constraints": _list_value(raw.get("target_constraints")),
        "conditions": _dict_value(raw.get("conditions")),
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
        conditions[key] = _list_value(conditions.get(key))

    output = normalized["output"]
    output["row_grain_hint"] = str(output.get("row_grain_hint") or "unspecified")
    output["requested_columns"] = _list_value(output.get("requested_columns"))
    output["preserve_source_rows"] = str(output.get("preserve_source_rows") or "unknown")
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
