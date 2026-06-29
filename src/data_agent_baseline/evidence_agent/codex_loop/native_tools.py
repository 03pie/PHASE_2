from __future__ import annotations

import json
import re
from typing import Any

from data_agent_baseline.evidence_agent.codex_loop.protocol import ModelAction, ToolOutputEnvelope


def _object_schema(
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
    additional_properties: bool = False,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": additional_properties,
    }


_ARG_SEGMENT_PATTERN = re.compile(
    r"<arg_key>\s*([^<\n\r>]+)\s*>?\s*(?:</arg_key>)?\s*(?:<arg_value>)?\s*([\s\S]*?)(?=</parameter|<arg_key>|$)",
    re.IGNORECASE,
)
_REF_PATTERN = re.compile(r"\b(?:src|ev|bind|comp|req|sec|cand|rel)_\d{4}\b")
_LIST_FIELDS = {
    "target_refs",
    "requirement_refs",
    "knowledge_section_ids",
    "section_ids",
    "tokens",
    "card_ids",
    "semantic_card_ids",
    "canonical_fields",
    "evidence_refs",
    "binding_refs",
    "compute_refs",
    "source_refs",
    "source_candidates",
    "target_fields",
    "slice_ids",
    "allowed_columns",
}
_INT_FIELDS = {
    "limit",
    "sample_limit",
    "max_pairs",
}
_BINDING_TYPE_ALIASES = {
    "source": "structured_source",
    "data_source": "structured_source",
    "database_source": "structured_source",
    "table_source": "structured_source",
    "relation": "structured_source",
    "structured_relation": "structured_source",
    "column": "structured_field",
    "field": "structured_field",
}


def _strip_tool_markup(value: str) -> str:
    text = str(value)
    if "</parameter" in text:
        text = text.split("</parameter", 1)[0]
    text = re.sub(r"</?arg_(?:key|value)>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^<arg_value>\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</arg_value>\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _coerce_tool_value(key: str, value: Any) -> Any:
    if isinstance(value, dict):
        return _normalize_tool_args("", value)
    if isinstance(value, list):
        items: list[Any] = []
        for item in value:
            coerced = _coerce_tool_value(key, item)
            if key in _LIST_FIELDS and isinstance(coerced, list):
                items.extend(coerced)
            else:
                items.append(coerced)
        return items
    if not isinstance(value, str):
        return value
    cleaned = _strip_tool_markup(value)
    if key in _INT_FIELDS:
        match = re.search(r"-?\d+", cleaned)
        if match:
            return int(match.group(0))
    if key == "binding_type":
        return _BINDING_TYPE_ALIASES.get(cleaned, cleaned)
    if key in {"source_ref", "source_id", "binding_ref", "compute_ref", "candidate_ref"}:
        match = _REF_PATTERN.search(cleaned)
        return match.group(0) if match else cleaned
    if key in _LIST_FIELDS:
        if cleaned.startswith("["):
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        refs = _REF_PATTERN.findall(cleaned)
        if refs:
            return refs
        if "," in cleaned:
            return [part.strip() for part in cleaned.split(",") if part.strip()]
    return cleaned


def _extract_spilled_arguments(value: str) -> dict[str, Any]:
    if "<arg_key" not in value:
        return {}
    normalized = value.replace("</parameter<arg_key", "</parameter><arg_key")
    spilled: dict[str, Any] = {}
    for match in _ARG_SEGMENT_PATTERN.finditer(normalized):
        raw_key = match.group(1)
        raw_value = match.group(2)
        key = re.sub(r"[^0-9A-Za-z_]+", "", raw_key).strip()
        if not key:
            continue
        spilled[key] = _coerce_tool_value(key, raw_value)
    return spilled


def _normalize_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    del tool_name
    normalized: dict[str, Any] = {}
    spilled: dict[str, Any] = {}
    for key, value in args.items():
        clean_key = str(key).strip()
        if isinstance(value, str):
            spilled.update(_extract_spilled_arguments(value))
        normalized[clean_key] = _coerce_tool_value(clean_key, value)
    for key, value in spilled.items():
        if key not in normalized or normalized[key] in {"", None, []}:
            normalized[key] = value
    if "requirements" in normalized and isinstance(normalized["requirements"], list):
        fixed_requirements: list[Any] = []
        for item in normalized["requirements"]:
            if isinstance(item, dict):
                fixed_requirements.append(_normalize_tool_args("track_requirements", item))
            else:
                fixed_requirements.append(item)
        normalized["requirements"] = fixed_requirements
    return normalized


MODEL_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_inventory",
            "description": "List observed context files and their data forms.",
            "parameters": _object_schema({}),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": (
                "Navigate knowledge.md without treating it as physical schema. Use mode='semantic' "
                "for semantic cards/source mappings, mode='catalog' to list sections/tokens, "
                "mode='token' to resolve mentions to complete sections, mode='section' to read "
                "complete section slices, or mode='search' for lexical candidates."
            ),
            "parameters": _object_schema(
                {
                    "mode": {
                        "type": "string",
                        "enum": ["semantic", "catalog", "search", "token", "section"],
                    },
                    "query": {"type": "string"},
                    "section_ids": {"type": "array", "items": {"type": "string"}},
                    "card_ids": {"type": "array", "items": {"type": "string"}},
                    "tokens": {"type": "array", "items": {"type": "string"}},
                    "include_neighbors": {"type": "boolean"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 80},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "locate_sources",
            "description": "Find candidate sources/tables/fields by lexical match over observed inventory.",
            "parameters": _object_schema(
                {
                    "query": {"type": "string"},
                    "tokens": {"type": "array", "items": {"type": "string"}},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_source",
            "description": "Inspect an observed SQLite/CSV/JSON source. Use source_ref or path.",
            "parameters": _object_schema(
                {
                    "source_ref": {"type": "string"},
                    "path": {"type": "string"},
                    "table": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sample_records",
            "description": "Read a bounded sample from an observed structured source.",
            "parameters": _object_schema(
                {
                    "source_ref": {"type": "string"},
                    "path": {"type": "string"},
                    "table": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_values",
            "description": "Search literal values across observed SQLite/CSV/JSON/PDF/MD sources.",
            "parameters": _object_schema(
                {
                    "query": {"type": "string"},
                    "value": {"type": "string"},
                    "source_ref": {"type": "string"},
                    "path": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_document_agent",
            "description": (
                "Delegate PDF/MD work to the bounded DocumentAgent sub-loop. It inspects "
                "record indexes, searches/reads record slices, performs local semantic extraction, "
                "checks coverage, and returns a compact DocEvidencePackage. It cannot compute or final-answer."
            ),
            "parameters": _object_schema(
                {
                    "question": {"type": "string"},
                    "target_fields": {"type": "array", "items": {"type": "string"}},
                    "semantic_cards": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                    "source_candidates": {"type": "array", "items": {"type": "string"}},
                    "required_record_grain": {"type": "string"},
                    "coverage_policy": {"type": "object", "additionalProperties": True},
                    "records": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                    "slice_decisions": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_relation",
            "description": (
                "Inspect verified compute relations after bind: relation_name, columns, "
                "types, row count, and sample. Use before SQL repair."
            ),
            "parameters": _object_schema(
                {
                    "binding_ref": {"type": "string"},
                    "relation_name": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "discover_join_paths",
            "description": (
                "Inspect verified relations for generic join candidates using observed column names "
                "and sample value overlap. This only returns evidence; it does not bind or execute a join."
            ),
            "parameters": _object_schema(
                {
                    "binding_refs": {"type": "array", "items": {"type": "string"}},
                    "sample_limit": {"type": "integer", "minimum": 1, "maximum": 5000},
                    "max_pairs": {"type": "integer", "minimum": 1, "maximum": 200},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_requirements",
            "description": (
                "Declare or update generic answer requirements derived from the question, "
                "knowledge text, or observed evidence. This is a coverage ledger only; "
                "it does not bind data or compute answers."
            ),
            "parameters": _object_schema(
                {
                    "requirements": {
                        "type": "array",
                        "items": _object_schema(
                            {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "satisfied",
                                        "not_applicable",
                                        "conflict",
                                        "blocked",
                                    ],
                                },
                                "source_refs": {"type": "array", "items": {"type": "string"}},
                                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                                "binding_refs": {"type": "array", "items": {"type": "string"}},
                                "compute_refs": {"type": "array", "items": {"type": "string"}},
                                "note": {"type": "string"},
                            },
                            required=["text"],
                        ),
                    }
                },
                required=["requirements"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_alignment",
            "description": (
                "Record a structured verifier decision about whether cited evidence, "
                "bindings, or compute results align with the question and knowledge. "
                "This tool never creates a binding or final answer by itself."
            ),
            "parameters": _object_schema(
                {
                    "decision": {
                        "type": "string",
                        "enum": [
                            "bindable",
                            "candidate_answer",
                            "intermediate",
                            "not_applicable",
                            "needs_more_evidence",
                            "conflict",
                            "blocked_ok",
                        ],
                    },
                    "target_kind": {
                        "type": "string",
                        "enum": [
                            "source",
                            "field",
                            "document_window",
                            "document_record_set",
                            "compute_result",
                            "direct_evidence",
                            "alternative_source",
                            "requirement",
                            "blocked",
                            "final_answer",
                        ],
                    },
                    "target_refs": {"type": "array", "items": {"type": "string"}},
                    "requirement_refs": {"type": "array", "items": {"type": "string"}},
                    "knowledge_section_ids": {"type": "array", "items": {"type": "string"}},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "binding_refs": {"type": "array", "items": {"type": "string"}},
                    "compute_refs": {"type": "array", "items": {"type": "string"}},
                    "alignment": {"type": "string"},
                    "limitations": {"type": "string"},
                    "next_actions": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": True},
                    },
                },
                required=["decision", "target_kind", "alignment"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bind",
            "description": (
                "Create a verified binding from successful observed evidence. "
                "Candidate-only or failed evidence cannot be bound."
            ),
            "parameters": _object_schema(
                {
                    "binding_type": {
                        "type": "string",
                        "enum": [
                            "structured_source",
                            "structured_field",
                            "document_window",
                            "document_record_set",
                            "value",
                            "operation",
                            "answer_candidate",
                        ],
                    },
                    "source_ref": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "table": {"type": "string"},
                    "field": {"type": "string"},
                    "allowed_columns": {"type": "array", "items": {"type": "string"}},
                    "semantic_card_ids": {"type": "array", "items": {"type": "string"}},
                    "canonical_fields": {"type": "array", "items": {"type": "string"}},
                    "physical_field_mapping": {"type": "object", "additionalProperties": True},
                    "semantic_contract": {"type": "object", "additionalProperties": True},
                    "alignment": {"type": "string"},
                    "answer": {"type": "object", "additionalProperties": True},
                },
                required=["binding_type", "evidence_refs", "alignment"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_verified_compute",
            "description": (
                "Run SQL over verified relation names from bindings only. "
                "Use relation_name values such as rel_0001, not original file or table names."
            ),
            "parameters": _object_schema(
                {
                    "sql": {"type": "string"},
                    "binding_refs": {"type": "array", "items": {"type": "string"}},
                },
                required=["sql", "binding_refs"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_final",
            "description": (
                "Submit the final answer. A compute-backed final requires a prior "
                "verify_alignment candidate_answer decision for the compute_ref and an explicit answer.columns projection. "
                "For direct document/value evidence, provide answer plus binding_refs and evidence_refs. "
                "The answer object may only project or alias values already present in the cited compute result."
            ),
            "parameters": _object_schema(
                {
                    "compute_ref": {"type": "string"},
                    "answer": {"type": "object", "additionalProperties": True},
                    "binding_refs": {"type": "array", "items": {"type": "string"}},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "alignment": {"type": "string"},
                },
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_video",
            "description": "Inspect video metadata. V1 returns unsupported and cannot support final evidence.",
            "parameters": _object_schema(
                {"source_ref": {"type": "string"}, "path": {"type": "string"}},
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_video_observations",
            "description": "Video extraction placeholder. V1 is unsupported.",
            "parameters": _object_schema(
                {"source_ref": {"type": "string"}, "path": {"type": "string"}},
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "blocked",
            "description": (
                "Stop when evidence is insufficient, conflicting, or no valid action remains. "
                "Cite evidence_refs when blocked is based on observed evidence or negative searches."
            ),
            "parameters": _object_schema(
                {
                    "reason": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                },
                required=["reason"],
            ),
        },
    },
]


def _tool_specs_for(tool_names: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    if not tool_names:
        return MODEL_TOOL_SPECS
    allowed = {name for name in tool_names if name}
    return [
        spec
        for spec in MODEL_TOOL_SPECS
        if spec.get("function", {}).get("name") in allowed
    ]


def bind_native_tools(
    model: Any,
    *,
    tool_choice: str | dict[str, Any] | None = None,
    tool_names: tuple[str, ...] | None = None,
) -> Any:
    if not hasattr(model, "bind_tools"):
        raise TypeError("Configured model does not support native tool calling via bind_tools().")
    kwargs: dict[str, Any] = {"parallel_tool_calls": False}
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    return model.bind_tools(_tool_specs_for(tool_names), **kwargs)


def extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    calls = getattr(response, "tool_calls", None)
    if calls:
        return [dict(call) for call in calls]
    additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
    raw_calls = additional_kwargs.get("tool_calls") or []
    normalized: list[dict[str, Any]] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
        name = raw.get("name") or function.get("name")
        args = raw.get("args")
        if args is None:
            arguments = function.get("arguments") or raw.get("arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    args = json.loads(arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw_arguments": arguments}
            elif isinstance(arguments, dict):
                args = arguments
        normalized.append(
            {
                "id": raw.get("id") or raw.get("call_id") or f"call_{len(normalized)+1:04d}",
                "name": name,
                "args": args or {},
            }
        )
    return normalized


def action_from_tool_call(call: dict[str, Any]) -> ModelAction:
    name = str(call.get("name") or "").strip()
    raw_args = call.get("args") if isinstance(call.get("args"), dict) else {}
    args = _normalize_tool_args(name, raw_args)
    if name == "bind":
        evidence_refs = args.get("evidence_refs")
        if isinstance(evidence_refs, str):
            evidence_refs = [evidence_refs]
        return ModelAction(
            kind="bind",
            reason=str(args.get("alignment") or ""),
            binding_type=str(args.get("binding_type") or "").strip() or None,
            evidence_refs=tuple(str(item) for item in (evidence_refs or []) if str(item).strip()),
            source_ref=str(args.get("source_ref") or "").strip() or None,
            arguments=dict(args),
        )
    if name == "run_verified_compute":
        binding_refs = args.get("binding_refs")
        if isinstance(binding_refs, str):
            binding_refs = [binding_refs]
        return ModelAction(
            kind="compute",
            reason=str(args.get("reason") or ""),
            sql=str(args.get("sql") or "").strip() or None,
            binding_refs=tuple(str(item) for item in (binding_refs or []) if str(item).strip()),
            arguments=dict(args),
        )
    if name == "submit_final":
        binding_refs = args.get("binding_refs")
        if isinstance(binding_refs, str):
            binding_refs = [binding_refs]
        evidence_refs = args.get("evidence_refs")
        if isinstance(evidence_refs, str):
            evidence_refs = [evidence_refs]
        return ModelAction(
            kind="final",
            reason=str(args.get("reason") or ""),
            compute_ref=str(args.get("compute_ref") or "").strip() or None,
            binding_refs=tuple(str(item) for item in (binding_refs or []) if str(item).strip()),
            evidence_refs=tuple(str(item) for item in (evidence_refs or []) if str(item).strip()),
            answer=args.get("answer") if isinstance(args.get("answer"), dict) else None,
            arguments=dict(args),
        )
    if name == "blocked":
        evidence_refs = args.get("evidence_refs")
        if isinstance(evidence_refs, str):
            evidence_refs = [evidence_refs]
        return ModelAction(
            kind="blocked",
            reason=str(args.get("reason") or "blocked"),
            evidence_refs=tuple(str(item) for item in (evidence_refs or []) if str(item).strip()),
            arguments=dict(args),
        )
    return ModelAction(kind="tool_call", tool_name=name, arguments=dict(args))


def tool_output_content(
    envelope: ToolOutputEnvelope | Any,
    *,
    state_summary: dict[str, Any],
) -> str:
    payload = {
        "tool_output_envelope": (
            envelope.to_dict() if hasattr(envelope, "to_dict") else envelope
        ),
        "state_summary": state_summary,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)
