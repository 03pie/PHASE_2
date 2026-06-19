from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Command

from data_agent_baseline.agents.deep_state import BenchmarkDeepAgentState
from data_agent_baseline.agents.semantic_layer import parse_knowledge_content
from data_agent_baseline.tools.agent_tools.analyze_plan import analyze_plan_tool
from data_agent_baseline.tools.answer import (
    normalize_answer_columns,
    validate_prepared_answer,
)

_SOURCE_DISCOVERY_TOOLS = frozenset(
    {
        "execute_python",
        "execute_sql",
        "extract_narrative_records",
        "grep_file",
        "inspect_sqlite",
        "query_schema",
        "read_csv",
        "read_doc",
        "read_json",
    }
)
_ANSWER_CANDIDATE_RECOVERY_TOOLS = frozenset(
    {
        "analyze_plan",
        "execute_python",
        "finalize_answer_candidate",
        "set_answer",
    }
)
_PRE_PLAN_EXECUTION_TOOLS = frozenset(
    {
        "execute_python",
        "execute_sql",
        "extract_narrative_records",
    }
)
_PRE_PLAN_EXECUTION_LIMIT = 3
_PRE_PLAN_DISCOVERY_LIMIT = 8
_MAX_TOOL_CALLS_PER_MODEL_STEP = 8
_FORCED_TOOL_CALL_LIMIT = 1
_CONTEXT_PATH_PATTERN = re.compile(r"""["'](/context/[^"']+)["']""")
_TOOL_ARGUMENT_MARKUP_PATTERN = re.compile(
    r"</?\s*parameter(?:\b|=)|</?\s*tool_call\b",
    re.IGNORECASE,
)
_INJECTED_KNOWLEDGE_PATTERN = re.compile(
    r"<context_knowledge>\s*(.*?)\s*</context_knowledge>",
    re.DOTALL,
)
_REVISION_FIELDS = (
    "intent",
    "output_spec",
    "evidence",
    "steps",
    "delegation_candidates",
    "execution_spec",
)
_TODO_STATUSES = frozenset({"pending", "in_progress", "completed"})
_KEY_OUTPUT_ROLES = frozenset({"entity_key", "record_key", "time_key"})
_NARRATIVE_SOURCE_SUFFIXES = frozenset(
    {".log", ".md", ".markdown", ".pdf", ".txt"}
)
_NARRATIVE_SOURCE_TYPES = frozenset({"doc", "document", "pdf", "text"})
_STRUCTURED_SOURCE_TYPES = frozenset(
    {"csv", "json", "sqlite", "sqlite_table", "table"}
)
_NARRATIVE_FIELD_EVIDENCE_TOOLS = frozenset({"extract_narrative_records"})
_OPERATION_REQUIREMENT_TYPES: dict[str, frozenset[str]] = {
    "filter": frozenset({"entity", "filter", "scope", "time_range", "value"}),
    "aggregate": frozenset({"calculation"}),
    "derive": frozenset({"calculation"}),
    "sort": frozenset({"ordering", "selector"}),
    "limit": frozenset({"limit", "selector"}),
    "deduplicate": frozenset({"deduplication"}),
    "reshape": frozenset({"reshape"}),
}
_REQUIREMENT_TYPE_OPERATIONS: dict[str, frozenset[str]] = {
    requirement_type: frozenset(
        operation
        for operation, requirement_types in _OPERATION_REQUIREMENT_TYPES.items()
        if requirement_type in requirement_types
    )
    for requirement_type in {
        item
        for requirement_types in _OPERATION_REQUIREMENT_TYPES.values()
        for item in requirement_types
    }
}
_SELECTOR_EXPRESSION_PATTERN = re.compile(
    (
        r"\b(?:argmax|argmin|max|min)\s*\(|"
        r"\b(?:maximum|minimum|highest|lowest|first|last|latest|earliest)\b|"
        r"\bmost[\s_-]+recent\b"
    ),
    re.IGNORECASE,
)
_TRANSFORM_OPERATIONS = frozenset(
    {
        "aggregate",
        "derive",
        "deduplicate",
        "filter",
        "limit",
        "reshape",
        "sort",
    }
)
DISABLED_BUILTIN_TOOLS = frozenset(
    {
        "edit_file",
        "execute",
        "glob",
        "grep",
        "ls",
        "read_file",
        "write_file",
    }
)


@dataclass(frozen=True, slots=True)
class _DiscoveryState:
    knowledge_present: bool
    knowledge_checked: bool
    knowledge_available: bool
    knowledge_content: str
    context_sources: frozenset[str]
    needs_cross_validation: bool

    @property
    def knowledge_ready(self) -> bool:
        return True

    @property
    def context_ready(self) -> bool:
        return bool(self.context_sources)

    def tool_policy(self) -> tuple[set[str], str | None]:
        """Choose visible tools and optional forced tool choice."""

        if not self.context_ready:
            return set(_SOURCE_DISCOVERY_TOOLS), None
        return set(_SOURCE_DISCOVERY_TOOLS) | {"analyze_plan"}, None


def tool_name(value: Any) -> str:
    """Return a tool name from LangChain tool objects or dict definitions."""

    if isinstance(value, dict):
        return str(value.get("name") or value.get("function", {}).get("name") or "")
    return str(getattr(value, "name", ""))


def _invalid_tool_name(response: ModelResponse[Any]) -> str | None:
    """Detect a response with only invalid, unparsable tool calls."""

    for message in response.result:
        if not isinstance(message, AIMessage):
            continue
        if message.invalid_tool_calls and not message.tool_calls:
            return str(message.invalid_tool_calls[0].get("name") or "")
    return None


def _retry_invalid_tool_call(
    request: ModelRequest[None],
    handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    response: ModelResponse[Any],
) -> ModelResponse[Any]:
    """Let the model retry once when tool-call JSON was malformed."""

    invalid_name = _invalid_tool_name(response)
    if invalid_name is None:
        return response

    available_tool_names = {tool_name(item) for item in request.tools}
    retry_request = request.override(
        messages=[
            *request.messages,
            HumanMessage(
                content=(
                    f"The previous `{invalid_name or 'tool'}` call could not be "
                    "parsed. Reissue exactly one tool call with valid JSON arguments "
                    "that fully match the provided schema. Keep the intended task "
                    "semantics unchanged and do not answer in plain text."
                )
            ),
        ],
        tool_choice=(
            invalid_name
            if invalid_name and invalid_name in available_tool_names
            else request.tool_choice
        ),
    )
    return handler(retry_request)


def _tool_call_name(tool_call: Mapping[str, Any]) -> str:
    return str(
        tool_call.get("name")
        or (
            tool_call.get("function", {}).get("name")
            if isinstance(tool_call.get("function"), Mapping)
            else ""
        )
        or ""
    )


def _constrain_model_tool_calls(
    response: ModelResponse[Any],
    *,
    allowed_tool_names: set[str] | frozenset[str],
    forced_tool_name: str | None = None,
    max_tool_calls: int = _MAX_TOOL_CALLS_PER_MODEL_STEP,
) -> ModelResponse[Any]:
    """Keep model tool calls inside the currently advertised tool contract."""

    if not allowed_tool_names:
        return response

    changed = False
    constrained_messages: list[BaseMessage] = []
    for message in response.result:
        if not isinstance(message, AIMessage) or not message.tool_calls:
            constrained_messages.append(message)
            continue

        original_calls = list(message.tool_calls)
        if forced_tool_name:
            matching_calls = [
                call
                for call in original_calls
                if _tool_call_name(call) == forced_tool_name
            ]
            if matching_calls:
                kept_calls = matching_calls[:_FORCED_TOOL_CALL_LIMIT]
            else:
                kept_calls = original_calls[:_FORCED_TOOL_CALL_LIMIT]
        else:
            kept_calls = [
                call
                for call in original_calls
                if _tool_call_name(call) in allowed_tool_names
            ]
            if not kept_calls and original_calls:
                kept_calls = original_calls[:1]
            kept_calls = kept_calls[:max_tool_calls]

        if len(kept_calls) != len(original_calls) or any(
            kept is not original
            for kept, original in zip(kept_calls, original_calls, strict=False)
        ):
            changed = True
            message = message.model_copy(update={"tool_calls": kept_calls})
        constrained_messages.append(message)

    if not changed:
        return response
    return ModelResponse(
        result=constrained_messages,
        structured_response=response.structured_response,
    )


def _response_has_tool_call(
    response: ModelResponse[Any],
    *,
    tool_name_to_find: str | None = None,
) -> bool:
    for message in response.result:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            if tool_name_to_find is None or _tool_call_name(call) == tool_name_to_find:
                return True
    return False


def _retry_missing_forced_tool_call(
    request: ModelRequest[None],
    handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    response: ModelResponse[Any],
    *,
    forced_tool_name: str | None,
) -> ModelResponse[Any]:
    if not forced_tool_name:
        return response
    if _response_has_tool_call(response, tool_name_to_find=forced_tool_name):
        return response
    retry_request = request.override(
        messages=[
            *request.messages,
            *response.result,
            HumanMessage(
                content=(
                    f"The current step requires exactly one `{forced_tool_name}` "
                    "tool call. Reissue the next message as a valid tool call "
                    "matching that schema; do not answer in plain text."
                )
            ),
        ],
        tool_choice=forced_tool_name,
    )
    return handler(retry_request)


def _tool_error(request: ToolCallRequest, content: str) -> ToolMessage:
    return ToolMessage(
        content=content,
        name=str(request.tool_call.get("name") or ""),
        tool_call_id=str(request.tool_call.get("id") or ""),
        status="error",
    )


def _tool_arguments_contain_protocol_markup(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_TOOL_ARGUMENT_MARKUP_PATTERN.search(value))
    if isinstance(value, Mapping):
        return any(_tool_arguments_contain_protocol_markup(item) for item in value.values())
    if isinstance(value, list):
        return any(_tool_arguments_contain_protocol_markup(item) for item in value)
    return False


def _decode_json_like_argument(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in {"{", "["}:
            try:
                return _decode_json_like_argument(json.loads(stripped))
            except json.JSONDecodeError:
                try:
                    decoded, end_index = json.JSONDecoder().raw_decode(stripped)
                except json.JSONDecodeError:
                    return value
                trailing = stripped[end_index:].strip()
                if trailing and set(trailing) - {"}", "]"}:
                    return value
                return _decode_json_like_argument(decoded)
            except TypeError:
                return value
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _decode_json_like_argument(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_decode_json_like_argument(item) for item in value]
    return value


def _tagged_block(text: str, tag: str) -> str | None:
    match = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", text, re.S)
    if match is None:
        return None
    return match.group(1).strip()


def _decode_tagged_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    decoded = _decode_json_like_argument(stripped)
    if decoded is not stripped:
        return decoded
    if stripped.casefold() == "null":
        return None
    if re.fullmatch(r"-?\d+", stripped):
        try:
            return int(stripped)
        except ValueError:
            return stripped
    return stripped


def _decode_tagged_object(text: str, fields: Mapping[str, Any]) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    for key, default in fields.items():
        block = _tagged_block(text, key)
        if block is None:
            if default is not None:
                payload[key] = default
            continue
        payload[key] = _decode_tagged_value(block)
    return payload or None


def _decode_leading_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, Mapping):
        return dict(payload)
    return None


def _recover_analyze_plan_tagged_arguments(
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    recovered = dict(arguments)
    structured_fields = {
        "intent",
        "output_spec",
        "evidence",
        "revision",
        "execution_spec",
    }
    for key in structured_fields:
        value = recovered.get(key)
        if not isinstance(value, str):
            continue
        leading_payload = _decode_leading_json_object(value)
        if leading_payload is not None:
            recovered[key] = leading_payload
    text_blobs = [
        value
        for value in arguments.values()
        if isinstance(value, str) and "<" in value and ">" in value
    ]
    if not text_blobs:
        return recovered
    for text in text_blobs:
        leading_payload = _decode_leading_json_object(text)
        if leading_payload is not None:
            for key in (
                "schema_version",
                "intent",
                "output_spec",
                "evidence",
                "revision",
                "steps",
                "delegation_candidates",
                "execution_spec",
            ):
                if key not in recovered and key in leading_payload:
                    recovered[key] = leading_payload[key]
            execution_spec = recovered.get("execution_spec")
            if not isinstance(execution_spec, Mapping) and any(
                key in leading_payload
                for key in (
                    "sources",
                    "supporting_fields",
                    "operations",
                    "source_bindings",
                )
            ):
                recovered["execution_spec"] = leading_payload
        if "intent" not in recovered:
            intent_text = _tagged_block(text, "intent")
            if intent_text is not None:
                decoded_intent = _decode_tagged_value(intent_text)
                if isinstance(decoded_intent, Mapping):
                    recovered["intent"] = dict(decoded_intent)
                else:
                    intent_payload = _decode_tagged_object(
                        intent_text,
                        {"requirements": [], "unresolved": []},
                    )
                    if intent_payload is not None:
                        recovered["intent"] = intent_payload
        if "output_spec" not in recovered:
            output_text = _tagged_block(text, "output_spec")
            if output_text is not None:
                decoded_output = _decode_tagged_value(output_text)
                if isinstance(decoded_output, Mapping):
                    recovered["output_spec"] = dict(decoded_output)
                else:
                    output_payload = _decode_tagged_object(
                        output_text,
                        {
                            "columns": [],
                            "row_grain": "",
                            "row_policy": "",
                            "transformations": [],
                            "ordering": "",
                            "sort_keys": [],
                            "null_policy": "",
                            "expected_row_count": None,
                        },
                    )
                    if output_payload is not None:
                        recovered["output_spec"] = output_payload
        if "revision" not in recovered:
            revision_text = _tagged_block(text, "revision")
            if revision_text is not None:
                decoded_revision = _decode_tagged_value(revision_text)
                if isinstance(decoded_revision, Mapping):
                    recovered["revision"] = dict(decoded_revision)
                else:
                    revision_payload = _decode_tagged_object(
                        revision_text,
                        {
                            "version": 1,
                            "reason": "",
                            "evidence_changes": [],
                            "changed_fields": [],
                        },
                    )
                    if revision_payload is not None:
                        recovered["revision"] = revision_payload
        if "steps" not in recovered:
            steps_text = _tagged_block(text, "steps")
            if steps_text is not None:
                recovered["steps"] = _decode_tagged_value(steps_text)
        if "schema_version" not in recovered:
            schema_text = _tagged_block(text, "schema_version")
            if schema_text is not None:
                recovered["schema_version"] = str(_decode_tagged_value(schema_text))

        execution_spec = recovered.get("execution_spec")
        if not isinstance(execution_spec, Mapping):
            execution_payload = _decode_tagged_object(
                text,
                {
                    "sources": [],
                    "supporting_fields": [],
                    "operations": [],
                    "source_bindings": [],
                },
            )
            if execution_payload is not None:
                recovered["execution_spec"] = execution_payload
    return recovered


def _normalize_tool_call_arguments(request: ToolCallRequest) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    normalized_arguments = _decode_json_like_argument(arguments)
    if not isinstance(normalized_arguments, Mapping):
        return request
    if str(request.tool_call.get("name") or "") == "analyze_plan":
        normalized_arguments = _recover_analyze_plan_tagged_arguments(
            normalized_arguments,
        )
        execution_spec = normalized_arguments.get("execution_spec")
        if execution_spec is not None and not isinstance(execution_spec, Mapping):
            normalized_arguments.pop("execution_spec", None)
        allowed_keys = {
            "schema_version",
            "intent",
            "output_spec",
            "evidence",
            "revision",
            "steps",
            "delegation_candidates",
            "execution_spec",
        }
        normalized_arguments = {
            key: value
            for key, value in normalized_arguments.items()
            if key in allowed_keys
        }
    if normalized_arguments is arguments:
        return request
    return request.override(
        tool_call={
            **request.tool_call,
            "args": dict(normalized_arguments),
        }
    )


def _path_source_hint(path: Any) -> str:
    normalized = str(path or "").replace("\\", "/").strip()
    if not normalized:
        return ""
    tail = normalized.rsplit("::", 1)[-1].rsplit("/", 1)[-1]
    return tail.rsplit(".", 1)[0]


def _narrative_extraction_fields_for_source(
    *,
    state: Mapping[str, Any],
    discovery: _DiscoveryState,
    source_path: str,
) -> list[str]:
    source_hint = _path_source_hint(source_path)
    if not source_hint:
        return []
    knowledge_facts = list(parse_knowledge_content(discovery.knowledge_content))
    relevant_groups = _request_relevant_knowledge_source_hint_groups(
        knowledge_facts=knowledge_facts,
        state=state,
    )
    source_related_text = " ".join(
        str(getattr(fact, "quote", "") or "")
        for fact in knowledge_facts
        if _source_path_matches_hint(
            source_hint,
            str(getattr(fact, "section_key", "") or ""),
        )
        or source_hint in _source_hints_from_knowledge_text(
            str(getattr(fact, "quote", "") or ""),
        )
    )
    relevant_text = " ".join(
        [
            *(str(group.get("quote") or "") for group in relevant_groups),
            source_related_text,
        ]
    )
    relevant_terms = _contract_terms(relevant_text) | _contract_terms(_request_focus_text(state))
    section_facts = [
        fact
        for fact in knowledge_facts
        if _knowledge_fact_defines_field(fact)
        and _source_path_matches_hint(source_hint, str(getattr(fact, "section_key", "") or ""))
    ]
    selected_fields: list[str] = []
    for fact in section_facts:
        field_key = str(getattr(fact, "field_key", "") or "").strip()
        if not field_key:
            continue
        field_aliases = _knowledge_fact_field_aliases(fact)
        if field_aliases & relevant_terms or any(
            alias and alias in _normalized_field_alias(relevant_text)
            for alias in field_aliases
        ):
            selected_fields.append(field_key)
    if selected_fields and re.search(r"\bjoin\b|=", relevant_text, flags=re.IGNORECASE):
        for fact in section_facts:
            field_key = str(getattr(fact, "field_key", "") or "").strip()
            if not field_key or field_key in selected_fields:
                continue
            aliases = {
                _normalized_field_alias(field_key),
                *_knowledge_fact_field_aliases(fact),
            }
            if any(
                marker in alias
                for alias in aliases
                for marker in ("id", "key", "code")
            ):
                selected_fields.append(field_key)
    if selected_fields:
        return list(dict.fromkeys(selected_fields))
    return [
        str(getattr(fact, "field_key", "") or "").strip()
        for fact in section_facts[:6]
        if str(getattr(fact, "field_key", "") or "").strip()
    ]


def _canonicalize_extract_narrative_arguments(
    request: ToolCallRequest,
    discovery: _DiscoveryState,
) -> ToolCallRequest:
    if str(request.tool_call.get("name") or "") != "extract_narrative_records":
        return request
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        arguments = {}
    source_path = str(arguments.get("source_path") or arguments.get("path") or "").strip()
    source_fields = _narrative_extraction_fields_for_source(
        state=request.state,
        discovery=discovery,
        source_path=source_path,
    )
    provided_fields: list[str] = []
    raw_source_fields = arguments.get("source_fields")
    if isinstance(raw_source_fields, list):
        provided_fields.extend(
            str(field).strip()
            for field in raw_source_fields
            if str(field).strip()
        )
    elif isinstance(raw_source_fields, str) and raw_source_fields.strip():
        text = raw_source_fields.strip()
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                provided_fields.extend(
                    str(field).strip()
                    for field in decoded
                    if str(field).strip()
                )
            else:
                provided_fields.append(text)
        else:
            provided_fields.extend(
                field.strip()
                for field in re.split(r"[,;|]", text)
                if field.strip()
            )
    source_field = str(arguments.get("source_field") or "").strip()
    if source_field:
        provided_fields.append(source_field)
    merged_fields = list(dict.fromkeys([*provided_fields, *source_fields]))
    if not merged_fields:
        return request
    if merged_fields == provided_fields and arguments.get("source_fields"):
        return request
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "source_fields": merged_fields,
            },
        }
    )


_PROTOCOL_PARAMETER_MARKER = re.compile(
    r"\s*</?parameter[^>\n]*(?:>|\n|$)",
    re.IGNORECASE,
)


def _strip_protocol_markup_text(value: str) -> str:
    match = _PROTOCOL_PARAMETER_MARKER.search(value)
    if match is None:
        return value
    return value[: match.start()].strip()


def _strip_extract_narrative_protocol_markup(
    request: ToolCallRequest,
) -> ToolCallRequest:
    if str(request.tool_call.get("name") or "") != "extract_narrative_records":
        return request
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    updated_arguments: dict[str, Any] = {}
    changed = False
    for key, value in arguments.items():
        if isinstance(value, str):
            cleaned = _strip_protocol_markup_text(value)
            updated_arguments[key] = cleaned
            if cleaned != value:
                changed = True
        else:
            updated_arguments[key] = value
    if not changed:
        return request
    return request.override(
        tool_call={
            **request.tool_call,
            "args": updated_arguments,
        }
    )


def _normalize_write_todos_arguments(
    request: ToolCallRequest,
    plan: Mapping[str, Any],
) -> ToolCallRequest | ToolMessage:
    if str(request.tool_call.get("name") or "") != "write_todos":
        return request
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        arguments = {}
    todos = arguments.get("todos")
    plan_steps = [str(step) for step in plan.get("steps") or []]
    if not plan_steps:
        return _tool_error(
            request,
            "analysis_plan.steps must contain at least one step before write_todos.",
        )

    incoming_todos = todos if isinstance(todos, list) else []
    previous_todos = request.state.get("todos") or []
    previous_status_by_content = {
        str(todo.get("content") or ""): str(todo.get("status") or "")
        for todo in previous_todos
        if isinstance(todo, Mapping)
    }
    normalized_todos: list[dict[str, Any]] = []
    for index, plan_step in enumerate(plan_steps):
        incoming = incoming_todos[index] if index < len(incoming_todos) else {}
        incoming_status = (
            str(incoming.get("status") or "")
            if isinstance(incoming, Mapping)
            else ""
        )
        previous_status = previous_status_by_content.get(plan_step, "")
        status = (
            incoming_status
            if incoming_status in _TODO_STATUSES
            else previous_status
            if previous_status in _TODO_STATUSES
            else "pending"
        )
        normalized_todos.append({"content": plan_step, "status": status})

    if (
        normalized_todos
        and all(todo["status"] == "pending" for todo in normalized_todos)
    ):
        normalized_todos[0]["status"] = "in_progress"

    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "todos": normalized_todos,
            },
        }
    )


def _canonical_plan_steps(arguments: Mapping[str, Any]) -> list[str]:
    output_spec = arguments.get("output_spec") or {}
    evidence = arguments.get("evidence") or {}
    context_sources = (
        evidence.get("context_sources") if isinstance(evidence, Mapping) else []
    )
    source_paths = [
        str(source.get("path") or "")
        for source in context_sources or []
        if isinstance(source, Mapping) and source.get("path")
    ]
    if source_paths:
        source_label = ", ".join(source_paths[:2])
        if len(source_paths) > 2:
            source_label = f"{source_label}, and {len(source_paths) - 2} more"
        read_step = f"Read relevant source data from {source_label}"
    else:
        read_step = "Read relevant source data from analysis_plan.evidence.context_sources"

    columns = output_spec.get("columns") if isinstance(output_spec, Mapping) else []
    column_names = [
        str(column.get("name") or "")
        for column in columns or []
        if isinstance(column, Mapping) and column.get("name")
    ]
    column_label = ", ".join(column_names) if column_names else "output_spec.columns"
    transformations = (
        output_spec.get("transformations") if isinstance(output_spec, Mapping) else []
    )
    if transformations:
        operations = [
            str(item.get("operation") or "transformation")
            for item in transformations
            if isinstance(item, Mapping)
        ]
        operation_label = ", ".join(operations) if operations else "transformations"
        prepare_step = (
            "Apply authorized output_spec.transformations "
            f"({operation_label}) and project {column_label}"
        )
    elif (
        isinstance(output_spec, Mapping)
        and str(output_spec.get("row_policy") or "") == "preserve"
    ):
        prepare_step = (
            f"Project {column_label} from source rows preserving source order and nulls"
        )
    else:
        prepare_step = "Construct output rows according to analysis_plan.output_spec"

    return [
        read_step,
        prepare_step,
        "Submit final answer with analysis_plan.output_spec columns",
    ]


def _canonicalize_plan_steps(request: ToolCallRequest) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    canonical_steps = _canonical_plan_steps(arguments)
    if arguments.get("steps") == canonical_steps:
        return request
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "steps": canonical_steps,
            },
        }
    )


def _tool_payload(message: BaseMessage) -> Mapping[str, Any] | None:
    if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
        return None
    try:
        payload = json.loads(message.content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _pre_plan_execution_count(messages: list[BaseMessage]) -> int:
    count = 0
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        name = str(getattr(message, "name", "") or "")
        if name in _PRE_PLAN_EXECUTION_TOOLS:
            count += 1
    return count


def _pre_plan_discovery_count(messages: list[BaseMessage]) -> int:
    count = 0
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        name = str(getattr(message, "name", "") or "")
        if name in _SOURCE_DISCOVERY_TOOLS:
            count += 1
    return count


def _last_plan_error_requests_more_evidence(messages: list[BaseMessage]) -> bool:
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        if str(getattr(message, "name", "") or "") != "analyze_plan":
            return False
        if str(getattr(message, "status", "") or "") != "error":
            return False
        content = str(message.content or "").casefold()
        evidence_markers = (
            "complete discovery before analyze_plan",
            "bind them to observed source fields/values",
            "context sources must come from successful discovery",
            "run query_schema",
            "run extract_narrative_records",
            "source binding",
            "observed source",
            "证据",
        )
        return any(marker in content for marker in evidence_markers)
    return False


def _pre_plan_execution_gate_active(
    *,
    state: Mapping[str, Any],
    discovery: _DiscoveryState,
) -> bool:
    if state.get("analysis_plan") is not None:
        return False
    messages = state.get("messages")
    if not isinstance(messages, list):
        return False
    if _last_plan_error_requests_more_evidence(messages):
        return False
    if _last_narrative_extraction_incomplete(messages):
        return False
    return (
        discovery.context_ready
        and (
            _pre_plan_execution_count(messages) >= _PRE_PLAN_EXECUTION_LIMIT
            or _pre_plan_discovery_count(messages) >= _PRE_PLAN_DISCOVERY_LIMIT
        )
    )


def _has_narrative_extraction_attempt(messages: list[BaseMessage]) -> bool:
    return any(
        isinstance(message, ToolMessage)
        and str(getattr(message, "name", "") or "") == "extract_narrative_records"
        for message in messages
    )


def _last_narrative_extraction_incomplete(messages: list[BaseMessage]) -> bool:
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        if str(getattr(message, "name", "") or "") != "extract_narrative_records":
            return False
        try:
            payload = json.loads(str(message.content or ""))
        except json.JSONDecodeError:
            return "extracted_incomplete" in str(message.content or "")
        if not isinstance(payload, Mapping):
            return False
        return bool(payload.get("incomplete_fields")) or (
            str(payload.get("status") or "") == "extracted_incomplete"
        )
    return False


def _candidate_value_is_empty(value: Any) -> bool:
    return value is None or value == ""


def _answer_candidate_has_complete_columns(candidate: Any) -> bool:
    if not isinstance(candidate, Mapping):
        return False
    columns = candidate.get("columns")
    rows = candidate.get("rows")
    if not isinstance(columns, list) or not columns:
        return False
    if not isinstance(rows, list) or not rows:
        return False
    for column_index, _ in enumerate(columns):
        if not any(
            isinstance(row, list)
            and column_index < len(row)
            and not _candidate_value_is_empty(row[column_index])
            for row in rows
        ):
            return False
    return True


def _plan_needs_initial_narrative_extraction(state: Mapping[str, Any]) -> bool:
    plan = state.get("analysis_plan")
    if not isinstance(plan, Mapping):
        return False
    messages = state.get("messages")
    if isinstance(messages, list) and _has_narrative_extraction_attempt(messages):
        return False
    output_spec = plan.get("output_spec")
    execution_spec = plan.get("execution_spec")
    has_transform = False
    if isinstance(output_spec, Mapping):
        has_transform = bool(output_spec.get("transformations"))
    if isinstance(execution_spec, Mapping):
        has_transform = has_transform or bool(execution_spec.get("operations"))
    if not has_transform:
        return False
    paths: list[str] = []
    if isinstance(execution_spec, Mapping):
        for source in execution_spec.get("sources") or []:
            if isinstance(source, Mapping):
                paths.append(str(source.get("path") or ""))
    evidence = plan.get("evidence")
    if isinstance(evidence, Mapping):
        for source in evidence.get("context_sources") or []:
            if isinstance(source, Mapping):
                paths.append(str(source.get("path") or ""))
    if not any(_source_path_is_narrative(path) for path in paths):
        return False
    intent = plan.get("intent")
    unresolved_items = intent.get("unresolved", []) if isinstance(intent, Mapping) else []
    unresolved_text = " ".join(str(item or "") for item in unresolved_items).casefold()
    if "extract" in unresolved_text or "抽取" in unresolved_text:
        return True
    return any(
        _source_path_is_narrative(path)
        for path in _source_paths_from_plan_arguments(
            {
                "execution_spec": execution_spec,
                "evidence": evidence,
            }
        )
    )


def _last_plan_error_mentions_narrative_evidence(messages: list[BaseMessage]) -> bool:
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        if str(getattr(message, "name", "") or "") != "analyze_plan":
            return False
        if str(getattr(message, "status", "") or "") != "error":
            return False
        content = str(message.content or "").casefold()
        markers = (
            "extract_narrative_records",
            "narrative",
            "pdf",
            "document",
            "source/field binding",
            "source binding",
            "field evidence",
            "抽取",
            "文档",
            "证据",
        )
        return any(marker in content for marker in markers)
    return False


def _pre_plan_needs_narrative_extraction(state: Mapping[str, Any]) -> bool:
    if state.get("analysis_plan") is not None:
        return False
    messages = state.get("messages")
    if not isinstance(messages, list):
        return False
    if _has_narrative_extraction_attempt(messages):
        return False
    if not _last_plan_error_mentions_narrative_evidence(messages):
        return False
    return any(_observed_source_is_narrative(source) for source in _state_observed_sources(state))


def _pre_plan_observed_narrative_hint_needs_extraction(
    state: Mapping[str, Any],
    discovery: _DiscoveryState,
) -> bool:
    if state.get("analysis_plan") is not None:
        return False
    messages = state.get("messages")
    if isinstance(messages, list) and _has_narrative_extraction_attempt(messages):
        return False
    knowledge_facts = list(parse_knowledge_content(discovery.knowledge_content))
    hint_groups = _request_relevant_knowledge_source_hint_groups(
        knowledge_facts=knowledge_facts,
        state=state,
    )
    if not hint_groups:
        return False
    hinted_sources = {
        str(hint)
        for group in hint_groups
        for hint in group.get("source_hints") or []
        if str(hint or "").strip()
    }
    if not hinted_sources:
        return False
    for source in _state_observed_sources(state):
        if not _observed_source_is_narrative(source):
            continue
        path = str(source.get("path") or "").replace("\\", "/")
        source_name_hint = str(source.get("source_name_hint") or "")
        if any(
            _source_path_matches_hint(path, hint)
            or (
                source_name_hint
                and _normalized_quote_text(source_name_hint)
                == _normalized_quote_text(hint)
            )
            for hint in hinted_sources
        ):
            return True
    return False


def _observed_source_row_counts(messages: list[BaseMessage]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for message in messages:
        payload = _tool_payload(message)
        if payload is None:
            continue
        path = str(payload.get("path") or "").replace("\\", "/")
        if not path:
            continue
        row_count = payload.get("total_items")
        if row_count is None:
            row_count = payload.get("total_rows")
        if isinstance(row_count, int) and row_count >= 0:
            counts[path] = row_count
    return counts


def _state_observed_sources(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources = state.get("observed_sources")
    if not isinstance(sources, list):
        return []
    return [source for source in sources if isinstance(source, Mapping)]


def _source_path_aliases(path: Any) -> set[str]:
    normalized = str(path or "").replace("\\", "/").strip()
    if not normalized:
        return set()
    aliases = {normalized}
    if "::" in normalized:
        aliases.add(normalized.split("::", 1)[0])
    return aliases


def _source_mapping_paths(source: Mapping[str, Any]) -> set[str]:
    paths = set()
    for key in ("path", "base_path"):
        paths.update(_source_path_aliases(source.get(key)))
    return {path for path in paths if path}


def _source_path_is_narrative(path: Any) -> bool:
    normalized = str(path or "").replace("\\", "/").casefold()
    if "::" in normalized:
        normalized = normalized.split("::", 1)[0]
    return any(normalized.endswith(suffix) for suffix in _NARRATIVE_SOURCE_SUFFIXES)


def _observed_source_is_narrative(source: Mapping[str, Any]) -> bool:
    source_type = str(source.get("source_type") or "").strip().casefold()
    return (
        source_type in _NARRATIVE_SOURCE_TYPES
        or any(_source_path_is_narrative(path) for path in _source_mapping_paths(source))
    )


def _observed_source_is_structured(source: Mapping[str, Any]) -> bool:
    if _observed_source_is_narrative(source):
        return False
    source_type = str(source.get("source_type") or "").strip().casefold()
    return source_type in _STRUCTURED_SOURCE_TYPES or isinstance(
        source.get("fields"),
        list,
    )


def _field_aliases_from_observed_source(source: Mapping[str, Any]) -> set[str]:
    aliases: set[str] = set()
    fields = source.get("fields")
    if isinstance(fields, list):
        for item in fields:
            field = item.get("name") if isinstance(item, Mapping) else item
            text = str(field or "").strip()
            if not text:
                continue
            aliases.add(_normalized_field_alias(text))
            if "." in text:
                aliases.add(_normalized_field_alias(text.rsplit(".", 1)[-1]))
    for key in ("field", "source_field"):
        value = source.get(key)
        if str(value or "").strip():
            aliases.add(_normalized_field_alias(value))
    for evidence_key in ("value_evidence", "field_evidence"):
        evidence_items = source.get(evidence_key)
        if not isinstance(evidence_items, list):
            continue
        for item in evidence_items:
            if isinstance(item, Mapping) and str(item.get("field") or "").strip():
                aliases.add(_normalized_field_alias(item.get("field")))
    extracted_fields = source.get("extracted_fields")
    if isinstance(extracted_fields, list):
        aliases.update(
            _normalized_field_alias(item)
            for item in extracted_fields
            if str(item or "").strip()
        )
    return {alias for alias in aliases if alias}


def _observed_source_matches_path(
    source: Mapping[str, Any],
    path: Any,
) -> bool:
    wanted = _source_path_aliases(path)
    if not wanted:
        return False
    return bool(wanted & _source_mapping_paths(source))


def _observed_sources_for_paths(
    state: Mapping[str, Any],
    paths: Iterable[Any],
) -> list[Mapping[str, Any]]:
    requested_paths = [path for path in paths if str(path or "").strip()]
    if not requested_paths:
        return []
    matches: list[Mapping[str, Any]] = []
    for source in _state_observed_sources(state):
        if any(_observed_source_matches_path(source, path) for path in requested_paths):
            matches.append(source)
    return matches


def _source_field_has_observed_evidence(
    *,
    state: Mapping[str, Any],
    source_field: Any,
    source_paths: Iterable[Any],
) -> bool:
    alias = _normalized_field_alias(source_field)
    if not alias:
        return False
    path_list = [path for path in source_paths if str(path or "").strip()]
    observed_sources = (
        _observed_sources_for_paths(state, path_list)
        if path_list
        else _state_observed_sources(state)
    )
    for source in observed_sources:
        aliases = _field_aliases_from_observed_source(source)
        if alias not in aliases:
            continue
        if _observed_source_is_structured(source):
            return True
        if (
            _observed_source_is_narrative(source)
            and str(source.get("observed_by") or "").strip()
            in _NARRATIVE_FIELD_EVIDENCE_TOOLS
        ):
            return True
    return False


def _source_binding_has_observed_field_evidence(
    *,
    binding: Mapping[str, Any],
    state: Mapping[str, Any],
) -> bool:
    source_paths = binding.get("source_paths")
    if not isinstance(source_paths, list):
        source_paths = []
    return _source_field_has_observed_evidence(
        state=state,
        source_field=binding.get("source_field"),
        source_paths=source_paths,
    )


def _field_fact_has_observed_physical_field(
    *,
    fact: Any,
    state: Mapping[str, Any],
) -> bool:
    section_key = _normalized_quote_text(str(getattr(fact, "section_key", "") or ""))
    field_aliases = _knowledge_fact_field_aliases(fact)
    if not section_key or not field_aliases:
        return False
    for source in _state_observed_sources(state):
        if not _observed_source_is_structured(source):
            continue
        source_hints = {
            _normalized_quote_text(str(source.get("source_name_hint") or "")),
            _normalized_quote_text(str(source.get("table") or "")),
        }
        for path in _source_mapping_paths(source):
            path_tail = path.rsplit("::", 1)[-1].rsplit("/", 1)[-1]
            source_hints.add(_normalized_quote_text(path_tail.rsplit(".", 1)[0]))
        if section_key not in source_hints:
            continue
        if _field_aliases_from_observed_source(source) & field_aliases:
            return True
    return False


def _field_fact_has_observed_narrative_extraction(
    *,
    fact: Any,
    state: Mapping[str, Any],
) -> bool:
    section_key = _normalized_quote_text(str(getattr(fact, "section_key", "") or ""))
    field_aliases = _knowledge_fact_field_aliases(fact)
    if not section_key or not field_aliases:
        return False
    for source in _state_observed_sources(state):
        if not _observed_source_is_narrative(source):
            continue
        if (
            str(source.get("observed_by") or "").strip()
            not in _NARRATIVE_FIELD_EVIDENCE_TOOLS
        ):
            continue
        source_hints = {
            _normalized_quote_text(str(source.get("source_name_hint") or "")),
        }
        for path in _source_mapping_paths(source):
            path_tail = path.rsplit("/", 1)[-1]
            source_hints.add(_normalized_quote_text(path_tail.rsplit(".", 1)[0]))
        if section_key not in source_hints:
            continue
        if _field_aliases_from_observed_source(source) & field_aliases:
            return True
    return False


def _observed_context_sources_from_state(state: Mapping[str, Any]) -> set[str]:
    sources: set[str] = set()
    for source in _state_observed_sources(state):
        path = str(source.get("path") or "").replace("\\", "/")
        if not path or path.lower().endswith("/knowledge.md"):
            continue
        sources.add(path)
        base_path = str(source.get("base_path") or "").replace("\\", "/")
        if base_path:
            sources.add(base_path)
    candidate = state.get("answer_candidate")
    if isinstance(candidate, Mapping):
        for path in candidate.get("code_context_paths") or []:
            normalized = str(path or "").replace("\\", "/")
            if normalized and not normalized.lower().endswith("/knowledge.md"):
                sources.add(normalized)
    return sources


def _observed_source_row_counts_from_state(
    state: Mapping[str, Any],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in _state_observed_sources(state):
        path = str(source.get("path") or "").replace("\\", "/")
        row_count = source.get("row_count")
        if path and isinstance(row_count, int) and row_count >= 0:
            counts[path] = row_count
        base_path = str(source.get("base_path") or "").replace("\\", "/")
        if base_path and isinstance(row_count, int) and row_count >= 0:
            counts.setdefault(base_path, row_count)
    return counts


def _candidate_prepared_answer(
    *,
    candidate: Any,
    analysis_plan: Any,
) -> Any | None:
    if not isinstance(candidate, Mapping) or not isinstance(analysis_plan, dict):
        return None
    candidate_columns = candidate.get("columns")
    candidate_rows = candidate.get("rows")
    if not isinstance(candidate_columns, list) or not isinstance(candidate_rows, list):
        return None
    if not all(isinstance(row, list) for row in candidate_rows):
        return None
    columns = normalize_answer_columns(candidate_columns)
    rows = [list(row) for row in candidate_rows]
    audit = candidate.get("audit") if isinstance(candidate.get("audit"), dict) else None
    prepared_answer, answer_error = validate_prepared_answer(
        columns,
        rows,
        analysis_plan,
        audit,
    )
    if answer_error is not None:
        return None
    return prepared_answer


def _promote_answer_candidate_after_plan(
    request: ToolCallRequest,
    result: ToolMessage | Command[Any],
) -> ToolMessage | Command[Any]:
    if not isinstance(result, Command):
        return result
    update = getattr(result, "update", None)
    if not isinstance(update, dict):
        return result
    analysis_plan = update.get("analysis_plan")
    prepared_answer = _candidate_prepared_answer(
        candidate=request.state.get("answer_candidate"),
        analysis_plan=analysis_plan,
    )
    if prepared_answer is None:
        return result
    return Command(
        update={
            **update,
            "prepared_answer": prepared_answer,
            "answer_candidate": None,
        }
    )


def _observed_narrative_sources_by_source_hint(
    state: Mapping[str, Any],
) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}

    def add_source(source_name_hint: str, path: str) -> None:
        normalized_name = _normalized_quote_text(source_name_hint)
        normalized_path = path.replace("\\", "/")
        if not normalized_name or not normalized_path:
            return
        sources.setdefault(normalized_name, [])
        if normalized_path not in sources[normalized_name]:
            sources[normalized_name].append(normalized_path)

    for source in _state_observed_sources(state):
        if str(source.get("source_type") or "") != "doc":
            continue
        path = str(source.get("path") or "").replace("\\", "/")
        source_name_hint = str(source.get("source_name_hint") or "")
        if not source_name_hint or not path:
            continue
        add_source(source_name_hint, path)

    messages = state.get("messages")
    if isinstance(messages, list):
        for message in messages:
            payload = _tool_payload(message)
            if payload is None or "total_lines" not in payload:
                continue
            path = str(payload.get("path") or "").replace("\\", "/")
            lower_path = path.lower()
            if not lower_path.endswith((".md", ".markdown", ".txt", ".pdf")):
                continue
            filename = path.rsplit("/", 1)[-1]
            source_name_hint = filename.rsplit(".", 1)[0]
            add_source(source_name_hint, path)
    return sources


def _observed_sources_by_source_hint(
    state: Mapping[str, Any],
) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}
    for source in _state_observed_sources(state):
        path = str(source.get("path") or "").replace("\\", "/")
        if not path:
            continue
        source_name_hints = {
            _normalized_quote_text(str(source.get("source_name_hint") or "")),
            _normalized_quote_text(str(source.get("table") or "")),
        }
        path_tail = path.rsplit("::", 1)[-1].rsplit("/", 1)[-1]
        source_name_hints.add(_normalized_quote_text(path_tail.rsplit(".", 1)[0]))
        for source_name_hint in {name for name in source_name_hints if name}:
            sources.setdefault(source_name_hint, [])
            if path not in sources[source_name_hint]:
                sources[source_name_hint].append(path)
    return sources


def _canonicalize_preserve_expected_row_count(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    evidence = arguments.get("evidence")
    if not isinstance(output_spec, Mapping) or not isinstance(evidence, Mapping):
        return request
    if str(output_spec.get("row_policy") or "") != "preserve":
        return request
    if output_spec.get("transformations"):
        return request

    observed_counts = {
        **_observed_source_row_counts(request.state["messages"]),
        **_observed_source_row_counts_from_state(request.state),
    }
    context_sources = evidence.get("context_sources") or []
    candidate_counts = {
        observed_counts[path]
        for source in context_sources
        if isinstance(source, Mapping)
        and (path := str(source.get("path") or "").replace("\\", "/"))
        in observed_counts
    }
    if len(candidate_counts) != 1:
        return request

    updated_output_spec = dict(output_spec)
    expected_row_count = candidate_counts.pop()
    if updated_output_spec.get("expected_row_count") == expected_row_count:
        return request
    updated_output_spec["expected_row_count"] = expected_row_count
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "output_spec": updated_output_spec,
            },
        }
    )


def _canonicalize_preserve_output_policy(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return request
    if str(output_spec.get("row_policy") or "") != "preserve":
        return request
    if output_spec.get("transformations"):
        return request
    if output_spec.get("sort_keys"):
        return request
    if output_spec.get("null_policy") != "preserve":
        return request
    ordering = str(output_spec.get("ordering") or "")
    if ordering not in {"", "unspecified"}:
        return request

    updated_output_spec = dict(output_spec)
    updated_output_spec["ordering"] = "source"
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "output_spec": updated_output_spec,
            },
        }
    )


def _canonicalize_transform_output_policy(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return request
    if not output_spec.get("transformations"):
        return request
    if str(output_spec.get("row_policy") or "") == "transform":
        return request

    updated_output_spec = dict(output_spec)
    updated_output_spec["row_policy"] = "transform"
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "output_spec": updated_output_spec,
            },
        }
    )


def _canonicalize_sort_null_policy(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return request
    if not output_spec.get("sort_keys"):
        return request
    if str(output_spec.get("row_policy") or "") != "transform":
        return request
    null_policy = str(output_spec.get("null_policy") or "").strip().casefold()
    if null_policy and null_policy != "preserve":
        return request
    updated_output_spec = dict(output_spec)
    updated_output_spec["null_policy"] = "drop"
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "output_spec": updated_output_spec,
            },
        }
    )


def _canonicalize_unbacked_sort_keys(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return request
    if not output_spec.get("sort_keys"):
        return request
    declared_operations = {
        str(item.get("operation") or "").strip().casefold()
        for item in output_spec.get("transformations") or []
        if isinstance(item, Mapping)
    }
    execution_spec = arguments.get("execution_spec")
    if isinstance(execution_spec, Mapping):
        declared_operations.update(
            str(item.get("operation") or "").strip().casefold()
            for item in execution_spec.get("operations") or []
            if isinstance(item, Mapping)
        )
    if "sort" in declared_operations:
        return request
    updated_output_spec = dict(output_spec)
    updated_output_spec["ordering"] = "unspecified"
    updated_output_spec["sort_keys"] = []
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "output_spec": updated_output_spec,
            },
        }
    )


def _canonicalize_execution_supporting_fields(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    execution_spec = arguments.get("execution_spec")
    if isinstance(output_spec, str):
        try:
            output_spec = json.loads(output_spec)
        except json.JSONDecodeError:
            output_spec = {}
    if isinstance(execution_spec, str):
        try:
            execution_spec = json.loads(execution_spec)
        except json.JSONDecodeError:
            execution_spec = {}
    if not isinstance(output_spec, Mapping) or not isinstance(execution_spec, Mapping):
        return request

    output_field_names = {
        _normalized_field_alias(column.get("name"))
        for column in output_spec.get("columns") or []
        if isinstance(column, Mapping)
    }
    for column in output_spec.get("columns") or []:
        if not isinstance(column, Mapping):
            continue
        output_field_names.update(
            _normalized_field_alias(field)
            for field in column.get("source_fields") or []
            if _normalized_field_alias(field)
        )
    output_field_names = {name for name in output_field_names if name}
    if not output_field_names:
        return request

    supporting_fields = execution_spec.get("supporting_fields") or []
    if not isinstance(supporting_fields, list):
        return request
    normalized_fields = []
    changed = False
    for field in supporting_fields:
        if not isinstance(field, Mapping):
            normalized_fields.append(field)
            continue
        supporting_names = {
            _normalized_field_alias(field.get("name")),
            *(
                _normalized_field_alias(item)
                for item in field.get("source_fields") or []
                if _normalized_field_alias(item)
            ),
        }
        if output_field_names & {name for name in supporting_names if name}:
            changed = True
            continue
        normalized_fields.append(field)
    if not changed:
        return request

    updated_execution_spec = dict(execution_spec)
    updated_execution_spec["supporting_fields"] = normalized_fields
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "output_spec": output_spec,
                "execution_spec": updated_execution_spec,
            },
        }
    )


def _question_requested_output_texts(state: Mapping[str, Any]) -> set[str]:
    question_structure = state.get("question_structure")
    if not isinstance(question_structure, Mapping):
        return set()
    original_request = str(state.get("original_request") or "")
    texts: set[str] = set()
    output = question_structure.get("output")
    if isinstance(output, Mapping):
        texts.update(
            text
            for item in output.get("requested_columns") or []
            if (text := str(item or "").strip()) and text in original_request
        )
    for target in question_structure.get("targets") or []:
        if not isinstance(target, Mapping):
            continue
        target_type = str(target.get("target_type") or "").strip().casefold()
        if target_type in {"measure", "metric"}:
            continue
        quote = str(target.get("quote") or "").strip()
        if quote and quote in original_request:
            texts.add(quote)
        name = str(target.get("name") or "").strip()
        if name:
            texts.add(name)
    conditions = question_structure.get("conditions")
    if isinstance(conditions, Mapping):
        for item in conditions.get("output_columns") or []:
            if not isinstance(item, Mapping):
                continue
            for key in ("quote", "value"):
                text = str(item.get(key) or "").strip()
                if text:
                    texts.add(text)
    return texts


def _explicit_question_output_column_texts(state: Mapping[str, Any]) -> set[str]:
    question_structure = state.get("question_structure")
    if not isinstance(question_structure, Mapping):
        return set()
    texts: set[str] = set()
    conditions = question_structure.get("conditions")
    if not isinstance(conditions, Mapping):
        return texts
    for item in conditions.get("output_columns") or []:
        if not isinstance(item, Mapping):
            continue
        quote = str(item.get("quote") or "").strip()
        if quote:
            texts.add(quote)
    return texts


def _column_texts(column: Mapping[str, Any]) -> set[str]:
    texts = {str(column.get("name") or "").strip()}
    texts.update(
        str(field or "").strip()
        for field in column.get("source_fields") or []
        if str(field or "").strip()
    )
    return {text for text in texts if text}


def _normalized_field_alias(value: Any) -> str:
    normalized = _normalized_quote_text(str(value or "")).replace(" ", "")
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", normalized)


def _column_field_aliases(column: Mapping[str, Any]) -> set[str]:
    return {
        _normalized_field_alias(text)
        for text in _column_texts(column)
        if _normalized_field_alias(text)
    }


def _column_matches_requested_output(
    column: Mapping[str, Any],
    requested_texts: set[str],
) -> bool:
    column_texts = _column_texts(column)
    if not column_texts or not requested_texts:
        return False
    for column_text in column_texts:
        normalized_column = _normalized_quote_text(column_text)
        for requested_text in requested_texts:
            normalized_requested = _normalized_quote_text(requested_text)
            if not normalized_column or not normalized_requested:
                continue
            if normalized_column == normalized_requested:
                return True
            if normalized_column in normalized_requested or normalized_requested in normalized_column:
                return True
    return False


def _question_requested_measure_texts(state: Mapping[str, Any]) -> set[str]:
    question_structure = state.get("question_structure")
    if not isinstance(question_structure, Mapping):
        return set()
    texts: set[str] = set()
    output = question_structure.get("output")
    if isinstance(output, Mapping):
        texts.update(
            text
            for item in output.get("requested_columns") or []
            if (text := str(item or "").strip())
        )
    for target in question_structure.get("targets") or []:
        if not isinstance(target, Mapping):
            continue
        target_type = str(target.get("target_type") or "").strip().casefold()
        if target_type not in {"measure", "metric"}:
            continue
        for key in ("quote", "name", "description"):
            text = str(target.get(key) or "").strip()
            if text:
                texts.add(text)
    return texts


def _field_token_set(value: Any) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value or ""))
    text = text.replace("_", " ")
    return _normalized_quote_tokens(text)


def _column_matches_requested_measure(
    column: Mapping[str, Any],
    state: Mapping[str, Any],
) -> bool:
    requested_texts = _question_requested_measure_texts(state)
    if not requested_texts:
        return False
    if _column_matches_requested_output(column, requested_texts):
        return True
    for column_text in _column_texts(column):
        column_tokens = _field_token_set(column_text)
        if not column_tokens:
            continue
        for requested_text in requested_texts:
            requested_tokens = _field_token_set(requested_text)
            if not requested_tokens:
                continue
            if column_tokens <= requested_tokens or requested_tokens <= column_tokens:
                return True
    return False


def _should_keep_key_output_column(
    column: Mapping[str, Any],
    *,
    requested_texts: set[str],
    distribution_output: bool = False,
) -> bool:
    role = str(column.get("role") or "")
    if role in _KEY_OUTPUT_ROLES:
        if distribution_output:
            return True
        return _column_matches_requested_output(column, requested_texts)
    return True


def _canonicalize_unrequested_key_output_columns(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return request
    output_columns = output_spec.get("columns") or []
    if not isinstance(output_columns, list) or len(output_columns) <= 1:
        return request

    requested_texts = _question_requested_output_texts(request.state)
    distribution_output = _question_requests_distribution(request.state) and any(
        isinstance(column, Mapping) and _column_is_count_measure(column)
        for column in output_columns
    )
    kept_columns: list[Any] = []
    demoted_columns: list[Mapping[str, Any]] = []
    for column in output_columns:
        if not isinstance(column, Mapping):
            kept_columns.append(column)
            continue
        role = str(column.get("role") or "")
        if role in _KEY_OUTPUT_ROLES and not _should_keep_key_output_column(
            column,
            requested_texts=requested_texts,
            distribution_output=distribution_output,
        ):
            demoted_columns.append(column)
            continue
        kept_columns.append(column)

    if not demoted_columns or not kept_columns:
        return request

    updated_output_spec = dict(output_spec)
    updated_output_spec["columns"] = kept_columns
    execution_spec = arguments.get("execution_spec")
    updated_execution_spec = dict(execution_spec) if isinstance(execution_spec, Mapping) else {}
    updated_execution_spec.setdefault("sources", [])
    updated_execution_spec.setdefault("operations", [])
    supporting_fields = [
        dict(item)
        for item in updated_execution_spec.get("supporting_fields") or []
        if isinstance(item, Mapping)
    ]
    existing_supporting_keys = {
        (
            str(item.get("name") or "").casefold(),
            tuple(str(field or "").casefold() for field in item.get("source_fields") or []),
        )
        for item in supporting_fields
    }
    for column in demoted_columns:
        name = str(column.get("name") or "").strip()
        source_fields = [
            str(field or "").strip()
            for field in column.get("source_fields") or []
            if str(field or "").strip()
        ]
        key = (name.casefold(), tuple(field.casefold() for field in source_fields))
        if not name or key in existing_supporting_keys:
            continue
        supporting_fields.append(
            {
                "name": name,
                "source_fields": source_fields or [name],
                "purpose": "context",
            }
        )
        existing_supporting_keys.add(key)
    updated_execution_spec["supporting_fields"] = supporting_fields

    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "output_spec": updated_output_spec,
                "execution_spec": updated_execution_spec,
            },
        }
    )


def _demote_output_columns(
    *,
    arguments: Mapping[str, Any],
    output_spec: Mapping[str, Any],
    demoted_columns: list[Mapping[str, Any]],
    kept_columns: list[Any],
    purpose: str,
) -> dict[str, Any] | None:
    if not demoted_columns or not kept_columns:
        return None
    updated_output_spec = dict(output_spec)
    updated_output_spec["columns"] = kept_columns
    execution_spec = arguments.get("execution_spec")
    updated_execution_spec = (
        dict(execution_spec) if isinstance(execution_spec, Mapping) else {}
    )
    updated_execution_spec.setdefault("sources", [])
    updated_execution_spec.setdefault("operations", [])
    supporting_fields = [
        dict(item)
        for item in updated_execution_spec.get("supporting_fields") or []
        if isinstance(item, Mapping)
    ]
    existing_supporting_keys = {
        (
            str(item.get("name") or "").casefold(),
            tuple(
                str(field or "").casefold()
                for field in item.get("source_fields") or []
            ),
        )
        for item in supporting_fields
    }
    for column in demoted_columns:
        name = str(column.get("name") or "").strip()
        source_fields = [
            str(field or "").strip()
            for field in column.get("source_fields") or []
            if str(field or "").strip()
        ]
        key = (name.casefold(), tuple(field.casefold() for field in source_fields))
        if not name or key in existing_supporting_keys:
            continue
        supporting_fields.append(
            {
                "name": name,
                "source_fields": source_fields or [name],
                "purpose": purpose,
            }
        )
        existing_supporting_keys.add(key)
    updated_execution_spec["supporting_fields"] = supporting_fields
    return {
        **dict(arguments),
        "output_spec": updated_output_spec,
        "execution_spec": updated_execution_spec,
    }


def _selector_aliases_from_output_spec(
    output_spec: Mapping[str, Any],
    execution_spec: Mapping[str, Any] | None = None,
) -> set[str]:
    aliases = {
        _normalized_field_alias(item.get("field"))
        for item in output_spec.get("sort_keys") or []
        if isinstance(item, Mapping)
        and str(item.get("field") or "").strip()
    }
    if isinstance(execution_spec, Mapping):
        for operation in execution_spec.get("operations") or []:
            if not isinstance(operation, Mapping):
                continue
            operation_name = str(operation.get("operation") or "").casefold()
            if operation_name != "sort":
                continue
            for key in ("field", "sort_key", "sort_by", "order_by"):
                alias = _normalized_field_alias(operation.get(key))
                if alias:
                    aliases.add(alias)
        for field in execution_spec.get("supporting_fields") or []:
            if not isinstance(field, Mapping):
                continue
            purpose = str(field.get("purpose") or "").casefold()
            if purpose not in {"selector", "sort", "filter"}:
                continue
            alias = _normalized_field_alias(field.get("name"))
            if alias:
                aliases.add(alias)
            aliases.update(
                _normalized_field_alias(item)
                for item in field.get("source_fields") or []
                if _normalized_field_alias(item)
            )
    return {alias for alias in aliases if alias}


def _canonicalize_selector_output_columns(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    execution_spec = arguments.get("execution_spec")
    if not isinstance(output_spec, Mapping):
        return request
    output_columns = output_spec.get("columns") or []
    if not isinstance(output_columns, list) or len(output_columns) <= 1:
        return request
    selector_aliases = _selector_aliases_from_output_spec(
        output_spec,
        execution_spec if isinstance(execution_spec, Mapping) else None,
    )
    if not selector_aliases:
        return request

    requested_texts = _explicit_question_output_column_texts(request.state)
    requests_distribution = _question_requests_distribution(request.state)
    kept_columns: list[Any] = []
    demoted_columns: list[Mapping[str, Any]] = []
    for column in output_columns:
        if not isinstance(column, Mapping):
            kept_columns.append(column)
            continue
        if requests_distribution:
            kept_columns.append(column)
            continue
        if (
            _column_field_aliases(column) & selector_aliases
            and not _column_matches_requested_output(column, requested_texts)
        ):
            demoted_columns.append(column)
            continue
        kept_columns.append(column)

    updated_args = _demote_output_columns(
        arguments=arguments,
        output_spec=output_spec,
        demoted_columns=demoted_columns,
        kept_columns=kept_columns,
        purpose="selector",
    )
    if updated_args is None:
        return request
    return request.override(
        tool_call={**request.tool_call, "args": updated_args}
    )


def _canonicalize_unrequested_knowledge_output_columns(
    request: ToolCallRequest,
    discovery: _DiscoveryState,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return request
    output_columns = output_spec.get("columns") or []
    if not isinstance(output_columns, list) or len(output_columns) <= 1:
        return request
    evidence = arguments.get("evidence")
    if not isinstance(evidence, Mapping):
        return request
    knowledge_facts = [
        fact
        for fact in parse_knowledge_content(discovery.knowledge_content)
        if _knowledge_fact_defines_field(fact)
    ]
    facts_by_id = {str(fact.fact_id): fact for fact in knowledge_facts}
    facts_by_quote: dict[str, list[Any]] = {}
    for fact in knowledge_facts:
        facts_by_quote.setdefault(str(fact.quote), []).append(fact)
    field_facts: list[Any] = []
    for rule in evidence.get("knowledge_rules") or []:
        if not isinstance(rule, Mapping):
            continue
        fact_id = str(rule.get("fact_id") or "").strip()
        quote = str(rule.get("quote") or "").strip()
        if fact_id and fact_id in facts_by_id:
            field_facts.append(facts_by_id[fact_id])
        elif quote:
            field_facts.extend(
                facts_by_quote.get(quote)
                or _field_facts_for_knowledge_quote(
                    quote=quote,
                    field_facts=knowledge_facts,
                )
            )
    if not field_facts:
        return request
    authorized_aliases = {
        alias
        for fact in field_facts
        for alias in _knowledge_fact_field_aliases(fact)
    }
    if not authorized_aliases:
        return request

    requested_texts = _question_requested_output_texts(request.state)
    requests_distribution = _question_requests_distribution(request.state)
    kept_columns: list[Any] = []
    demoted_columns: list[Mapping[str, Any]] = []
    for column in output_columns:
        if not isinstance(column, Mapping):
            kept_columns.append(column)
            continue
        role = str(column.get("role") or "")
        column_aliases = _column_field_aliases(column)
        if requests_distribution and role in _KEY_OUTPUT_ROLES:
            kept_columns.append(column)
            continue
        if requests_distribution and _column_is_count_measure(column):
            kept_columns.append(column)
            continue
        if column_aliases & authorized_aliases:
            kept_columns.append(column)
            continue
        if _column_matches_requested_output(column, requested_texts):
            kept_columns.append(column)
            continue
        demoted_columns.append(column)

    updated_args = _demote_output_columns(
        arguments=arguments,
        output_spec=output_spec,
        demoted_columns=demoted_columns,
        kept_columns=kept_columns,
        purpose="context",
    )
    if updated_args is None:
        return request
    return request.override(
        tool_call={**request.tool_call, "args": updated_args}
    )


def _canonicalize_output_columns_from_valid_field_bindings(
    request: ToolCallRequest,
    discovery: _DiscoveryState,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    execution_spec = arguments.get("execution_spec")
    if not isinstance(output_spec, Mapping) or not isinstance(execution_spec, Mapping):
        return request
    output_columns = output_spec.get("columns") or []
    if not isinstance(output_columns, list) or not output_columns:
        return request

    knowledge_facts_by_id = {
        str(fact.fact_id): fact
        for fact in parse_knowledge_content(discovery.knowledge_content)
        if _knowledge_fact_defines_field(fact)
    }
    valid_bindings: list[Mapping[str, Any]] = []
    valid_binding_aliases: set[str] = set()
    for binding in execution_spec.get("source_bindings") or []:
        if not isinstance(binding, Mapping):
            continue
        fact = knowledge_facts_by_id.get(str(binding.get("fact_id") or ""))
        source_field = str(binding.get("source_field") or "").strip()
        if fact is None or not source_field:
            continue
        if _normalized_field_alias(source_field) not in _knowledge_fact_field_aliases(fact):
            continue
        if not _source_binding_has_observed_field_evidence(
            binding=binding,
            state=request.state,
        ):
            continue
        alias = _normalized_field_alias(source_field)
        if alias in valid_binding_aliases:
            continue
        valid_bindings.append(binding)
        valid_binding_aliases.add(alias)
    if not valid_bindings:
        return request

    output_spec_after_demote = output_spec
    if (
        len(output_columns) > 1
        and str(output_spec.get("row_policy") or "") == "preserve"
        and not output_spec.get("transformations")
    ):
        requested_texts = _question_requested_output_texts(request.state)
        kept_columns: list[Any] = []
        demoted_columns: list[Mapping[str, Any]] = []
        for column in output_columns:
            if not isinstance(column, Mapping):
                kept_columns.append(column)
                continue
            column_aliases = _column_field_aliases(column)
            if column_aliases & valid_binding_aliases:
                kept_columns.append(column)
                continue
            if _column_matches_requested_output(column, requested_texts):
                kept_columns.append(column)
                continue
            if _column_matches_requested_measure(column, request.state):
                kept_columns.append(column)
                continue
            demoted_columns.append(column)
        updated_args = _demote_output_columns(
            arguments=arguments,
            output_spec=output_spec,
            demoted_columns=demoted_columns,
            kept_columns=kept_columns,
            purpose="context",
        )
        if updated_args is not None:
            request = request.override(
                tool_call={**request.tool_call, "args": updated_args}
            )
            arguments = updated_args
            output_spec_after_demote = updated_args.get("output_spec", output_spec)
            output_columns = (
                output_spec_after_demote.get("columns") or []
                if isinstance(output_spec_after_demote, Mapping)
                else []
            )
            if not isinstance(output_columns, list) or not output_columns:
                return request

    output_aliases = {
        _normalized_field_alias(field)
        for column in output_columns
        if isinstance(column, Mapping)
        for field in column.get("source_fields") or []
        if str(field or "").strip()
    }
    if output_aliases & valid_binding_aliases:
        unused_bindings = [
            binding
            for binding in valid_bindings
            if _normalized_field_alias(binding.get("source_field")) not in output_aliases
        ]
    elif len(output_columns) == 1 and len(valid_bindings) == 1:
        unused_bindings = list(valid_bindings)
    else:
        return request
    if len(unused_bindings) != 1:
        return request
    replacement_field = str(unused_bindings[0].get("source_field") or "").strip()
    if not replacement_field:
        return request
    declared_operations = {
        str(item.get("operation") or "").strip().casefold()
        for item in output_spec_after_demote.get("transformations") or []
        if isinstance(item, Mapping)
    }
    if isinstance(execution_spec, Mapping):
        declared_operations.update(
            str(item.get("operation") or "").strip().casefold()
            for item in execution_spec.get("operations") or []
            if isinstance(item, Mapping)
        )

    updated_columns: list[Any] = []
    changed = False
    replacement_used = False
    for column in output_columns:
        if not isinstance(column, Mapping):
            updated_columns.append(column)
            continue
        role = str(column.get("role") or "").strip().casefold()
        if role in {"calculation", "calculated", "aggregate_value"} or (
            "aggregate" in declared_operations and _column_is_count_measure(column)
        ):
            updated_columns.append(column)
            continue
        column_aliases = _column_field_aliases(column)
        if column_aliases & valid_binding_aliases or replacement_used:
            updated_columns.append(column)
            continue
        updated_column = dict(column)
        updated_column["source_fields"] = [replacement_field]
        updated_columns.append(updated_column)
        replacement_used = True
        changed = True

    if not changed:
        return request
    updated_output_spec = dict(output_spec_after_demote)
    updated_output_spec["columns"] = updated_columns
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "output_spec": updated_output_spec,
            },
        }
    )


def _observed_field_case(
    *,
    alias: str,
    arguments: Mapping[str, Any],
) -> str | None:
    evidence = arguments.get("evidence")
    if not isinstance(evidence, Mapping):
        return None
    for source in evidence.get("context_sources") or []:
        if not isinstance(source, Mapping):
            continue
        for observation in source.get("observations") or []:
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(observation)):
                if _normalized_field_alias(token) == alias:
                    return token
    return None


def _observed_field_case_from_state(
    *,
    alias: str,
    state: Mapping[str, Any],
) -> str | None:
    if not alias:
        return None
    for source in _state_observed_sources(state):
        fields = source.get("fields")
        if not isinstance(fields, list):
            continue
        for item in fields:
            field = item.get("name") if isinstance(item, Mapping) else item
            text = str(field or "").strip()
            if text and _normalized_field_alias(text) == alias:
                return text
    return None


def _canonicalize_single_preserve_output_from_field_fact(
    request: ToolCallRequest,
    discovery: _DiscoveryState,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return request
    output_columns = output_spec.get("columns") or []
    if not isinstance(output_columns, list) or len(output_columns) != 1:
        return request
    if str(output_spec.get("row_policy") or "") != "preserve":
        return request
    if output_spec.get("transformations"):
        return request
    execution_spec = arguments.get("execution_spec")
    if isinstance(execution_spec, Mapping) and execution_spec.get("operations"):
        return request
    evidence = arguments.get("evidence")
    if not isinstance(evidence, Mapping):
        return request
    if str(evidence.get("knowledge_status") or "") != "authoritative":
        return request

    field_facts = [
        fact
        for fact in parse_knowledge_content(discovery.knowledge_content)
        if _knowledge_fact_defines_field(fact)
    ]
    facts_by_id = {str(fact.fact_id): fact for fact in field_facts}
    facts_by_quote: dict[str, list[Any]] = {}
    for fact in field_facts:
        facts_by_quote.setdefault(str(fact.quote), []).append(fact)

    cited_facts: list[Any] = []
    for rule in evidence.get("knowledge_rules") or []:
        if not isinstance(rule, Mapping):
            continue
        fact_id = str(rule.get("fact_id") or "").strip()
        quote = str(rule.get("quote") or "").strip()
        if fact_id and fact_id in facts_by_id:
            cited_facts.append(facts_by_id[fact_id])
        elif quote:
            cited_facts.extend(
                facts_by_quote.get(quote)
                or _field_facts_for_knowledge_quote(
                    quote=quote,
                    field_facts=field_facts,
                )
            )
    target_aliases = {
        alias
        for fact in cited_facts
        if _field_fact_has_observed_physical_field(fact=fact, state=request.state)
        for alias in _knowledge_fact_field_aliases(fact)
    }
    if len(target_aliases) != 1:
        return request
    target_alias = next(iter(target_aliases))
    column = output_columns[0]
    if not isinstance(column, Mapping):
        return request
    if _column_field_aliases(column) & target_aliases:
        return request
    requested_texts = _question_requested_output_texts(request.state)
    if _column_matches_requested_output(column, requested_texts):
        return request

    target_field = (
        _observed_field_case(alias=target_alias, arguments=arguments)
        or _observed_field_case_from_state(alias=target_alias, state=request.state)
        or target_alias
    )
    updated_column = dict(column)
    updated_column["name"] = target_field
    updated_column["source_fields"] = [target_field]
    role = str(updated_column.get("role") or "").strip().casefold()
    if role in {"entity_key", "record_key", "time_key", "context"}:
        updated_column["role"] = "measure"

    updated_output_spec = dict(output_spec)
    updated_output_spec["columns"] = [updated_column]
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "output_spec": updated_output_spec,
            },
        }
    )


def _canonicalize_output_columns_from_section_field_facts(
    request: ToolCallRequest,
    discovery: _DiscoveryState,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    execution_spec = arguments.get("execution_spec")
    if not isinstance(output_spec, Mapping):
        return request
    output_columns = output_spec.get("columns") or []
    if not isinstance(output_columns, list) or not output_columns:
        return request
    evidence = arguments.get("evidence")
    if not isinstance(evidence, Mapping):
        return request

    field_facts = [
        fact
        for fact in parse_knowledge_content(discovery.knowledge_content)
        if _knowledge_fact_defines_field(fact)
    ]
    facts_by_id = {str(fact.fact_id): fact for fact in field_facts}
    facts_by_quote: dict[str, list[Any]] = {}
    for fact in field_facts:
        facts_by_quote.setdefault(str(fact.quote), []).append(fact)

    cited_facts: list[Any] = []
    for rule in evidence.get("knowledge_rules") or []:
        if not isinstance(rule, Mapping):
            continue
        fact_id = str(rule.get("fact_id") or "").strip()
        quote = str(rule.get("quote") or "").strip()
        if fact_id and fact_id in facts_by_id:
            cited_facts.append(facts_by_id[fact_id])
        elif quote:
            cited_facts.extend(facts_by_quote.get(quote, []))
    cited_sections = {
        str(getattr(fact, "section_key", "") or "")
        for fact in cited_facts
        if str(getattr(fact, "section_key", "") or "").strip()
    }
    if not cited_sections:
        return request
    observed_sections = _observed_section_keys_from_plan_arguments(
        arguments,
        field_facts=field_facts,
    )
    active_sections = cited_sections
    if observed_sections:
        observed_cited_sections = cited_sections & observed_sections
        if observed_cited_sections:
            active_sections = observed_cited_sections

    active_cited_facts = [
        fact
        for fact in cited_facts
        if str(getattr(fact, "section_key", "") or "") in active_sections
        and _field_fact_has_observed_physical_field(
            fact=fact,
            state=request.state,
        )
    ]
    if not active_cited_facts:
        return request
    used_aliases = {
        alias
        for fact in active_cited_facts
        for alias in _knowledge_fact_field_aliases(fact)
    }
    candidates = [
        fact
        for fact in field_facts
        if str(getattr(fact, "section_key", "") or "") in active_sections
        and _field_fact_has_observed_physical_field(
            fact=fact,
            state=request.state,
        )
        and not (_knowledge_fact_field_aliases(fact) & used_aliases)
    ]
    if len(candidates) != 1:
        return request
    candidate_aliases = _knowledge_fact_field_aliases(candidates[0])
    if not candidate_aliases:
        return request
    replacement_alias = sorted(candidate_aliases, key=lambda item: (len(item), item))[0]
    replacement_field = (
        _observed_field_case(alias=replacement_alias, arguments=arguments)
        or str(getattr(candidates[0], "field_key", "") or "")
    )
    if not replacement_field:
        return request

    selector_aliases = _selector_aliases_from_output_spec(
        output_spec,
        execution_spec if isinstance(execution_spec, Mapping) else None,
    )
    updated_columns: list[Any] = []
    changed = False
    replacement_used = False
    for column in output_columns:
        if not isinstance(column, Mapping):
            updated_columns.append(column)
            continue
        column_aliases = _column_field_aliases(column)
        if column_aliases & (used_aliases | candidate_aliases | selector_aliases):
            source_aliases = {
                _normalized_field_alias(field)
                for field in column.get("source_fields") or []
                if _normalized_field_alias(field)
            }
            name_alias = _normalized_field_alias(column.get("name"))
            if (
                source_aliases & selector_aliases
                and name_alias not in selector_aliases
                and not replacement_used
            ):
                updated_column = dict(column)
                updated_column["source_fields"] = [replacement_field]
                updated_columns.append(updated_column)
                replacement_used = True
                changed = True
                continue
            updated_columns.append(column)
            continue
        if replacement_used:
            updated_columns.append(column)
            continue
        updated_column = dict(column)
        updated_column["source_fields"] = [replacement_field]
        updated_columns.append(updated_column)
        replacement_used = True
        changed = True
    if not changed:
        return request
    updated_output_spec = dict(output_spec)
    updated_output_spec["columns"] = updated_columns
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "output_spec": updated_output_spec,
            },
        }
    )


def _source_paths_from_execution_spec(execution_spec: Any) -> set[str]:
    if not isinstance(execution_spec, Mapping):
        return set()
    return {
        str(source.get("path") or "").replace("\\", "/")
        for source in execution_spec.get("sources") or []
        if isinstance(source, Mapping) and str(source.get("path") or "").strip()
    }


def _source_paths_from_plan_arguments(arguments: Mapping[str, Any]) -> set[str]:
    paths: set[str] = set()
    evidence = arguments.get("evidence")
    if isinstance(evidence, Mapping):
        paths.update(
            str(source.get("path") or "").replace("\\", "/")
            for source in evidence.get("context_sources") or []
            if isinstance(source, Mapping)
            and str(source.get("path") or "").strip()
        )
    execution_spec = arguments.get("execution_spec")
    if isinstance(execution_spec, str):
        try:
            decoded = json.loads(execution_spec)
        except json.JSONDecodeError:
            decoded = {}
        execution_spec = decoded
    paths.update(_source_paths_from_execution_spec(execution_spec))
    if isinstance(execution_spec, Mapping):
        for binding in execution_spec.get("source_bindings") or []:
            if not isinstance(binding, Mapping):
                continue
            paths.update(
                str(path or "").replace("\\", "/")
                for path in binding.get("source_paths") or []
                if str(path or "").strip()
            )
    return {path for path in paths if path}


def _observed_section_keys_from_plan_arguments(
    arguments: Mapping[str, Any],
    *,
    field_facts: list[Any],
) -> set[str]:
    paths = _source_paths_from_plan_arguments(arguments)
    if not paths:
        return set()
    normalized_paths = {
        _normalized_quote_text(path.replace("/", " ").replace("\\", " "))
        for path in paths
    }
    observed_sections: set[str] = set()
    for fact in field_facts:
        section_key = str(getattr(fact, "section_key", "") or "").strip()
        if not section_key:
            continue
        normalized_section = _normalized_quote_text(section_key)
        if not normalized_section:
            continue
        if any(normalized_section in normalized_path for normalized_path in normalized_paths):
            observed_sections.add(section_key)
    return observed_sections


def _canonicalize_semantic_source_bindings(
    request: ToolCallRequest,
    discovery: _DiscoveryState,
) -> ToolCallRequest:
    """Never turn knowledge-document field definitions into source bindings.

    A markdown/PDF knowledge table can name semantic targets and source-name
    hints, but it is not evidence that a real data source has those columns.
    Source bindings for narrative documents are extraction targets supplied by
    the plan/model; they become field evidence only after a mechanical
    extractor reports extracted fields and line evidence.
    """

    return request


def _canonicalize_revision(request: ToolCallRequest) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    previous_plan = request.state.get("analysis_plan")
    revision = dict(arguments.get("revision") or {})
    if previous_plan is None:
        revision["version"] = 1
        revision["changed_fields"] = []
        revision["evidence_changes"] = []
    else:
        previous_version = (previous_plan.get("revision") or {}).get("version")
        revision["version"] = (
            previous_version + 1
            if isinstance(previous_version, int)
            else 2
        )
        actual_changes = sorted(
            field
            for field in _REVISION_FIELDS
            if previous_plan.get(field) != arguments.get(field)
        )
        revision["changed_fields"] = actual_changes
        if "evidence" in actual_changes and not revision.get("evidence_changes"):
            revision["evidence_changes"] = [
                "Evidence changed in the revised analysis plan."
            ]
    if revision == arguments.get("revision"):
        return request
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "revision": revision,
            },
        }
    )


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if not content:
        return ""
    return "\n".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
    )


def _injected_knowledge_content(messages: list[BaseMessage]) -> str | None:
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        match = _INJECTED_KNOWLEDGE_PATTERN.search(_message_text(message))
        if match is None:
            continue
        content = match.group(1).strip()
        if content in {"", "<missing>", "<empty>"} or content.startswith("<unreadable:"):
            return None
        return content
    return None


def _normalized_quote_text(value: str) -> str:
    """Normalize markdown and punctuation for exact quote localization."""

    return " ".join(
        re.sub(r"[^\w\u4e00-\u9fff]+", " ", value, flags=re.UNICODE)
        .casefold()
        .split()
    )


def _normalized_quote_tokens(value: str) -> set[str]:
    return {
        token
        for token in _normalized_quote_text(value).split()
        if token not in {"column", "definition", "field", "semantic", "unit"}
    }


_CONTRACT_TERM_STOPWORDS = frozenset(
    {
        "and",
        "column",
        "data",
        "database",
        "definition",
        "field",
        "fields",
        "record",
        "records",
        "semantic",
        "source",
        "table",
        "unit",
        "value",
        "values",
        "with",
    }
)


def _contract_terms(value: Any) -> set[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    terms: set[str] = set()
    for token in re.findall(r"[0-9a-z_]{2,}", text):
        parts = [part for part in token.split("_") if part]
        for part in [token, *parts]:
            if len(part) > 3 and part.endswith("ies"):
                part = f"{part[:-3]}y"
            elif len(part) > 3 and part.endswith("s"):
                part = part[:-1]
            if part and part not in _CONTRACT_TERM_STOPWORDS:
                terms.add(part)
    for sequence in re.findall(r"[\u3400-\u9fff]+", text):
        if len(sequence) < 2:
            continue
        terms.add(sequence)
        for size in (2, 3, 4):
            if len(sequence) < size:
                continue
            terms.update(
                sequence[index : index + size]
                for index in range(0, len(sequence) - size + 1)
            )
    return terms


def _source_hints_from_knowledge_text(text: Any) -> set[str]:
    raw_text = str(text or "")
    hints: set[str] = set()
    for match in re.finditer(
        r"\b(?:FROM|JOIN)\s+[`\"\[]?([A-Za-z][A-Za-z0-9_]*)",
        raw_text,
        flags=re.IGNORECASE,
    ):
        hints.add(match.group(1))
    for match in re.finditer(r"`([A-Za-z][A-Za-z0-9_]*)\.[^`]+`", raw_text):
        hints.add(match.group(1))
    for match in re.finditer(
        r"\b([A-Za-z][A-Za-z0-9_]*)\.[A-Za-z][A-Za-z0-9_]*",
        raw_text,
    ):
        hints.add(match.group(1))
    return {hint for hint in hints if hint}


def _request_focus_text(state: Mapping[str, Any]) -> str:
    question_structure = state.get("question_structure")
    structure_text = ""
    if isinstance(question_structure, Mapping):
        structure_text = json.dumps(
            question_structure,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    return "\n".join(
        text
        for text in (
            str(state.get("original_request") or ""),
            structure_text,
        )
        if text.strip()
    )


def _request_relevant_knowledge_source_hint_groups(
    *,
    knowledge_facts: Iterable[Any],
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    focus_terms = _contract_terms(_request_focus_text(state))
    if not focus_terms:
        return []

    groups: list[dict[str, Any]] = []
    known_sections = {
        str(getattr(fact, "section_key", "") or "").strip()
        for fact in knowledge_facts
        if str(getattr(fact, "section_key", "") or "").strip()
    }
    for fact in knowledge_facts:
        fact_text = " ".join(
            str(item or "")
            for item in (
                getattr(fact, "section_key", ""),
                getattr(fact, "field_key", ""),
                getattr(fact, "operation", ""),
                getattr(fact, "quote", ""),
            )
        )
        overlap = _contract_terms(fact_text) & focus_terms
        if len(overlap) < 2:
            continue
        source_hints = set()
        section_key = str(getattr(fact, "section_key", "") or "").strip()
        if section_key:
            source_hints.add(section_key)
        source_hints.update(_source_hints_from_knowledge_text(getattr(fact, "quote", "")))
        source_hints = {
            hint
            for hint in source_hints
            if hint in known_sections or not known_sections
        }
        if not source_hints:
            continue
        groups.append(
            {
                "fact_id": str(getattr(fact, "fact_id", "") or ""),
                "kind": str(getattr(fact, "kind", "") or ""),
                "score": len(overlap),
                "matched_terms": sorted(overlap)[:12],
                "source_hints": sorted(source_hints),
                "quote": str(getattr(fact, "quote", "") or "")[:240],
            }
        )
    if not groups:
        return []
    max_score = max(int(group["score"]) for group in groups)
    return [
        group
        for group in sorted(
            groups,
            key=lambda item: (
                -int(item["score"]),
                str(item.get("fact_id") or ""),
            ),
        )
        if int(group["score"]) == max_score
    ][:3]


def _source_path_matches_hint(path: str, source_hint: str) -> bool:
    normalized_hint = _normalized_quote_text(source_hint)
    if not normalized_hint:
        return False
    normalized_path = path.replace("\\", "/")
    path_tail = normalized_path.rsplit("::", 1)[-1].rsplit("/", 1)[-1]
    path_stem = path_tail.rsplit(".", 1)[0]
    candidates = {
        _normalized_quote_text(path_tail),
        _normalized_quote_text(path_stem),
    }
    if "::" in normalized_path:
        candidates.add(_normalized_quote_text(normalized_path.rsplit("::", 1)[-1]))
    return normalized_hint in candidates


def _plan_sources_match_hint(
    *,
    plan_sources: set[str],
    source_hint: str,
    state: Mapping[str, Any],
) -> bool:
    if any(_source_path_matches_hint(path, source_hint) for path in plan_sources):
        return True
    normalized_plan_sources = {
        path.replace("\\", "/")
        for path in plan_sources
        if str(path or "").strip()
    }
    normalized_hint = _normalized_quote_text(source_hint)
    if not normalized_hint:
        return False
    for source in _state_observed_sources(state):
        source_paths = _source_mapping_paths(source)
        base_path = str(source.get("base_path") or "").replace("\\", "/")
        if base_path:
            source_paths.add(base_path)
        if not (source_paths & normalized_plan_sources):
            continue
        source_hints = {
            _normalized_quote_text(str(source.get("source_name_hint") or "")),
            _normalized_quote_text(str(source.get("table") or "")),
        }
        for path in source_paths:
            path_tail = path.rsplit("::", 1)[-1].rsplit("/", 1)[-1]
            source_hints.add(_normalized_quote_text(path_tail.rsplit(".", 1)[0]))
        if normalized_hint in {hint for hint in source_hints if hint}:
            return True
    return False


def _missing_relevant_source_hint_groups(
    *,
    knowledge_facts: Iterable[Any],
    state: Mapping[str, Any],
    plan_sources: set[str],
) -> list[dict[str, Any]]:
    relevant_source_hint_groups = _request_relevant_knowledge_source_hint_groups(
        knowledge_facts=knowledge_facts,
        state=state,
    )
    if not relevant_source_hint_groups:
        return []
    missing_by_group: list[dict[str, Any]] = []
    for group in relevant_source_hint_groups:
        missing_hints = [
            hint
            for hint in group.get("source_hints") or []
            if not _plan_sources_match_hint(
                plan_sources=plan_sources,
                source_hint=str(hint),
                state=state,
            )
        ]
        if not missing_hints:
            return []
        missing_by_group.append(
            {
                "fact_id": group.get("fact_id"),
                "source_hints": group.get("source_hints"),
                "missing": missing_hints,
            }
        )
    return missing_by_group


def _source_hint_has_observed_source(
    *,
    state: Mapping[str, Any],
    source_hint: str,
) -> bool:
    normalized_hint = _normalized_quote_text(source_hint)
    if not normalized_hint:
        return False
    if _observed_sources_by_source_hint(state).get(normalized_hint):
        return True
    for source in _state_observed_sources(state):
        source_hints = {
            _normalized_quote_text(str(source.get("source_name_hint") or "")),
            _normalized_quote_text(str(source.get("table") or "")),
        }
        for path in _source_mapping_paths(source):
            path_tail = path.rsplit("::", 1)[-1].rsplit("/", 1)[-1]
            source_hints.add(_normalized_quote_text(path_tail.rsplit(".", 1)[0]))
            if _source_path_matches_hint(path, source_hint):
                return True
        if normalized_hint in {hint for hint in source_hints if hint}:
            return True
    return False


def _condition_requirement_type(container_name: str) -> str:
    return {
        "filters": "filter",
        "time_ranges": "time_range",
        "groupings": "grouping",
        "orderings": "ordering",
        "limits": "limit",
        "calculations": "calculation",
        "output_columns": "output_column",
    }.get(container_name, container_name.rstrip("s"))


def _constraint_requirement_type(constraint: Mapping[str, Any]) -> str:
    constraint_type = str(constraint.get("constraint_type") or "").strip()
    selector_text = (
        f"{constraint.get('quote') or ''} {constraint.get('value') or ''}"
    )
    if constraint_type in {"filter", "ordering"} and _SELECTOR_EXPRESSION_PATTERN.search(
        selector_text
    ):
        return "selector"
    return {
        "filter": "filter",
        "equality": "filter",
        "aggregation_equality": "filter",
        "value": "value",
        "ranking": "filter",
        "row_filter": "filter",
        "time_range": "time_range",
        "grouping": "grouping",
        "ordering": "ordering",
        "ascending_sort": "ordering",
        "descending_sort": "ordering",
        "limit": "limit",
        "top_k": "limit",
        "selector": "selector",
        "calculation": "calculation",
        "aggregate_min": "calculation",
        "aggregate_max": "calculation",
        "aggregate_sum": "calculation",
        "aggregate_avg": "calculation",
        "aggregate_average": "calculation",
        "aggregate_count": "calculation",
        "min_aggregation": "calculation",
        "max_aggregation": "calculation",
        "sum_aggregation": "calculation",
        "avg_aggregation": "calculation",
        "average_aggregation": "calculation",
        "count_aggregation": "calculation",
        "aggregation": "calculation",
        "deduplication": "deduplication",
        "reshape": "reshape",
        "output_shape": "reshape",
        "entity": "entity",
        "geography": "scope",
        "scope": "scope",
        "field": "scope",
        "table_scope": "scope",
        "source_scope": "scope",
    }.get(constraint_type, "scope")


def _target_requirement_type(target_type: str) -> str:
    return {
        "record_set": "output",
        "metric": "measure",
        "field": "output_column",
    }.get(target_type, target_type)


def _quote_type_pairs_from_question_structure(
    question_structure: Any,
) -> list[tuple[str, str]]:
    if not isinstance(question_structure, Mapping):
        return []
    pairs: list[tuple[str, str]] = []
    for target in question_structure.get("targets") or []:
        if not isinstance(target, Mapping):
            continue
        quote = str(target.get("quote") or "").strip()
        target_type = str(target.get("target_type") or "").strip()
        if quote and target_type:
            pairs.append((quote, _target_requirement_type(target_type)))
            if _SELECTOR_EXPRESSION_PATTERN.search(quote):
                pairs.append((quote, "selector"))
    for constraint in question_structure.get("target_constraints") or []:
        if not isinstance(constraint, Mapping):
            continue
        quote = str(constraint.get("quote") or "").strip()
        if not quote:
            continue
        explicitness = str(constraint.get("explicitness") or "").strip()
        if explicitness == "explicit":
            requirement_type = _constraint_requirement_type(constraint)
        else:
            requirement_type = "scope"
        pairs.append((quote, requirement_type))
    conditions = question_structure.get("conditions")
    if isinstance(conditions, Mapping):
        for container_name, items in conditions.items():
            if not isinstance(items, list):
                continue
            container_requirement_type = _condition_requirement_type(str(container_name))
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                quote = str(item.get("quote") or "").strip()
                if quote:
                    item_type = str(
                        item.get("condition_type")
                        or item.get("constraint_type")
                        or ""
                    ).strip()
                    requirement_type = (
                        _constraint_requirement_type(
                            {
                                "constraint_type": item_type,
                                "quote": quote,
                                "value": item.get("value"),
                            }
                        )
                        if item_type
                        else container_requirement_type
                    )
                    pairs.append((quote, requirement_type))
    return pairs


def _question_structure_exact_types_for_quote(
    question_structure: Any,
    quote: str,
) -> set[str]:
    if not isinstance(question_structure, Mapping):
        return set()
    normalized_quote = _normalized_quote_text(quote)
    exact_types = {
        requirement_type
        for item_quote, requirement_type in _quote_type_pairs_from_question_structure(
            question_structure
        )
        if item_quote == quote
        or _normalized_quote_text(item_quote) == normalized_quote
    }
    return {item for item in exact_types if item}


def _requirement_types_for_quote(
    *,
    quote: str,
    arguments: Mapping[str, Any],
    state: Mapping[str, Any],
) -> set[str]:
    stripped = quote.strip()
    if not stripped:
        return set()
    question_structure = state.get("question_structure")
    if state.get("question_structure_enforced"):
        exact_types = _question_structure_exact_types_for_quote(
            question_structure,
            stripped,
        )
        return exact_types
    pairs: list[tuple[str, str]] = []
    intent = arguments.get("intent")
    if isinstance(intent, Mapping):
        for item in intent.get("requirements") or []:
            if not isinstance(item, Mapping):
                continue
            item_quote = str(item.get("quote") or "").strip()
            requirement_type = str(item.get("requirement_type") or "").strip()
            if item_quote and requirement_type:
                pairs.append((item_quote, requirement_type))
    pairs.extend(_quote_type_pairs_from_question_structure(question_structure))
    types: set[str] = set()
    normalized_quote = _normalized_quote_text(stripped)
    for item_quote, requirement_type in pairs:
        normalized_item = _normalized_quote_text(item_quote)
        if not normalized_item:
            continue
        if (
            item_quote in stripped
            or stripped in item_quote
            or normalized_item in normalized_quote
            or normalized_quote in normalized_item
        ):
            types.add(requirement_type)
    return types


def _effective_requirement_types_by_quote(
    raw_requirement_types_by_quote: Mapping[str, set[str]],
    state: Mapping[str, Any],
) -> dict[str, set[str]]:
    if not state.get("question_structure_enforced"):
        return {quote: set(types) for quote, types in raw_requirement_types_by_quote.items()}
    question_structure = state.get("question_structure")
    effective: dict[str, set[str]] = {}
    for quote, types in raw_requirement_types_by_quote.items():
        del types
        exact_types = _question_structure_exact_types_for_quote(
            question_structure,
            quote,
        )
        effective[quote] = exact_types
    return effective


def _condition_quote_is_explicit(item: Mapping[str, Any], original_request: str) -> bool:
    quote = str(item.get("quote") or "").strip()
    if not quote or quote not in original_request:
        return False
    explicitness = str(item.get("explicitness") or "").strip()
    return explicitness not in {"ambiguous", "unquoted_hint"}


def _question_structure_authorized_operations_by_quote(
    question_structure: Any,
    original_request: str,
) -> dict[str, set[str]]:
    if not isinstance(question_structure, Mapping):
        return {}
    operations_by_quote: dict[str, set[str]] = {}
    for quote, requirement_type in _quote_type_pairs_from_question_structure(
        question_structure
    ):
        if not quote or quote not in original_request:
            continue
        operations = _REQUIREMENT_TYPE_OPERATIONS.get(requirement_type, frozenset())
        if not operations:
            continue
        operations_by_quote.setdefault(quote, set()).update(operations)
    for item in question_structure.get("intent_operators") or []:
        if not isinstance(item, Mapping):
            continue
        quote = str(item.get("quote") or "").strip()
        operation = str(item.get("operation") or "").strip()
        if (
            quote
            and quote in original_request
            and operation in _TRANSFORM_OPERATIONS
        ):
            operations_by_quote.setdefault(quote, set()).add(operation)
    return operations_by_quote


def _question_structure_user_authorized_operations(
    question_structure: Any,
    original_request: str,
) -> set[str]:
    if not isinstance(question_structure, Mapping):
        return set()

    operations_by_quote = _question_structure_authorized_operations_by_quote(
        question_structure,
        original_request,
    )
    authorized: set[str] = {
        operation
        for operations in operations_by_quote.values()
        for operation in operations
    }
    for item in question_structure.get("intent_operators") or []:
        if not isinstance(item, Mapping):
            continue
        quote = str(item.get("quote") or "").strip()
        operation = str(item.get("operation") or "").strip()
        if (
            quote
            and quote in original_request
            and operation in operations_by_quote.get(quote, set())
            and operation in _TRANSFORM_OPERATIONS
        ):
            authorized.add(operation)

    return authorized


def _question_structure_scope_constraints(question_structure: Any) -> list[dict[str, str]]:
    if not isinstance(question_structure, Mapping):
        return []
    constraints: list[dict[str, str]] = []
    for item in question_structure.get("target_constraints") or []:
        if not isinstance(item, Mapping):
            continue
        quote = str(item.get("quote") or "").strip()
        if not quote:
            continue
        constraints.append(
            {
                "quote": quote,
                "constraint_type": str(item.get("constraint_type") or "").strip(),
                "explicitness": str(item.get("explicitness") or "").strip(),
                "value": str(item.get("value") or "").strip(),
            }
        )
    return constraints[:12]


def _unresolved_explicit_scope_quotes(
    *,
    question_structure: Any,
    arguments: Mapping[str, Any],
) -> list[str]:
    issue_parts: list[str] = []
    intent = arguments.get("intent")
    if isinstance(intent, Mapping):
        issue_parts.extend(str(item or "") for item in intent.get("unresolved") or [])
    evidence = arguments.get("evidence")
    if isinstance(evidence, Mapping):
        issue_parts.append(str(evidence.get("knowledge_issue") or ""))
        issue_parts.append(str(evidence.get("cross_validated_inference") or ""))
    issue_text = _normalized_quote_text(" ".join(issue_parts))
    if not issue_text:
        return []
    unresolved_markers = {
        "cannot",
        "unclear",
        "unresolved",
        "not defined",
        "not specified",
        "no clear",
        "无法",
        "没有",
        "不存在",
        "未命中",
        "未定义",
        "不明确",
        "未指定",
    }
    if not any(marker in issue_text for marker in unresolved_markers):
        return []
    output_spec = arguments.get("output_spec")
    if not isinstance(output_spec, Mapping):
        output_spec = {}
    execution_spec = arguments.get("execution_spec")
    if not isinstance(execution_spec, Mapping):
        execution_spec = {}
    plan_preserves_source = (
        str(output_spec.get("row_policy") or "").strip().casefold() == "preserve"
        and not output_spec.get("transformations")
        and not execution_spec.get("operations")
    )
    observed_conflict_markers = {
        "no direct",
        "no matching",
        "not found",
        "does not contain",
        "without",
        "不存在",
        "没有",
        "无",
        "未命中",
        "无法直接",
        "粒度",
        "分省",
        "省份",
    }

    unresolved_quotes: list[str] = []
    for constraint in _question_structure_scope_constraints(question_structure):
        if constraint.get("explicitness") != "explicit":
            continue
        selector_text = (
            f"{constraint.get('quote') or ''} {constraint.get('value') or ''}"
        )
        if _SELECTOR_EXPRESSION_PATTERN.search(selector_text):
            continue
        if constraint.get("constraint_type") not in {
            "entity",
            "filter",
            "geography",
            "scope",
            "time_range",
        }:
            continue
        quote = str(constraint.get("quote") or "").strip()
        value = str(constraint.get("value") or "").strip()
        normalized_terms = [
            _normalized_quote_text(term)
            for term in (quote, value)
            if str(term or "").strip()
        ]
        if not any(term and term in issue_text for term in normalized_terms):
            continue
        if plan_preserves_source and any(
            marker in issue_text for marker in observed_conflict_markers
        ):
            continue
        if any(term and term in issue_text for term in normalized_terms):
            unresolved_quotes.append(quote)
    return list(dict.fromkeys(unresolved_quotes))


def _unresolved_binding_issues(intent: Mapping[str, Any]) -> list[str]:
    markers = {
        "substitute",
        "replacement",
        "replace",
        "instead",
        "not exist",
        "does not exist",
        "not found",
        "missing",
        "unavailable",
        "unresolved",
        "binding",
        "无法",
        "不存在",
        "缺失",
        "未找到",
        "未解决",
        "替代",
        "代替",
        "绑定",
    }
    issues: list[str] = []
    for item in intent.get("unresolved") or []:
        text = str(item or "").strip()
        normalized = _normalized_quote_text(text)
        if text and any(marker in normalized for marker in markers):
            issues.append(text)
    return issues


def _evidence_contract_payload(
    state: Mapping[str, Any],
    discovery: _DiscoveryState,
) -> dict[str, Any]:
    original_request = str(state.get("original_request") or "")
    question_structure = state.get("question_structure")
    authorized_operations: list[dict[str, Any]] = []
    if isinstance(question_structure, Mapping):
        for item in question_structure.get("intent_operators") or []:
            if not isinstance(item, Mapping):
                continue
            quote = str(item.get("quote") or "").strip()
            operation = str(item.get("operation") or "").strip()
            if quote and quote in original_request and operation:
                authorized_operations.append(
                    {
                        "operation": operation,
                        "source": "user",
                        "quote": quote,
                    }
                )
        conditions = question_structure.get("conditions")
        for quote, operations in _question_structure_authorized_operations_by_quote(
            question_structure,
            original_request,
        ).items():
            for operation in sorted(operations):
                authorized_operations.append(
                    {
                        "operation": operation,
                        "source": "user",
                        "quote": quote,
                    }
                )

    seen_operations: set[tuple[str, str, str]] = set()
    unique_operations: list[dict[str, Any]] = []
    for item in authorized_operations:
        key = (
            str(item.get("operation") or ""),
            str(item.get("source") or ""),
            str(item.get("quote") or ""),
        )
        if key in seen_operations:
            continue
        seen_operations.add(key)
        unique_operations.append(item)

    observed_evidence: list[dict[str, Any]] = []
    for source in _state_observed_sources(state):
        path = str(source.get("path") or "").replace("\\", "/")
        if not path:
            continue
        fields = source.get("fields")
        matched_lines = source.get("matched_lines")
        has_extracted_fields = (
            _observed_source_is_narrative(source)
            and str(source.get("observed_by") or "").strip()
            in _NARRATIVE_FIELD_EVIDENCE_TOOLS
        )
        observed_evidence.append(
            {
                "path": path,
                "field": (
                    fields[:8]
                    if isinstance(fields, list)
                    and (_observed_source_is_structured(source) or has_extracted_fields)
                    else None
                ),
                "value": None,
                "evidence_type": (
                    "narrative_extraction"
                    if has_extracted_fields
                    else
                    "line"
                    if isinstance(matched_lines, list) and matched_lines
                    else "schema"
                ),
                "line_number": (
                    matched_lines[0].get("line_number")
                    if isinstance(matched_lines, list)
                    and matched_lines
                    and isinstance(matched_lines[0], Mapping)
                    else None
                ),
                "sample_hash": source.get("sample_hash"),
            }
        )

    candidate_answers: list[dict[str, Any]] = []
    candidate = state.get("answer_candidate")
    if isinstance(candidate, Mapping):
        candidate_audit = candidate.get("audit")
        candidate_source = candidate.get("source")
        if not candidate_source and isinstance(candidate_audit, Mapping):
            candidate_source = candidate_audit.get("audit_origin")
        candidate_answers.append(
            {
                "columns": candidate.get("columns"),
                "rows": candidate.get("rows"),
                "audit": candidate_audit,
                "source": candidate_source or "answer_candidate",
            }
        )
    knowledge_source_hints = _request_relevant_knowledge_source_hint_groups(
        knowledge_facts=parse_knowledge_content(discovery.knowledge_content),
        state=state,
    )

    return {
        "authorized_operations": unique_operations,
        "scope_constraints": [
            {
                "kind": item.get("constraint_type"),
                "quote": item.get("quote"),
                "status": "unresolved",
                "evidence": [],
            }
            for item in _question_structure_scope_constraints(question_structure)
        ],
        "observed_evidence": [
            {
                key: value
                for key, value in item.items()
                if value is not None and value != []
            }
            for item in observed_evidence[:12]
        ],
        "candidate_answers": candidate_answers,
        "knowledge_source_hints": knowledge_source_hints,
        "knowledge_available": discovery.knowledge_available,
    }


def _question_structure_preserve_hint(question_structure: Any) -> bool:
    if not isinstance(question_structure, Mapping):
        return False
    output = question_structure.get("output")
    if not isinstance(output, Mapping):
        return False
    return (
        str(output.get("row_grain_hint") or "") == "source_records"
        and str(output.get("preserve_source_rows") or "") == "true"
    )


def _evidence_boundary_source_summary(source: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(source.get("path") or "").replace("\\", "/"),
        "source_type": str(source.get("source_type") or "").strip(),
    }
    for key in ("row_count", "line_count", "table", "selected_path"):
        value = source.get(key)
        if value is not None and value != "":
            summary[key] = value
    fields = source.get("fields")
    if isinstance(fields, list):
        summary["fields"] = [str(field) for field in fields[:16]]
        if len(fields) > 16:
            summary["field_count"] = len(fields)
    return {
        key: value
        for key, value in summary.items()
        if value != "" and value != []
    }


def _active_plan_boundary_summary(state: Mapping[str, Any]) -> dict[str, Any] | None:
    analysis_plan = state.get("analysis_plan")
    if not isinstance(analysis_plan, Mapping):
        return None
    output_spec = analysis_plan.get("output_spec")
    execution_spec = analysis_plan.get("execution_spec")
    summary: dict[str, Any] = {}
    if isinstance(output_spec, Mapping):
        summary["row_policy"] = str(output_spec.get("row_policy") or "")
        summary["row_grain"] = str(output_spec.get("row_grain") or "")
        summary["columns"] = [
            {
                "name": str(column.get("name") or ""),
                "source_fields": [
                    str(field)
                    for field in column.get("source_fields") or []
                    if str(field).strip()
                ],
            }
            for column in output_spec.get("columns") or []
            if isinstance(column, Mapping)
        ][:12]
        transformations = [
            str(item.get("operation") or "")
            for item in output_spec.get("transformations") or []
            if isinstance(item, Mapping) and str(item.get("operation") or "").strip()
        ]
        summary["transformations"] = transformations
    if isinstance(execution_spec, Mapping):
        summary["execution_sources"] = sorted(_source_paths_from_execution_spec(execution_spec))
        summary["execution_operations"] = [
            str(item.get("operation") or "")
            for item in execution_spec.get("operations") or []
            if isinstance(item, Mapping) and str(item.get("operation") or "").strip()
        ]
    return {
        key: value
        for key, value in summary.items()
        if value != "" and value != []
    }


def _decision_evidence_boundary_payload(
    state: Mapping[str, Any],
    discovery: _DiscoveryState,
) -> dict[str, Any]:
    original_request = str(state.get("original_request") or "")
    question_structure = state.get("question_structure")
    user_authorized_operations = _question_structure_user_authorized_operations(
        question_structure,
        original_request,
    )
    forbidden_operations = sorted(_TRANSFORM_OPERATIONS - user_authorized_operations)
    source_summaries = [
        _evidence_boundary_source_summary(source)
        for source in _state_observed_sources(state)
    ]
    source_summaries = [
        source for source in source_summaries if source.get("path")
    ][:8]
    preserve_hint = _question_structure_preserve_hint(question_structure)

    return {
        "schema_version": "1.0",
        "purpose": (
            "Runtime authorization and evidence boundary. Use it for every "
            "decision, including discovery, analyze_plan, revisions, and execution."
        ),
        "evidence_contract": _evidence_contract_payload(state, discovery),
        "discovery_ready": discovery.context_ready,
        "observed_sources": source_summaries,
        "active_plan": _active_plan_boundary_summary(state),
        "user_authorized_operations": sorted(user_authorized_operations),
        "forbidden_without_user_or_knowledge_authorization": forbidden_operations,
        "scope_constraints_are_evidence_not_authorization": (
            _question_structure_scope_constraints(question_structure)
        ),
        "preserve_source_rows_hint": preserve_hint,
        "rules": [
            (
                "Context/source observations can support evidence.context_sources "
                "but cannot authorize filter, aggregate, derive, sort, limit, "
                "deduplicate, or reshape."
            ),
            (
                "A transformation needs an exact user quote detected as an "
                "operation, or an observed knowledge quote/KnowledgeFact.fact_id."
            ),
            (
                "When source grain or scope is ambiguous, record the conflict in "
                "evidence or unresolved items; do not invent a transformation."
            ),
            (
                "Knowledge markdown/PDF field definitions are semantic targets "
                "or source-name hints only. They are not observed physical fields "
                "unless a structured source lists the field or a narrative "
                "extraction result reports it."
            ),
        ],
        "knowledge_available": discovery.knowledge_available,
    }


def _decision_evidence_boundary_text(
    state: Mapping[str, Any],
    discovery: _DiscoveryState,
) -> str:
    if not (
        state.get("original_request")
        or state.get("question_structure")
        or _state_observed_sources(state)
    ):
        return ""
    payload = _decision_evidence_boundary_payload(state, discovery)
    return (
        "## Decision Evidence Boundary\n"
        "<evidence_boundary>\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n"
        "</evidence_boundary>\n"
        "Run the next decision under this authorization/evidence boundary. "
        "Do not infer or execute transformations outside it."
    )


def _inject_decision_evidence_boundary(
    request: ModelRequest[None],
    discovery: _DiscoveryState,
) -> ModelRequest[None]:
    boundary_text = _decision_evidence_boundary_text(request.state, discovery)
    if not boundary_text:
        return request
    current_system = request.system_message
    current_text = _message_text(current_system) if current_system is not None else ""
    messages = list(request.messages)
    messages_have_boundary = any(
        isinstance(message, SystemMessage)
        and "</evidence_boundary>" in _message_text(message)
        for message in messages
    )
    if not messages_have_boundary:
        messages = [SystemMessage(content=boundary_text), *messages]
    if "</evidence_boundary>" in current_text:
        return request.override(messages=messages)
    return request.override(
        messages=messages,
        system_message=SystemMessage(
            content=f"{current_text.rstrip()}\n\n{boundary_text}".strip()
        )
    )


def _user_quote_authorizes_operation(
    *,
    operation: str,
    quote: str,
    arguments: Mapping[str, Any],
    state: Mapping[str, Any],
) -> bool:
    operation_name = operation.casefold()
    if not state.get("question_structure_enforced"):
        original_request = str(state.get("original_request") or "")
        return bool(quote and quote in original_request) or operation_name == "join"
    requirement_types = _requirement_types_for_quote(
        quote=quote,
        arguments=arguments,
        state=state,
    )
    if requirement_types & _OPERATION_REQUIREMENT_TYPES.get(operation_name, frozenset()):
        return True
    return operation_name == "join"


def _operation_item_is_supported(
    item: Mapping[str, Any],
    *,
    arguments: Mapping[str, Any],
    state: Mapping[str, Any],
) -> bool:
    if not state.get("question_structure_enforced"):
        return True
    operation = str(item.get("operation") or "").strip()
    if not operation:
        return True
    authorization = item.get("authorization")
    if not isinstance(authorization, Mapping):
        return True
    if str(authorization.get("source") or "") != "user":
        return True
    quote = str(authorization.get("quote") or "").strip()
    return _user_quote_authorizes_operation(
        operation=operation,
        quote=quote,
        arguments=arguments,
        state=state,
    )


def _canonicalize_unsupported_operations(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    changed = False
    updated_arguments = dict(arguments)

    output_spec = updated_arguments.get("output_spec")
    if isinstance(output_spec, Mapping):
        transformations = output_spec.get("transformations") or []
        if isinstance(transformations, list):
            supported_transformations = [
                item
                for item in transformations
                if not isinstance(item, Mapping)
                or _operation_item_is_supported(
                    item,
                    arguments=updated_arguments,
                    state=request.state,
                )
            ]
            if supported_transformations != transformations:
                updated_output_spec = dict(output_spec)
                updated_output_spec["transformations"] = supported_transformations
                if not supported_transformations:
                    updated_output_spec["row_policy"] = "preserve"
                    updated_output_spec["ordering"] = "source"
                    updated_output_spec["sort_keys"] = []
                    updated_output_spec["null_policy"] = "preserve"
                    updated_output_spec["expected_row_count"] = None
                updated_arguments["output_spec"] = updated_output_spec
                changed = True

    execution_spec = updated_arguments.get("execution_spec")
    if isinstance(execution_spec, Mapping):
        operations = execution_spec.get("operations") or []
        if isinstance(operations, list):
            supported_operations = [
                item
                for item in operations
                if not isinstance(item, Mapping)
                or _operation_item_is_supported(
                    item,
                    arguments=updated_arguments,
                    state=request.state,
                )
            ]
            if supported_operations != operations:
                updated_execution_spec = dict(execution_spec)
                updated_execution_spec["operations"] = supported_operations
                updated_arguments["execution_spec"] = updated_execution_spec
                changed = True

    if not changed:
        return request
    return request.override(
        tool_call={
            **request.tool_call,
            "args": updated_arguments,
        }
    )


def _operation_description(operation: str, quote: str) -> str:
    return f"Apply {operation} authorized by user quote {quote!r}."


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def _limit_count_from_question_structure(
    *,
    operation_item: Mapping[str, Any],
    state: Mapping[str, Any],
) -> int | None:
    authorization = operation_item.get("authorization")
    quote = (
        str(authorization.get("quote") or "").strip()
        if isinstance(authorization, Mapping)
        else ""
    )
    question_structure = state.get("question_structure")
    if not isinstance(question_structure, Mapping):
        return None

    for item in question_structure.get("intent_operators") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("operation") or "").strip() != "limit":
            continue
        item_quote = str(item.get("quote") or "").strip()
        if quote and item_quote and item_quote != quote:
            continue
        for key in ("limit", "count", "n", "value"):
            count = _positive_int(item.get(key))
            if count is not None:
                return count
        if str(item.get("operator_type") or "") == "selector":
            return 1

    conditions = question_structure.get("conditions")
    if isinstance(conditions, Mapping):
        for item in conditions.get("limits") or []:
            if not isinstance(item, Mapping):
                continue
            item_quote = str(item.get("quote") or "").strip()
            if quote and item_quote and item_quote != quote:
                continue
            for key in ("limit", "count", "n", "value"):
                count = _positive_int(item.get(key))
                if count is not None:
                    return count
    return None


def _expected_row_count_from_limit_operation(
    operation_item: Mapping[str, Any],
    state: Mapping[str, Any],
) -> int | None:
    operation = str(operation_item.get("operation") or "").casefold()
    if operation != "limit":
        return None
    for key in ("limit", "count", "n"):
        count = _positive_int(operation_item.get(key))
        if count is not None:
            return count
    return _limit_count_from_question_structure(
        operation_item=operation_item,
        state=state,
    )


def _authorization_quote_for_operation(
    *,
    operation: str,
    arguments: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str | None:
    original_request = str(state.get("original_request") or "")
    question_structure = state.get("question_structure")
    if state.get("question_structure_enforced"):
        operations_by_quote = _question_structure_authorized_operations_by_quote(
            question_structure,
            original_request,
        )
        authorized_quotes = sorted(
            (
                quote
                for quote, operations in operations_by_quote.items()
                if operation in operations
            ),
            key=lambda item: (-len(item), item),
        )
        return authorized_quotes[0] if authorized_quotes else None

    intent = arguments.get("intent")
    if isinstance(intent, Mapping):
        for item in intent.get("requirements") or []:
            if not isinstance(item, Mapping):
                continue
            quote = str(item.get("quote") or "").strip()
            if not quote:
                continue
            requirement_types = _requirement_types_for_quote(
                quote=quote,
                arguments=arguments,
                state=state,
            )
            operation_types = _OPERATION_REQUIREMENT_TYPES.get(
                operation,
                frozenset(),
            )
            if requirement_types & operation_types:
                return quote

    if not isinstance(question_structure, Mapping):
        return None

    for item in question_structure.get("intent_operators") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("operation") or "").strip() != operation:
            continue
        quote = str(item.get("quote") or "").strip()
        if quote and quote in original_request:
            return quote

    condition_operations = {
        "filter": ("filters", "time_ranges"),
        "aggregate": ("calculations",),
        "derive": ("calculations",),
        "sort": ("orderings",),
        "limit": ("limits",),
        "deduplicate": (),
        "reshape": (),
    }
    conditions = question_structure.get("conditions")
    if not isinstance(conditions, Mapping):
        return None
    for container_name in condition_operations.get(operation, ()):
        for item in conditions.get(container_name) or []:
            if not isinstance(item, Mapping):
                continue
            quote = str(item.get("quote") or "").strip()
            if quote and quote in original_request:
                return quote
    return None


def _authorized_operations_from_contract(
    arguments: Mapping[str, Any],
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for operation in sorted(_TRANSFORM_OPERATIONS):
        quote = _authorization_quote_for_operation(
            operation=operation,
            arguments=arguments,
            state=state,
        )
        if not quote or operation in seen:
            continue
        operation_payload = {
            "operation": operation,
            "description": _operation_description(operation, quote),
            "authorization": {"source": "user", "quote": quote},
        }
        if operation == "limit":
            count = _limit_count_from_question_structure(
                operation_item=operation_payload,
                state=state,
            )
            if count is not None:
                operation_payload["limit"] = count
        operations.append(
            operation_payload
        )
        seen.add(operation)
    return operations


def _canonicalize_authorized_plan_operations(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return request

    updated_arguments = dict(arguments)
    updated_output_spec = dict(output_spec)
    execution_spec = updated_arguments.get("execution_spec")
    updated_execution_spec = (
        dict(execution_spec) if isinstance(execution_spec, Mapping) else {}
    )
    updated_execution_spec.setdefault("sources", [])
    updated_execution_spec.setdefault("supporting_fields", [])
    updated_execution_spec.setdefault("operations", [])

    transformations = [
        item
        for item in updated_output_spec.get("transformations") or []
        if isinstance(item, Mapping)
    ]
    execution_operations = [
        item
        for item in updated_execution_spec.get("operations") or []
        if isinstance(item, Mapping)
    ]
    transformation_operations = {
        str(item.get("operation") or "").casefold()
        for item in transformations
        if str(item.get("operation") or "").strip()
    }
    mirrored_transformations = [
        dict(item)
        for item in execution_operations
        if str(item.get("operation") or "").casefold() in _TRANSFORM_OPERATIONS
        and str(item.get("operation") or "").casefold() not in transformation_operations
    ]
    changed = False
    if mirrored_transformations:
        transformations = [*transformations, *mirrored_transformations]
        updated_output_spec["transformations"] = transformations
        changed = True

    declared = {
        str(item.get("operation") or "").casefold()
        for item in [*transformations, *execution_operations]
        if str(item.get("operation") or "").strip()
    }

    additions = [
        item
        for item in _authorized_operations_from_contract(arguments, request.state)
        if item["operation"] not in declared
    ]
    if additions:
        transformations = [*transformations, *additions]
        execution_operations = [*execution_operations, *additions]
        updated_output_spec["transformations"] = transformations
        updated_execution_spec["operations"] = execution_operations
        changed = True

    limit_counts = [
        count
        for count in (
            _expected_row_count_from_limit_operation(item, request.state)
            for item in [*transformations, *execution_operations]
            if isinstance(item, Mapping)
        )
        if count is not None
    ]
    if limit_counts:
        expected_row_count = min(limit_counts)
        if updated_output_spec.get("expected_row_count") != expected_row_count:
            updated_output_spec["expected_row_count"] = expected_row_count
            changed = True

    if transformations:
        if updated_output_spec.get("row_policy") != "transform":
            updated_output_spec["row_policy"] = "transform"
            changed = True
        if not str(updated_output_spec.get("ordering") or "").strip():
            updated_output_spec["ordering"] = (
                "specified"
                if any(
                    str(item.get("operation") or "") == "sort"
                    for item in transformations
                )
                else "source"
            )
            changed = True
        if not str(updated_output_spec.get("null_policy") or "").strip():
            updated_output_spec["null_policy"] = "preserve"
            changed = True
    else:
        preserve_defaults = {
            "row_policy": "preserve",
            "ordering": "source",
            "sort_keys": [],
            "null_policy": "preserve",
        }
        for key, value in preserve_defaults.items():
            if updated_output_spec.get(key) != value:
                updated_output_spec[key] = value
                changed = True

    if not changed:
        return request
    updated_arguments["output_spec"] = updated_output_spec
    updated_arguments["execution_spec"] = updated_execution_spec
    return request.override(
        tool_call={
            **request.tool_call,
            "args": updated_arguments,
        }
    )


def _canonicalize_distribution_output_columns(
    request: ToolCallRequest,
) -> ToolCallRequest:
    if not _question_requests_distribution(request.state):
        return request
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return request
    output_columns = output_spec.get("columns")
    if not isinstance(output_columns, list) or not output_columns:
        return request

    execution_spec = arguments.get("execution_spec")
    declared_operations = {
        str(item.get("operation") or "").strip().casefold()
        for item in output_spec.get("transformations") or []
        if isinstance(item, Mapping)
    }
    if isinstance(execution_spec, Mapping):
        declared_operations.update(
            str(item.get("operation") or "").strip().casefold()
            for item in execution_spec.get("operations") or []
            if isinstance(item, Mapping)
        )
    if "aggregate" not in declared_operations:
        return request

    updated_columns = list(output_columns)
    count_expression = _knowledge_count_expression(arguments)
    normalized_columns: list[Any] = []
    normalized_changed = False
    for column in updated_columns:
        if not isinstance(column, Mapping):
            normalized_columns.append(column)
            continue
        if not _column_is_count_measure(column):
            updated_column = dict(column)
            if str(updated_column.get("role") or "").casefold() != "entity_key":
                updated_column["role"] = "entity_key"
                normalized_changed = True
            normalized_columns.append(updated_column)
            continue
        column_name = str(column.get("name") or "").strip() or "count"
        normalized_column_name = _normalized_field_alias(column_name)
        source_fields = [
            str(field or "").strip()
            for field in column.get("source_fields") or []
            if str(field or "").strip()
        ]
        if (
            count_expression
            and normalized_column_name
            in {"count", "frequency", "freq", "recordcount", "rowcount"}
        ):
            updated_column = dict(column)
            updated_column["name"] = count_expression
            updated_column["source_fields"] = [count_expression]
            normalized_columns.append(updated_column)
            normalized_changed = True
            continue
        if not any(
            _normalized_field_alias(field)
            in {"count", "frequency", "freq", "recordcount", "rowcount"}
            for field in source_fields
        ):
            updated_column = dict(column)
            updated_column["source_fields"] = [column_name]
            normalized_columns.append(updated_column)
            normalized_changed = True
            continue
        normalized_columns.append(column)
    if normalized_changed:
        updated_columns = normalized_columns
    has_count = any(
        isinstance(column, Mapping) and _column_is_count_measure(column)
        for column in updated_columns
    )
    has_group_dimension = any(
        isinstance(column, Mapping) and not _column_is_count_measure(column)
        for column in updated_columns
    )
    if not has_group_dimension and isinstance(execution_spec, Mapping):
        for field in execution_spec.get("supporting_fields") or []:
            if not isinstance(field, Mapping):
                continue
            purpose = str(field.get("purpose") or "").casefold()
            if purpose not in {"group", "grouping", "selector"}:
                continue
            source_fields = [
                str(item).strip()
                for item in field.get("source_fields") or []
                if str(item).strip()
            ]
            name = str(field.get("name") or (source_fields[0] if source_fields else "")).strip()
            if not name:
                continue
            updated_columns.insert(
                0,
                {
                    "name": name,
                    "source_fields": source_fields or [name],
                    "role": "entity_key",
                },
            )
            has_group_dimension = True
            break
    if not has_group_dimension:
        return request
    if not has_count:
        updated_columns.append(
            {"name": "Count", "source_fields": ["Count"], "role": "measure"}
        )
    if updated_columns == output_columns:
        return request

    updated_output_spec = dict(output_spec)
    updated_output_spec["columns"] = updated_columns
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "output_spec": updated_output_spec,
            },
        }
    )


def _source_paths_from_plan_arguments(arguments: Mapping[str, Any]) -> set[str]:
    paths: set[str] = set()
    execution_spec = arguments.get("execution_spec")
    paths.update(_source_paths_from_execution_spec(execution_spec))
    evidence = arguments.get("evidence")
    if isinstance(evidence, Mapping):
        paths.update(
            str(source.get("path") or "").replace("\\", "/")
            for source in evidence.get("context_sources") or []
            if isinstance(source, Mapping)
            and str(source.get("path") or "").strip()
        )
    return {path for path in paths if path and not path.lower().endswith("/knowledge.md")}


def _observed_row_count_for_plan_sources(
    *,
    arguments: Mapping[str, Any],
    state: Mapping[str, Any],
) -> int | None:
    source_paths = _source_paths_from_plan_arguments(arguments)
    if not source_paths:
        return None
    row_counts = {
        **_observed_source_row_counts(state.get("messages", [])),
        **_observed_source_row_counts_from_state(state),
    }
    matched_counts = {
        row_counts[path]
        for path in source_paths
        if path in row_counts
    }
    if len(matched_counts) != 1:
        return None
    return matched_counts.pop()


def _output_columns_are_direct_source_projection(output_spec: Mapping[str, Any]) -> bool:
    columns = output_spec.get("columns")
    if not isinstance(columns, list) or not columns:
        return False
    return all(
        isinstance(column, Mapping)
        and any(str(field or "").strip() for field in column.get("source_fields") or [])
        for column in columns
    )


def _canonicalize_direct_source_projection_plan(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return request
    transformations = [
        item
        for item in output_spec.get("transformations") or []
        if isinstance(item, Mapping)
    ]
    execution_spec = arguments.get("execution_spec")
    execution_operations = (
        [
            item
            for item in execution_spec.get("operations") or []
            if isinstance(item, Mapping)
        ]
        if isinstance(execution_spec, Mapping)
        else []
    )
    if not transformations and not execution_operations:
        return request
    if output_spec.get("sort_keys"):
        return request
    ordering = str(output_spec.get("ordering") or "").strip().casefold()
    if ordering not in {"", "source", "unspecified"}:
        return request
    null_policy = str(output_spec.get("null_policy") or "").strip().casefold()
    if null_policy not in {"", "preserve"}:
        return request
    if not _output_columns_are_direct_source_projection(output_spec):
        return request
    expected_row_count = output_spec.get("expected_row_count")
    observed_row_count = _observed_row_count_for_plan_sources(
        arguments=arguments,
        state=request.state,
    )
    if (
        not isinstance(expected_row_count, int)
        or observed_row_count is None
        or expected_row_count != observed_row_count
    ):
        return request

    updated_output_spec = dict(output_spec)
    updated_output_spec["transformations"] = []
    updated_output_spec["row_policy"] = "preserve"
    updated_output_spec["ordering"] = "source"
    updated_output_spec["sort_keys"] = []
    updated_output_spec["null_policy"] = "preserve"
    updated_execution_spec = (
        dict(execution_spec) if isinstance(execution_spec, Mapping) else {}
    )
    updated_execution_spec["operations"] = []
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "output_spec": updated_output_spec,
                "execution_spec": updated_execution_spec,
            },
        }
    )


def _canonicalize_duplicate_output_columns(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return request
    columns = output_spec.get("columns")
    if not isinstance(columns, list) or len(columns) <= 1:
        return request
    seen: set[tuple[str, ...]] = set()
    deduplicated: list[Any] = []
    changed = False
    for column in columns:
        if not isinstance(column, Mapping):
            deduplicated.append(column)
            continue
        source_fields = tuple(
            _normalized_field_alias(field)
            for field in column.get("source_fields") or []
            if _normalized_field_alias(field)
        )
        key = source_fields or (_normalized_field_alias(column.get("name")),)
        if key and key in seen:
            changed = True
            continue
        if key:
            seen.add(key)
        deduplicated.append(column)
    if not changed:
        return request
    updated_output_spec = dict(output_spec)
    updated_output_spec["columns"] = deduplicated
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "output_spec": updated_output_spec,
            },
        }
    )


def _requested_measure_output_count(state: Mapping[str, Any]) -> int:
    question_structure = state.get("question_structure")
    if not isinstance(question_structure, Mapping):
        return 0
    output = question_structure.get("output")
    requested_columns = set()
    if isinstance(output, Mapping):
        requested_columns = {
            _normalized_quote_text(str(item or ""))
            for item in output.get("requested_columns") or []
            if str(item or "").strip()
        }
    if not requested_columns:
        return 0
    count = 0
    for target in question_structure.get("targets") or []:
        if not isinstance(target, Mapping):
            continue
        target_type = str(target.get("target_type") or "").strip().casefold()
        if target_type not in {"measure", "metric"}:
            continue
        target_texts = {
            _normalized_quote_text(str(target.get(key) or ""))
            for key in ("quote", "name", "description")
            if str(target.get(key) or "").strip()
        }
        if target_texts & requested_columns:
            count += 1
    return count


def _question_requests_record_set(state: Mapping[str, Any]) -> bool:
    question_structure = state.get("question_structure")
    if not isinstance(question_structure, Mapping):
        return False
    return any(
        isinstance(target, Mapping)
        and str(target.get("target_type") or "").strip().casefold() == "record_set"
        for target in question_structure.get("targets") or []
    )


def _question_requests_distribution(state: Mapping[str, Any]) -> bool:
    question_structure = state.get("question_structure")
    if not isinstance(question_structure, Mapping):
        return False
    return any(
        isinstance(item, Mapping)
        and str(item.get("operation") or "").strip().casefold() == "aggregate"
        and str(item.get("operator_type") or "").strip().casefold()
        == "distribution"
        for item in question_structure.get("intent_operators") or []
    )


def _column_is_count_measure(column: Mapping[str, Any]) -> bool:
    aliases = _column_field_aliases(column)
    return bool(
        aliases
        & {
            "count",
            "frequency",
            "freq",
            "recordcount",
            "rowcount",
            "数量",
            "频数",
            "频率",
        }
    )


def _knowledge_count_expression(arguments: Mapping[str, Any]) -> str | None:
    evidence = arguments.get("evidence")
    if not isinstance(evidence, Mapping):
        return None
    for rule in evidence.get("knowledge_rules") or []:
        if not isinstance(rule, Mapping):
            continue
        quote = str(rule.get("quote") or "")
        match = re.search(r"\bCOUNT\s*\(\s*\*\s*\)", quote, flags=re.IGNORECASE)
        if match is not None:
            return "count(*)"
    return None


def _source_binding_field_aliases(arguments: Mapping[str, Any]) -> set[str]:
    execution_spec = arguments.get("execution_spec")
    if isinstance(execution_spec, str):
        try:
            execution_spec = json.loads(execution_spec)
        except json.JSONDecodeError:
            execution_spec = {}
    if not isinstance(execution_spec, Mapping):
        return set()
    return {
        _normalized_field_alias(binding.get("source_field"))
        for binding in execution_spec.get("source_bindings") or []
        if isinstance(binding, Mapping)
        and str(binding.get("source_field") or "").strip()
    }


def _canonicalize_unrequested_measure_output_columns(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    if not isinstance(output_spec, Mapping):
        return request
    output_columns = output_spec.get("columns")
    if not isinstance(output_columns, list) or len(output_columns) <= 1:
        return request
    requested_texts = _question_requested_output_texts(request.state)
    requested_measure_count = _requested_measure_output_count(request.state)
    requests_record_set = _question_requests_record_set(request.state)
    requests_distribution = _question_requests_distribution(request.state)
    source_binding_aliases = _source_binding_field_aliases(arguments)
    measure_columns = [
        column
        for column in output_columns
        if isinstance(column, Mapping)
        and str(column.get("role") or "").casefold() in {"measure", "metric"}
    ]
    kept_columns: list[Any] = []
    demoted_columns: list[Mapping[str, Any]] = []
    for column in output_columns:
        if not isinstance(column, Mapping):
            kept_columns.append(column)
            continue
        role = str(column.get("role") or "").casefold()
        if role in {"measure", "metric"}:
            if requests_distribution and _column_is_count_measure(column):
                kept_columns.append(column)
                continue
            if requests_record_set and (
                not source_binding_aliases
                or (_column_field_aliases(column) & source_binding_aliases)
            ):
                kept_columns.append(column)
                continue
            if _column_matches_requested_output(column, requested_texts):
                kept_columns.append(column)
                continue
            if _column_matches_requested_measure(column, request.state):
                kept_columns.append(column)
                continue
            if (
                requested_measure_count
                and len(measure_columns) <= requested_measure_count
            ):
                kept_columns.append(column)
                continue
            demoted_columns.append(column)
            continue
        kept_columns.append(column)
    if not demoted_columns or not kept_columns:
        return request
    updated_arguments = _demote_output_columns(
        arguments=arguments,
        output_spec=output_spec,
        demoted_columns=demoted_columns,
        kept_columns=kept_columns,
        purpose="selector",
    )
    if updated_arguments is None:
        return request
    return request.override(
        tool_call={
            **request.tool_call,
            "args": updated_arguments,
        }
    )


def _canonicalize_missing_intent_requirements(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    original_request = str(request.state.get("original_request") or "").strip()
    if not original_request:
        return request
    intent = arguments.get("intent")
    if isinstance(intent, Mapping) and intent.get("requirements"):
        return request
    updated_intent = dict(intent) if isinstance(intent, Mapping) else {}
    updated_intent["requirements"] = [
        {
            "statement": "Answer the original user request.",
            "requirement_type": "output",
            "quote": original_request,
        }
    ]
    updated_intent.setdefault("unresolved", [])
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "intent": updated_intent,
            },
        }
    )


def _canonicalize_non_authoritative_knowledge(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    evidence = arguments.get("evidence")
    if not isinstance(evidence, Mapping):
        return request
    knowledge_status = str(evidence.get("knowledge_status") or "")
    if knowledge_status == "authoritative":
        return request

    updated_evidence = dict(evidence)
    changed = False
    output_spec = arguments.get("output_spec")
    output_columns = (
        output_spec.get("columns")
        if isinstance(output_spec, Mapping)
        else []
    )
    has_unbound_output = any(
        isinstance(column, Mapping)
        and not [
            field
            for field in column.get("source_fields") or []
            if str(field or "").strip()
        ]
        for column in (output_columns if isinstance(output_columns, list) else [])
    )
    narrative_paths = sorted(
        {
            str(source.get("path") or "").replace("\\", "/")
            for source in _state_observed_sources(request.state)
            if str(source.get("source_type") or "") == "doc"
            and str(source.get("path") or "").strip()
        }
    )
    if (
        knowledge_status in {"unavailable", "invalid"}
        and narrative_paths
        and not has_unbound_output
    ):
        updated_evidence["knowledge_status"] = "insufficient"
        changed = True
    if not str(updated_evidence.get("knowledge_issue") or "").strip():
        updated_evidence["knowledge_issue"] = (
            "Knowledge is not fully authoritative for this plan; rely on observed "
            "context sources and explicitly cited usable facts only."
        )
        changed = True
    if not str(updated_evidence.get("cross_validated_inference") or "").strip():
        source_summary = (
            f"Observed narrative sources: {narrative_paths}."
            if narrative_paths
            else "Observed context sources are used as evidence only."
        )
        updated_evidence["cross_validated_inference"] = source_summary
        changed = True
    if changed:
        return request.override(
            tool_call={
                **request.tool_call,
                "args": {
                    **dict(arguments),
                    "evidence": updated_evidence,
                },
            }
        )
    return request


def _canonicalize_empty_authoritative_knowledge(
    request: ToolCallRequest,
    discovery: _DiscoveryState,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    evidence = arguments.get("evidence")
    if not isinstance(evidence, Mapping):
        return request
    if str(evidence.get("knowledge_status") or "") != "authoritative":
        return request
    if evidence.get("knowledge_rules"):
        return request
    if parse_knowledge_content(discovery.knowledge_content):
        return request

    updated_evidence = dict(evidence)
    updated_evidence["knowledge_status"] = "unavailable"
    updated_evidence["knowledge_issue"] = (
        "knowledge.md contains no parseable knowledge facts or quoted rules; "
        "use observed context evidence and user-authorized operations only."
    )
    updated_evidence.setdefault(
        "cross_validated_inference",
        "Observed context sources are used as evidence only.",
    )
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "evidence": updated_evidence,
            },
        }
    )


def _canonicalize_unsatisfied_authoritative_source_hints(
    request: ToolCallRequest,
    discovery: _DiscoveryState,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    evidence = arguments.get("evidence")
    if not isinstance(evidence, Mapping):
        return request
    if str(evidence.get("knowledge_status") or "") != "authoritative":
        return request

    requested_sources = {
        str(item.get("path") or "").replace("\\", "/")
        for item in evidence.get("context_sources") or []
        if isinstance(item, Mapping) and str(item.get("path") or "").strip()
    }
    execution_sources = _source_paths_from_execution_spec(
        arguments.get("execution_spec") or {},
    )
    missing_by_group = _missing_relevant_source_hint_groups(
        knowledge_facts=parse_knowledge_content(discovery.knowledge_content),
        state=request.state,
        plan_sources=requested_sources | execution_sources,
    )
    if not missing_by_group:
        return request

    observed_missing_hints = sorted(
        {
            str(hint)
            for group in missing_by_group
            for hint in group.get("missing") or []
            if _source_hint_has_observed_source(
                state=request.state,
                source_hint=str(hint),
            )
        }
    )
    if not observed_missing_hints:
        return request

    issue_parts = [
        str(evidence.get("knowledge_issue") or ""),
        str(evidence.get("cross_validated_inference") or ""),
    ]
    intent = arguments.get("intent")
    if isinstance(intent, Mapping):
        issue_parts.extend(str(item or "") for item in intent.get("unresolved") or [])
    issue_text = _normalized_quote_text(" ".join(issue_parts))
    if not issue_text:
        return request

    updated_evidence = dict(evidence)
    updated_evidence["knowledge_status"] = "insufficient"
    updated_evidence["knowledge_rules"] = []
    if not str(updated_evidence.get("knowledge_issue") or "").strip():
        updated_evidence["knowledge_issue"] = (
            "Request-relevant knowledge source hints were observed but cannot be "
            "satisfied by the execution sources; use observed context evidence "
            "for the remaining bindings."
        )
    updated_evidence["source_hint_issue"] = {
        "status": "observed_hint_not_satisfied_by_plan_sources",
        "observed_missing_hints": observed_missing_hints,
        "missing_by_group": missing_by_group[:3],
    }
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "evidence": updated_evidence,
            },
        }
    )


def _time_overlap_terms(value: str) -> set[str]:
    normalized = _normalized_quote_text(value)
    terms = {
        term.lstrip("近第")
        for term in re.findall(
            r"[近第]?(?:\d+|一|二|三|四|五|六|七|八|九|十|两)+[年月日季周天]",
            normalized,
        )
    }
    terms.update(
        term.replace("-", "").replace("_", "").replace(" ", "")
        for term in re.findall(
            (
                r"(?:one|two|three|four|five|six|seven|eight|nine|ten|"
                r"\d+)[-_\s]?(?:year|month|week|day|quarter)s?"
            ),
            normalized,
        )
    )
    return {term for term in terms if term}


def _fact_has_time_terms(fact: Any) -> bool:
    fact_blob = " ".join(
        str(item or "")
        for item in (fact.field_key, fact.section_key, fact.quote)
    )
    return bool(_time_overlap_terms(fact_blob))


def _knowledge_fact_defines_field(fact: Any) -> bool:
    return bool(
        str(getattr(fact, "section_key", "") or "").strip()
        and str(getattr(fact, "field_key", "") or "").strip()
        and not str(getattr(fact, "operation", "") or "").strip()
    )


def _knowledge_fact_field_aliases(fact: Any) -> set[str]:
    aliases = {
        _normalized_field_alias(getattr(fact, "field_key", ""))
    }
    aliases.update(
        _normalized_field_alias(item)
        for item in re.findall(r"`([^`]+)`", str(getattr(fact, "quote", "") or ""))
    )
    return {alias for alias in aliases if alias}


def _knowledge_fact_operation_names(fact: Any) -> set[str]:
    operation_text = str(getattr(fact, "operation", "") or "")
    operations = {
        item.strip().casefold()
        for item in re.split(r"[,;/\s]+", operation_text)
        if item.strip()
    }
    quote = str(getattr(fact, "quote", "") or "")
    if re.search(r"\bJOIN\b", quote, re.IGNORECASE):
        operations.add("join")
    if re.search(
        r"\b[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*\s*=\s*"
        r"[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*\b",
        quote,
    ):
        operations.add("join")
    if re.search(r"\b(?:join|link|relat|connect|map)\b", quote, re.IGNORECASE):
        operations.add("join")
    return operations


def _field_facts_for_knowledge_quote(
    *,
    quote: str,
    field_facts: list[Any],
) -> list[Any]:
    normalized_quote = _normalized_quote_text(quote)
    quote_tokens = _normalized_quote_tokens(quote)
    matches: list[Any] = []
    for fact in field_facts:
        normalized_fact = _normalized_quote_text(fact.quote)
        fact_tokens = _normalized_quote_tokens(fact.quote)
        if (
            quote == fact.quote
            or (quote and quote in fact.quote)
            or (normalized_quote and normalized_quote in normalized_fact)
            or (quote_tokens and quote_tokens.issubset(fact_tokens))
        ):
            matches.append(fact)
    return matches


def _canonical_knowledge_quote(
    quote: str,
    knowledge_content: str | None,
) -> str | None:
    """Resolve a simplified quote only when it maps to one knowledge line."""

    if not knowledge_content:
        return None
    content_lines = [line.strip() for line in knowledge_content.splitlines()]
    if quote and quote in content_lines:
        return quote
    normalized_quote = _normalized_quote_text(quote)
    if not normalized_quote:
        return None
    matches = [
        line
        for line in content_lines
        if normalized_quote in _normalized_quote_text(line)
    ]
    if len(matches) == 1:
        return matches[0]
    quote_tokens = _normalized_quote_tokens(quote)
    if quote_tokens:
        token_matches = [
            line.strip()
            for line in knowledge_content.splitlines()
            if quote_tokens.issubset(_normalized_quote_tokens(line))
        ]
        if len(token_matches) == 1:
            return token_matches[0]
    return None


def _collect_exact_question_quotes(value: Any, original_request: str) -> set[str]:
    quotes: set[str] = set()
    if isinstance(value, Mapping):
        quote = value.get("quote")
        if isinstance(quote, str):
            normalized = quote.strip()
            if normalized and normalized in original_request:
                quotes.add(normalized)
        for item in value.values():
            quotes.update(_collect_exact_question_quotes(item, original_request))
    elif isinstance(value, list):
        for item in value:
            quotes.update(_collect_exact_question_quotes(item, original_request))
    return quotes


def _state_question_quotes(state: Mapping[str, Any]) -> list[str]:
    original_request = str(state.get("original_request") or "").strip()
    if not original_request:
        return []
    quotes = {original_request}
    question_structure = state.get("question_structure")
    if isinstance(question_structure, Mapping):
        quotes.update(_collect_exact_question_quotes(question_structure, original_request))
    return sorted(quotes, key=lambda item: (-len(item), item))


def _canonical_user_quote(quote: str, state: Mapping[str, Any]) -> str | None:
    stripped = quote.strip()
    original_request = str(state.get("original_request") or "")
    if stripped and stripped in original_request:
        return stripped
    for candidate in _state_question_quotes(state):
        if len(candidate) < 2:
            continue
        if candidate in stripped:
            return candidate
    return None


def _canonicalize_user_authorization_quotes(
    arguments: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    updated_arguments = dict(arguments)
    changed = False

    def normalize_authorization(authorization: Any) -> tuple[Any, bool]:
        if not isinstance(authorization, Mapping):
            return authorization, False
        if str(authorization.get("source") or "") != "user":
            return authorization, False
        quote = str(authorization.get("quote") or "").strip()
        canonical_quote = _canonical_user_quote(quote, state)
        if canonical_quote is None or canonical_quote == quote:
            return authorization, False
        return {**dict(authorization), "quote": canonical_quote}, True

    output_spec = updated_arguments.get("output_spec")
    if isinstance(output_spec, Mapping):
        updated_output_spec = dict(output_spec)
        normalized_transformations: list[Any] = []
        for transformation in updated_output_spec.get("transformations") or []:
            if not isinstance(transformation, Mapping):
                normalized_transformations.append(transformation)
                continue
            normalized_transformation = dict(transformation)
            authorization, authorization_changed = normalize_authorization(
                transformation.get("authorization")
            )
            if authorization_changed:
                normalized_transformation["authorization"] = authorization
                changed = True
            normalized_transformations.append(normalized_transformation)
        if changed:
            updated_output_spec["transformations"] = normalized_transformations
            updated_arguments["output_spec"] = updated_output_spec

    execution_spec = updated_arguments.get("execution_spec")
    if isinstance(execution_spec, Mapping):
        updated_execution_spec = dict(execution_spec)
        normalized_operations: list[Any] = []
        operations_changed = False
        for operation in updated_execution_spec.get("operations") or []:
            if not isinstance(operation, Mapping):
                normalized_operations.append(operation)
                continue
            normalized_operation = dict(operation)
            authorization, authorization_changed = normalize_authorization(
                operation.get("authorization")
            )
            if authorization_changed:
                normalized_operation["authorization"] = authorization
                operations_changed = True
            normalized_operations.append(normalized_operation)
        if operations_changed:
            updated_execution_spec["operations"] = normalized_operations
            updated_arguments["execution_spec"] = updated_execution_spec
            changed = True

    return updated_arguments, changed



def _canonicalize_plan_quotes(
    request: ToolCallRequest,
    discovery: _DiscoveryState,
) -> ToolCallRequest:
    """Replace uniquely located knowledge quotes before validation."""

    raw_arguments = request.tool_call.get("args") or {}
    if not isinstance(raw_arguments, Mapping):
        return request
    arguments = dict(raw_arguments)
    arguments, changed = _canonicalize_user_authorization_quotes(
        arguments,
        request.state,
    )
    raw_evidence = arguments.get("evidence") or {}
    if not isinstance(raw_evidence, Mapping):
        return request
    evidence = dict(raw_evidence)
    if str(evidence.get("knowledge_status") or "") != "authoritative":
        context_sources = [
            source
            for source in evidence.get("context_sources") or []
            if not (
                isinstance(source, Mapping)
                and str(source.get("path") or "").replace("\\", "/").lower()
                == "/context/knowledge.md"
            )
        ]
        if context_sources != evidence.get("context_sources"):
            evidence["context_sources"] = context_sources
            changed = True
        if evidence.get("knowledge_rules"):
            evidence["knowledge_rules"] = []
            changed = True
        if changed:
            arguments["evidence"] = evidence
            return request.override(
                tool_call={
                    **request.tool_call,
                    "args": arguments,
                }
            )
        return request

    knowledge_facts_by_id = {
        fact.fact_id: fact for fact in parse_knowledge_content(discovery.knowledge_content)
    }

    def quote_for_fact_ids(value: Any) -> str | None:
        fact_id_values = value if isinstance(value, list) else []
        fact_quotes = [
            fact.quote
            for fact_id in (
                str(item).strip()
                for item in fact_id_values
                if str(item).strip()
            )
            if (fact := knowledge_facts_by_id.get(fact_id)) is not None
        ]
        unique_quotes = sorted(set(fact_quotes))
        if len(unique_quotes) == 1:
            return unique_quotes[0]
        return None

    def quote_for_authorization_fact_ids(
        operation_item: Mapping[str, Any],
        authorization: Mapping[str, Any],
    ) -> str | None:
        fact_ids: list[str] = []
        for key in ("fact_id", "knowledge_fact_id"):
            value = authorization.get(key)
            if str(value or "").strip():
                fact_ids.append(str(value).strip())
        for value in operation_item.get("authorization_fact_ids") or []:
            if str(value or "").strip():
                fact_ids.append(str(value).strip())
        return quote_for_fact_ids(fact_ids)

    normalized_rules: list[Any] = []
    for rule in evidence.get("knowledge_rules") or []:
        if not isinstance(rule, dict):
            normalized_rules.append(rule)
            continue
        normalized_rule = dict(rule)
        fact_id = str(rule.get("fact_id") or "").strip()
        fact = knowledge_facts_by_id.get(fact_id) if fact_id else None
        if fact is not None:
            if normalized_rule.get("quote") != fact.quote:
                normalized_rule["quote"] = fact.quote
                changed = True
            if normalized_rule.get("source_path") != fact.source_path:
                normalized_rule["source_path"] = fact.source_path
                changed = True
        quote = str(rule.get("quote") or "").strip()
        canonical_quote = _canonical_knowledge_quote(
            str(normalized_rule.get("quote") or quote),
            discovery.knowledge_content,
        )
        if canonical_quote is not None and canonical_quote != quote:
            normalized_rule["quote"] = canonical_quote
            changed = True
        if fact_id and fact_id not in knowledge_facts_by_id and canonical_quote is not None:
            normalized_rule.pop("fact_id", None)
            normalized_rule.setdefault("source_path", "/context/knowledge.md")
            changed = True
        normalized_rules.append(normalized_rule)

    operation_knowledge_authorizations: dict[str, dict[str, Any]] = {}
    for rule in normalized_rules:
        if not isinstance(rule, Mapping):
            continue
        fact = knowledge_facts_by_id.get(str(rule.get("fact_id") or "").strip())
        if fact is None:
            quote = str(rule.get("quote") or "")
            fact = next(
                (
                    item
                    for item in knowledge_facts_by_id.values()
                    if str(getattr(item, "quote", "") or "") == quote
                ),
                None,
            )
        if fact is None:
            continue
        for operation_name in _knowledge_fact_operation_names(fact):
            operation_knowledge_authorizations.setdefault(
                operation_name,
                {
                    "source": "knowledge",
                    "quote": fact.quote,
                    "fact_id": fact.fact_id,
                },
            )

    def normalize_operation_authorization(
        operation_item: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        normalized_item = dict(operation_item)
        operation_name = str(operation_item.get("operation") or "").strip().casefold()
        if not operation_name:
            return normalized_item, False
        authorization = operation_item.get("authorization")
        if not isinstance(authorization, Mapping):
            knowledge_authorization = operation_knowledge_authorizations.get(operation_name)
            if knowledge_authorization is None:
                return normalized_item, False
            normalized_item["authorization"] = knowledge_authorization
            return normalized_item, True
        source = str(authorization.get("source") or "").strip()
        if source == "knowledge":
            quote = str(authorization.get("quote") or "").strip()
            canonical_quote = _canonical_knowledge_quote(
                quote,
                discovery.knowledge_content,
            )
            if canonical_quote is None:
                canonical_quote = quote_for_authorization_fact_ids(
                    operation_item,
                    authorization,
                )
            if canonical_quote is None:
                knowledge_authorization = operation_knowledge_authorizations.get(
                    operation_name,
                )
                if knowledge_authorization is not None:
                    normalized_item["authorization"] = knowledge_authorization
                    return normalized_item, True
                return normalized_item, False
            if canonical_quote != quote:
                normalized_item["authorization"] = {
                    **dict(authorization),
                    "quote": canonical_quote,
                }
                return normalized_item, True
            return normalized_item, False
        if source == "user":
            quote = str(authorization.get("quote") or "").strip()
            if _user_quote_authorizes_operation(
                operation=operation_name,
                quote=quote,
                arguments=arguments,
                state=request.state,
            ):
                return normalized_item, False
            knowledge_authorization = operation_knowledge_authorizations.get(operation_name)
            if knowledge_authorization is None:
                return normalized_item, False
            normalized_item["authorization"] = knowledge_authorization
            return normalized_item, True
        return normalized_item, False

    normalized_output_spec = arguments.get("output_spec")
    if isinstance(normalized_output_spec, Mapping):
        updated_output_spec = dict(normalized_output_spec)
        normalized_transformations = []
        for transformation in updated_output_spec.get("transformations") or []:
            if not isinstance(transformation, Mapping):
                normalized_transformations.append(transformation)
                continue
            normalized_transformation, authorization_changed = (
                normalize_operation_authorization(transformation)
            )
            if authorization_changed:
                changed = True
            normalized_transformations.append(normalized_transformation)
        if normalized_transformations != updated_output_spec.get("transformations"):
            updated_output_spec["transformations"] = normalized_transformations
            arguments["output_spec"] = updated_output_spec
    normalized_execution_spec = arguments.get("execution_spec")
    if isinstance(normalized_execution_spec, Mapping):
        updated_execution_spec = dict(normalized_execution_spec)
        normalized_operations = []
        for operation in updated_execution_spec.get("operations") or []:
            if not isinstance(operation, Mapping):
                normalized_operations.append(operation)
                continue
            normalized_operation, authorization_changed = normalize_operation_authorization(
                operation,
            )
            if authorization_changed:
                changed = True
            normalized_operations.append(normalized_operation)
        if normalized_operations != updated_execution_spec.get("operations"):
            updated_execution_spec["operations"] = normalized_operations
            arguments["execution_spec"] = updated_execution_spec
    if not changed:
        return request

    evidence["knowledge_rules"] = normalized_rules
    arguments["evidence"] = evidence
    return request.override(
        tool_call={
            **request.tool_call,
            "args": arguments,
        }
    )


def _context_sources(tool: str, arguments: dict[str, Any]) -> set[str]:
    """Extract context sources inspected by successful discovery calls."""

    if tool in {
        "grep_file",
        "inspect_sqlite",
        "extract_narrative_records",
        "read_csv",
        "read_doc",
        "read_json",
        "execute_sql",
    }:
        path = str(
            arguments.get("file_path")
            or arguments.get("path")
            or ""
        ).replace("\\", "/")
        if path in {"", ".", "/context", "context"}:
            return set()
        if not path.startswith("/context/"):
            path = f"/context/{path.removeprefix('context/').lstrip('/')}"
        if not path.lower().endswith("/knowledge.md"):
            return {path}
        return set()
    if tool == "execute_python":
        code = str(arguments.get("code") or "")
        return {
            path.replace("\\", "/")
            for path in _CONTEXT_PATH_PATTERN.findall(code)
            if not path.lower().endswith("/knowledge.md")
        }
    return set()


def _discovery_state(
    messages: list[BaseMessage],
    state: Mapping[str, Any] | None = None,
) -> _DiscoveryState:
    """Summarize knowledge and context sources observed before planning."""

    tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}
    state_knowledge = ""
    if state is not None:
        state_knowledge = str(state.get("knowledge_content") or "").strip()
        if state_knowledge in {"", "<missing>", "<empty>"} or state_knowledge.startswith(
            "<unreadable:"
        ):
            state_knowledge = ""
    injected_knowledge = state_knowledge or (_injected_knowledge_content(messages) or "")
    knowledge_present = bool(injected_knowledge)
    knowledge_checked = knowledge_present
    knowledge_available = knowledge_present
    knowledge_contents: list[str] = [injected_knowledge] if injected_knowledge else []
    context_sources: set[str] = set()
    needs_cross_validation = False

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                tool_call_id = str(tool_call.get("id") or "")
                if tool_call_id:
                    raw_arguments = _decode_json_like_argument(tool_call.get("args") or {})
                    arguments = (
                        dict(raw_arguments)
                        if isinstance(raw_arguments, Mapping)
                        else {}
                    )
                    tool_calls[tool_call_id] = (
                        str(tool_call.get("name") or ""),
                        arguments,
                    )
            continue
        if not isinstance(message, ToolMessage):
            continue

        tool_call = tool_calls.get(str(message.tool_call_id or ""))
        if tool_call is None:
            continue
        current_tool_name, arguments = tool_call

        if current_tool_name == "read_doc":
            file_path = str(
                arguments.get("file_path")
                or arguments.get("path")
                or ""
            ).replace("\\", "/")
            if file_path.lower().endswith("/knowledge.md"):
                knowledge_checked = True
                knowledge_available = getattr(message, "status", "success") != "error"
                if knowledge_available and not injected_knowledge:
                    knowledge_contents.append(_message_text(message))
                continue

        if (
            current_tool_name in {"execute_python", "grep_file"}
            and getattr(message, "status", "success") != "error"
        ):
            inspected_text = str(
                arguments.get("code")
                or arguments.get("file_path")
                or arguments.get("path")
                or ""
            ).replace("\\", "/")
            if "/context/knowledge.md" in inspected_text:
                knowledge_checked = True
                knowledge_available = True
                observed_content = _message_text(message)
                if observed_content and not injected_knowledge:
                    knowledge_contents.append(observed_content)

        if (
            current_tool_name == "analyze_plan"
            and getattr(message, "status", "success") == "error"
        ):
            evidence = arguments.get("evidence") or {}
            needs_cross_validation = (
                not isinstance(evidence, Mapping)
                or
                str(evidence.get("knowledge_status") or "") != "authoritative"
            )
            continue

        if (
            current_tool_name in _SOURCE_DISCOVERY_TOOLS
            and getattr(message, "status", "success") != "error"
        ):
            context_sources.update(_context_sources(current_tool_name, arguments))

    if state is not None:
        context_sources.update(_observed_context_sources_from_state(state))

    knowledge_content = "\n".join(knowledge_contents)
    knowledge_present = knowledge_present or bool(knowledge_content.strip())

    return _DiscoveryState(
        knowledge_present=knowledge_present,
        knowledge_checked=knowledge_checked,
        knowledge_available=knowledge_available,
        knowledge_content=knowledge_content,
        context_sources=frozenset(context_sources),
        needs_cross_validation=needs_cross_validation,
    )


def _plan_error(
    request: ToolCallRequest,
    content: str,
) -> ToolMessage:
    hints: dict[str, Any] = {"message": content}
    if "row_policy" in content or "preserve plans" in content:
        hints["repair_hints"] = [
            "If output_spec.transformations is empty, use row_policy='preserve', ordering='source', sort_keys=[], null_policy='preserve'.",
            "If the user or knowledge authorizes a transformation, declare it in output_spec.transformations and execution_spec.operations.",
        ]
    elif "knowledge" in content:
        hints["repair_hints"] = [
            "Use knowledge_status_for_plan from the injected knowledge schema.",
            "If knowledge is not authoritative, include a concrete knowledge_issue and cross_validated_inference.",
        ]
    elif "unresolved source/field binding" in content:
        hints["repair_hints"] = [
            "Do not execute a plan that depends on substitute or unresolved source bindings.",
            "Use query_schema/read_json/read_csv/inspect_sqlite for direct structured field evidence, or grep_file/read_doc plus extract_narrative_records for narrative evidence.",
            "Only revise analyze_plan after the missing source field/value evidence is observed.",
        ]
    elif "operation" in content or "authorization" in content:
        hints["repair_hints"] = [
            "Use only exact user quotes or observed knowledge quotes/fact_ids as operation authorization.",
            "Context observations can provide evidence but cannot authorize transformations.",
        ]
    elif "explicit user scope constraints are unresolved" in content:
        hints["repair_hints"] = [
            "Do not retry analyze_plan with the same unresolved scope.",
            "Run mechanical discovery to bind the quoted scope to observed fields, values, lines, or extraction evidence.",
            "For narrative/PDF sources, use grep_file/read_doc windows and then extract_narrative_records with source_fields for the target fields.",
        ]
    else:
        hints["repair_hints"] = [
            "Revise only the invalid plan fields and keep the original task semantics unchanged."
        ]
    return _tool_error(
        request,
        (
            f"Invalid analysis plan: {content}\n"
            f"{json.dumps({'repair_hints': hints}, ensure_ascii=False)}"
        ),
    )


def _validate_plan_contract(
    request: ToolCallRequest,
    discovery: _DiscoveryState,
) -> ToolMessage | None:
    """Validate plans using explicit quotes and observed state only."""

    arguments = request.tool_call.get("args") or {}
    if not isinstance(arguments, Mapping):
        return _plan_error(request, "tool arguments must be a JSON object.")
    original_request = str(request.state.get("original_request") or "")
    if not original_request:
        return _plan_error(request, "original_request is missing from agent state.")

    intent = arguments.get("intent") or {}
    if not isinstance(intent, Mapping):
        return _plan_error(request, "intent must be a JSON object.")
    requirements = intent.get("requirements") or []
    requirement_types_by_quote: dict[str, set[str]] = {}
    for item in requirements:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote") or "").strip()
        requirement_type = str(item.get("requirement_type") or "").strip()
        if quote and requirement_type:
            requirement_types_by_quote.setdefault(quote, set()).add(requirement_type)
    if not requirement_types_by_quote or any(
        not quote or quote not in original_request
        for quote in requirement_types_by_quote
    ):
        return _plan_error(
            request,
            "every intent requirement quote must occur verbatim in original_request.",
        )
    effective_requirement_types_by_quote = _effective_requirement_types_by_quote(
        requirement_types_by_quote,
        request.state,
    )
    evidence = arguments.get("evidence") or {}
    if not isinstance(evidence, Mapping):
        return _plan_error(request, "evidence must be a JSON object.")
    output_spec = arguments.get("output_spec") or {}
    execution_spec = arguments.get("execution_spec") or {}
    plan_has_transformations = bool(
        isinstance(output_spec, Mapping)
        and isinstance(output_spec.get("transformations"), list)
        and output_spec.get("transformations")
    )
    plan_has_operations = bool(
        isinstance(execution_spec, Mapping)
        and isinstance(execution_spec.get("operations"), list)
        and execution_spec.get("operations")
    )
    unresolved_binding_issues = _unresolved_binding_issues(intent)
    if unresolved_binding_issues and (plan_has_transformations or plan_has_operations):
        return _plan_error(
            request,
            (
                "unresolved source/field binding issues cannot be executed as a "
                "plan contract; collect direct field, value, row, or extraction "
                f"evidence first: {unresolved_binding_issues[:3]}."
            ),
        )
    knowledge_status = str(evidence.get("knowledge_status") or "")
    context_sources = evidence.get("context_sources") or []
    requested_sources = {
        str(item.get("path") or "").replace("\\", "/")
        for item in context_sources
        if isinstance(item, dict) and str(item.get("path") or "").strip()
    }
    unobserved_sources = requested_sources - discovery.context_sources
    if discovery.knowledge_present:
        unobserved_sources.discard("/context/knowledge.md")
    if unobserved_sources:
        return _plan_error(
            request,
            (
                "context sources must come from successful discovery calls; "
                f"unobserved sources: {sorted(unobserved_sources)}."
            ),
        )
    if knowledge_status == "authoritative" and (
        not discovery.knowledge_present or not discovery.knowledge_available
    ):
        return _plan_error(
            request,
            "knowledge cannot be authoritative because knowledge.md was unavailable.",
        )

    knowledge_facts = parse_knowledge_content(discovery.knowledge_content)
    knowledge_facts_by_id = {fact.fact_id: fact for fact in knowledge_facts}
    field_definition_facts = [
        fact
        for fact in knowledge_facts
        if _knowledge_fact_defines_field(fact)
    ]
    field_definition_facts_by_quote: dict[str, list[Any]] = {}
    for fact in field_definition_facts:
        field_definition_facts_by_quote.setdefault(fact.quote, []).append(fact)
    narrative_sources = _observed_narrative_sources_by_source_hint(request.state)
    observed_sources_by_source_hint = _observed_sources_by_source_hint(request.state)
    if knowledge_status in {"unavailable", "invalid"}:
        observed_fact_sources = sorted(
            {
                path
                for fact in knowledge_facts
                for path in narrative_sources.get(
                    _normalized_quote_text(fact.section_key or ""),
                    [],
                )
            }
        )
        if observed_fact_sources:
            return _plan_error(
                request,
                (
                    "knowledge facts with observed narrative sources cannot be "
                    "marked unavailable or invalid; include the narrative source "
                    "in evidence.context_sources and plan extraction from it: "
                    f"{observed_fact_sources}."
                ),
            )
    knowledge_rules = evidence.get("knowledge_rules") or []
    knowledge_rule_quotes: set[str] = set()
    cited_field_definition_facts: list[Any] = []
    for rule in knowledge_rules:
        if not isinstance(rule, dict):
            continue
        source_path = str(rule.get("source_path") or "").replace("\\", "/")
        quote = str(rule.get("quote") or "").strip()
        fact_id = str(rule.get("fact_id") or "").strip()
        if fact_id:
            fact = knowledge_facts_by_id.get(fact_id)
            if fact is None:
                if not (
                    source_path.lower() == "/context/knowledge.md"
                    and quote
                    and quote in discovery.knowledge_content
                ):
                    return _plan_error(
                        request,
                        f"knowledge rule cites unknown KnowledgeFact.fact_id {fact_id!r}.",
                    )
            else:
                if source_path and source_path.lower() != fact.source_path.lower():
                    return _plan_error(
                        request,
                        (
                            f"knowledge rule fact_id {fact_id!r} must use source_path "
                            f"{fact.source_path!r}."
                        ),
                    )
                quote = quote or fact.quote
                source_path = fact.source_path
                if _knowledge_fact_defines_field(fact):
                    cited_field_definition_facts.append(fact)
        elif quote:
            cited_field_definition_facts.extend(
                field_definition_facts_by_quote.get(quote)
                or _field_facts_for_knowledge_quote(
                    quote=quote,
                    field_facts=field_definition_facts,
                )
            )
        if (
            source_path.lower() != "/context/knowledge.md"
            or not quote
            or quote not in discovery.knowledge_content
        ):
            return _plan_error(
                request,
                (
                    "every knowledge rule must quote text observed in "
                    "/context/knowledge.md."
                ),
            )
        knowledge_rule_quotes.add(quote)

    output_spec = arguments.get("output_spec") or {}
    if not isinstance(output_spec, Mapping):
        return _plan_error(request, "output_spec must be a JSON object.")
    output_columns = output_spec.get("columns") or []
    if not isinstance(output_columns, list):
        return _plan_error(request, "output_spec.columns must be a list.")
    transformations = output_spec.get("transformations") or []
    row_policy = str(output_spec.get("row_policy") or "")
    execution_spec = arguments.get("execution_spec") or {}
    if execution_spec and not isinstance(execution_spec, Mapping):
        return _plan_error(request, "execution_spec must be an object when provided.")
    declared_operations = {
        str(item.get("operation") or "").casefold()
        for item in transformations
        if isinstance(item, Mapping)
    }
    if isinstance(execution_spec, Mapping):
        declared_operations.update(
            str(item.get("operation") or "").casefold()
            for item in execution_spec.get("operations") or []
            if isinstance(item, Mapping)
        )
    required_types = {
        requirement_type
        for types in effective_requirement_types_by_quote.values()
        for requirement_type in types
    }
    missing_operation_requirements: list[str] = []
    if required_types & {"filter", "time_range"} and "filter" not in declared_operations:
        missing_operation_requirements.append("filter")
    if "calculation" in required_types and not (
        declared_operations & {"aggregate", "derive"}
    ):
        missing_operation_requirements.append("aggregate/derive")
    if "ordering" in required_types and "sort" not in declared_operations:
        missing_operation_requirements.append("sort")
    if "limit" in required_types and "limit" not in declared_operations:
        missing_operation_requirements.append("limit")
    if missing_operation_requirements:
        return _plan_error(
            request,
            (
                "intent requirements declare executable operations but the plan "
                "does not declare them in output_spec.transformations or "
                "execution_spec.operations: "
                f"{sorted(set(missing_operation_requirements))}."
            ),
        )
    unresolved_scope_quotes = _unresolved_explicit_scope_quotes(
        question_structure=request.state.get("question_structure"),
        arguments=arguments,
    )
    if unresolved_scope_quotes:
        return _plan_error(
            request,
            (
                "explicit user scope constraints are unresolved; bind them to "
                "observed source fields/values before planning or answering: "
                f"{unresolved_scope_quotes}."
            ),
        )
    execution_sources = _source_paths_from_execution_spec(execution_spec)
    if isinstance(execution_spec, Mapping):
        invalid_fact_bindings: list[str] = []
        for binding in execution_spec.get("source_bindings") or []:
            if not isinstance(binding, Mapping):
                continue
            fact_id = str(binding.get("fact_id") or "").strip()
            source_field = str(binding.get("source_field") or "").strip()
            fact = knowledge_facts_by_id.get(fact_id)
            if (
                fact_id
                and source_field
                and fact is not None
                and _knowledge_fact_defines_field(fact)
                and _normalized_field_alias(source_field)
                not in _knowledge_fact_field_aliases(fact)
            ):
                invalid_fact_bindings.append(
                    f"{fact_id}:{source_field}->{getattr(fact, 'field_key', '')}"
                )
        if invalid_fact_bindings:
            return _plan_error(
                request,
                (
                    "source_bindings that cite field KnowledgeFact.fact_id must "
                    "bind source_field to the fact field_key or an explicitly "
                    "quoted field alias: "
                    f"{sorted(invalid_fact_bindings)}."
                ),
            )
    if knowledge_status == "authoritative":
        plan_sources_for_hints = requested_sources | execution_sources
        missing_by_group = _missing_relevant_source_hint_groups(
            knowledge_facts=knowledge_facts,
            state=request.state,
            plan_sources=plan_sources_for_hints,
        )
        if missing_by_group:
            return _plan_error(
                request,
                (
                    "unresolved source/field binding: request-relevant "
                    "knowledge facts provide exact source hints that the "
                    "plan sources do not satisfy; collect direct evidence "
                    "from the hinted sources before using alternative "
                    f"sources: {missing_by_group[:3]}."
                ),
            )
        output_source_fields = {
            _normalized_field_alias(field)
            for column in output_columns
            if isinstance(column, Mapping)
            for field in column.get("source_fields") or []
            if str(field or "").strip()
        }
        binding_source_fields = {
            _normalized_field_alias(binding.get("source_field"))
            for binding in execution_spec.get("source_bindings") or []
            if isinstance(binding, Mapping)
            and str(binding.get("source_field") or "").strip()
        }
        supporting_source_fields = {
            _normalized_field_alias(field)
            for item in execution_spec.get("supporting_fields") or []
            if isinstance(item, Mapping)
            for field in item.get("source_fields") or []
            if str(field or "").strip()
        }
        cited_fact_ids = {str(fact.fact_id) for fact in cited_field_definition_facts}
        explicit_cited_target_facts = [
            fact
            for fact in cited_field_definition_facts
            if len(cited_field_definition_facts) == 1
            or _normalized_field_alias(fact.field_key) in output_source_fields
            or _normalized_field_alias(fact.field_key) in binding_source_fields
        ]
        request_target_field_facts: list[Any] = []
        seen_request_target_fact_ids: set[str] = set()
        for fact in explicit_cited_target_facts:
            if str(fact.fact_id) in seen_request_target_fact_ids:
                continue
            request_target_field_facts.append(fact)
            seen_request_target_fact_ids.add(str(fact.fact_id))
        undiscovered_target_fields = sorted(
            {
                f"{fact.section_key}.{fact.field_key}"
                for fact in request_target_field_facts
                if fact.section_key
                and _fact_has_time_terms(fact)
                if not observed_sources_by_source_hint.get(
                    _normalized_quote_text(fact.section_key or ""),
                    [],
                )
            }
        )
        if undiscovered_target_fields:
            return _plan_error(
                request,
                (
                    "request-target knowledge field facts require physical "
                    "binding discovery before planning; run query_schema or "
                    "inspect the source hint first: "
                    f"{undiscovered_target_fields}."
                ),
            )
        observed_target_field_facts = [
            fact
            for fact in request_target_field_facts
            if observed_sources_by_source_hint.get(
                _normalized_quote_text(fact.section_key or ""),
                [],
            )
        ]
        uncited_target_fields = sorted(
            {
                f"{fact.section_key}.{fact.field_key}"
                for fact in observed_target_field_facts
                if str(fact.fact_id) not in cited_fact_ids
                and _field_fact_has_observed_physical_field(
                    fact=fact,
                    state=request.state,
                )
            }
        )
        if uncited_target_fields:
            return _plan_error(
                request,
                (
                    "authoritative plans must cite request-target knowledge "
                    "field facts by exact quote or fact_id when their narrative "
                    f"sources were observed: {uncited_target_fields}."
                ),
            )
        target_field_facts = [
            fact
            for fact in observed_target_field_facts
            if str(fact.fact_id) in cited_fact_ids
        ]
        physical_target_field_facts = [
            fact
            for fact in target_field_facts
            if _field_fact_has_observed_physical_field(
                fact=fact,
                state=request.state,
            )
        ]
        physical_target_fact_ids = {
            str(fact.fact_id)
            for fact in physical_target_field_facts
        }
        narrative_target_field_facts = [
            fact
            for fact in target_field_facts
            if str(fact.fact_id) not in physical_target_fact_ids
            and narrative_sources.get(
                _normalized_quote_text(fact.section_key or ""),
                [],
            )
        ]
        missing_physical_field_keys = sorted(
            {
                str(fact.field_key)
                for fact in physical_target_field_facts
                if _normalized_field_alias(fact.field_key)
                not in output_source_fields
                and _normalized_field_alias(fact.field_key)
                not in binding_source_fields
                and _normalized_field_alias(fact.field_key)
                not in supporting_source_fields
            }
        )
        if missing_physical_field_keys:
            return _plan_error(
                request,
                (
                    "output columns authorized by observed structured knowledge "
                    "field facts must cite those physical field keys in "
                    "source_fields; do not replace them with semantic-neighbor "
                    f"fields: {missing_physical_field_keys}."
                ),
            )
        plan_sources = requested_sources | execution_sources
        missing_source_bindings = sorted(
            {
                f"{fact.section_key}.{fact.field_key}"
                for fact in target_field_facts
                if narrative_sources.get(
                    _normalized_quote_text(fact.section_key or ""),
                    [],
                )
                and not _field_fact_has_observed_narrative_extraction(
                    fact=fact,
                    state=request.state,
                )
                if not (
                    set(
                        narrative_sources.get(
                            _normalized_quote_text(fact.section_key or ""),
                            [],
                        )
                    )
                    & plan_sources
                )
            }
        )
        if missing_source_bindings:
            return _plan_error(
                request,
                (
                    "request-target knowledge field facts must be bound to their "
                    "observed narrative source in evidence.context_sources or "
                    f"execution_spec.sources: {missing_source_bindings}."
                ),
            )
        authorized_output_field_keys = {
            _normalized_field_alias(fact.field_key)
            for fact in target_field_facts
            if str(fact.field_key or "").strip()
        }
        if authorized_output_field_keys:
            requested_output_texts = _question_requested_output_texts(request.state)
            unauthorized_measure_outputs = []
            for column in output_columns:
                if not isinstance(column, Mapping):
                    continue
                role = str(column.get("role") or "").casefold()
                if role not in {"measure", "metric"}:
                    continue
                column_aliases = _column_field_aliases(column)
                if column_aliases & authorized_output_field_keys:
                    continue
                if _question_requests_distribution(request.state) and _column_is_count_measure(
                    column,
                ):
                    continue
                if _column_matches_requested_output(column, requested_output_texts):
                    continue
                unauthorized_measure_outputs.append(
                    str(column.get("name") or sorted(column_aliases) or column)
                )
            if unauthorized_measure_outputs:
                return _plan_error(
                    request,
                    (
                        "final measure output columns must be authorized by "
                        "the user target or a request-target knowledge field "
                        "fact; move unrequested same-source fields to "
                        "execution_spec.supporting_fields: "
                        f"{unauthorized_measure_outputs}."
                    ),
                )
    if row_policy == "preserve" and not transformations:
        missing_source_fields = [
            str(column.get("name") or "")
            for column in output_columns
            if isinstance(column, Mapping)
            and not [
                field
                for field in column.get("source_fields") or []
                if str(field or "").strip()
            ]
        ]
        if missing_source_fields:
            return _plan_error(
                request,
                (
                    "preserve output columns must cite source_fields from the "
                    f"observed source projection: {missing_source_fields}."
                ),
            )
    target_output_columns = output_columns
    if isinstance(execution_spec, Mapping):
        unobserved_execution_sources = execution_sources - discovery.context_sources
        if discovery.knowledge_present:
            unobserved_execution_sources.discard("/context/knowledge.md")
        if unobserved_execution_sources:
            return _plan_error(
                request,
                (
                    "execution_spec.sources must come from successful discovery "
                    f"calls; unobserved sources: {sorted(unobserved_execution_sources)}."
                ),
            )
        output_field_names = {
            str(column.get("name") or "").casefold()
            for column in target_output_columns
            if isinstance(column, Mapping)
        }
        for column in target_output_columns:
            if isinstance(column, Mapping):
                output_field_names.update(
                    str(field or "").casefold()
                    for field in column.get("source_fields") or []
                    if str(field or "").strip()
                )
        for field in execution_spec.get("supporting_fields") or []:
            if not isinstance(field, Mapping):
                continue
            supporting_names = {
                str(field.get("name") or "").casefold(),
                *(
                    str(item or "").casefold()
                    for item in field.get("source_fields") or []
                    if str(item or "").strip()
                ),
            }
            overlap = sorted(output_field_names & {name for name in supporting_names if name})
            if overlap:
                return _plan_error(
                    request,
                    (
                        "selector/filter/join/context fields declared in "
                        "execution_spec.supporting_fields must not also appear in "
                        f"final output_spec.columns: {overlap}."
                    ),
                )
    if transformations and row_policy != "transform":
        return _plan_error(
            request,
            "row_policy must be transform when transformations are present.",
        )
    if not transformations:
        if row_policy != "preserve":
            return _plan_error(
                request,
                "row_policy must be preserve when no transformation is authorized.",
            )
        if output_spec.get("ordering") != "source":
            return _plan_error(
                request,
                "preserve plans must keep source ordering.",
            )
        if output_spec.get("null_policy") != "preserve":
            return _plan_error(
                request,
                "preserve plans must keep source null values.",
            )
        if output_spec.get("sort_keys"):
            return _plan_error(
                request,
                "preserve plans cannot define sort keys.",
            )

    def user_authorization_error(operation: str, quote: str) -> str | None:
        if not quote or quote not in original_request:
            return (
                "user authorization must cite an exact quote from "
                "original_request."
            )
        if not _user_quote_authorizes_operation(
            operation=operation,
            quote=quote,
            arguments=arguments,
            state=request.state,
        ):
            return (
                "user authorization quote is not authorized for this operation "
                "by the question-structure contract."
            )
        return None

    def knowledge_authorization_error(operation: str, quote: str) -> str | None:
        if not quote or quote not in knowledge_rule_quotes:
            return (
                "knowledge authorization must cite an observed knowledge quote."
            )
        operation_name = operation.casefold()
        quoted_facts = [
            fact
            for fact in knowledge_facts
            if fact.quote == quote
        ]
        if not any(
            operation_name in _knowledge_fact_operation_names(fact)
            for fact in quoted_facts
        ):
            return (
                "knowledge transformation authorization must cite a knowledge "
                "quote or fact_id with an operation matching the transformation."
            )
        return None

    def fact_authorizes_operation(operation: str, fact_id: str) -> bool:
        fact = knowledge_facts_by_id.get(fact_id)
        if fact is None:
            return False
        return operation.casefold() in _knowledge_fact_operation_names(fact)

    authorized_transformation_operations: set[str] = set()
    for transformation in transformations:
        if not isinstance(transformation, dict):
            return _plan_error(request, "transformations must be structured objects.")
        operation = str(transformation.get("operation") or "")
        authorization = transformation.get("authorization") or {}
        if not isinstance(authorization, Mapping):
            continue
        raw_fact_ids = transformation.get("authorization_fact_ids")
        fact_id_values = raw_fact_ids if isinstance(raw_fact_ids, list) else []
        fact_ids = [
            str(item).strip()
            for item in fact_id_values
            if str(item).strip()
        ]
        fact_authorized = any(
            fact_authorizes_operation(operation, fact_id)
            for fact_id in fact_ids
        )
        source = str(authorization.get("source") or "")
        quote = str(authorization.get("quote") or "").strip()
        if source == "user":
            if error_message := user_authorization_error(operation, quote):
                return _plan_error(request, error_message)
        if source == "knowledge":
            if (
                error_message := knowledge_authorization_error(operation, quote)
            ) and not fact_authorized:
                return _plan_error(request, error_message)
        if source not in {"user", "knowledge"}:
            return _plan_error(
                request,
                "context evidence cannot authorize a transformation.",
            )
        if operation:
            authorized_transformation_operations.add(operation.casefold())

    if isinstance(execution_spec, Mapping):
        for index, operation_item in enumerate(execution_spec.get("operations") or []):
            if not isinstance(operation_item, Mapping):
                return _plan_error(
                    request,
                    f"execution_spec.operations[{index}] must be an object.",
                )
            operation = str(operation_item.get("operation") or "")
            if not operation:
                return _plan_error(
                    request,
                    f"execution_spec.operations[{index}].operation is required.",
                )
            if operation == "join":
                left_source = str(operation_item.get("left_source") or "").replace(
                    "\\", "/"
                )
                right_source = str(operation_item.get("right_source") or "").replace(
                    "\\", "/"
                )
                left_key = str(operation_item.get("left_key") or "").strip()
                right_key = str(operation_item.get("right_key") or "").strip()
                if not (left_source and right_source and left_key and right_key):
                    return _plan_error(
                        request,
                        (
                            "join operations require left_source, right_source, "
                            "left_key, and right_key."
                        ),
                    )
                undeclared_join_sources = {
                    source
                    for source in {left_source, right_source}
                    if source not in execution_sources
                }
                if undeclared_join_sources:
                    return _plan_error(
                        request,
                        (
                            "join sources must be declared in execution_spec.sources: "
                            f"{sorted(undeclared_join_sources)}."
                        ),
                    )
            authorized = False
            authorization = operation_item.get("authorization")
            if isinstance(authorization, Mapping):
                source = str(authorization.get("source") or "")
                quote = str(authorization.get("quote") or "").strip()
                if source == "user":
                    authorized = user_authorization_error(operation, quote) is None
                elif source == "knowledge":
                    authorized = knowledge_authorization_error(operation, quote) is None
                elif source:
                    return _plan_error(
                        request,
                        "context evidence cannot authorize an execution operation.",
                    )
            fact_ids = [
                str(item).strip()
                for item in operation_item.get("authorization_fact_ids") or []
                if str(item).strip()
            ]
            if not authorized and fact_ids:
                authorized = any(
                    fact_authorizes_operation(operation, fact_id)
                    for fact_id in fact_ids
                )
            if not authorized and operation.casefold() in authorized_transformation_operations:
                authorized = True
            if not authorized:
                return _plan_error(
                    request,
                    (
                        f"execution_spec.operations[{index}] for {operation!r} "
                        "requires an exact user quote, an observed knowledge rule, "
                        "or a valid KnowledgeFact.fact_id."
                    ),
                )
    revision = arguments.get("revision") or {}
    if not isinstance(revision, Mapping):
        return _plan_error(request, "revision must be a JSON object.")
    version = revision.get("version")
    reported_changes = {
        str(field).strip()
        for field in revision.get("changed_fields") or []
        if str(field).strip()
    }
    previous_plan = request.state.get("analysis_plan")
    if previous_plan is None:
        if version != 1:
            return _plan_error(request, "the initial plan must use revision version 1.")
        if reported_changes:
            return _plan_error(
                request,
                "the initial plan cannot report changed_fields.",
            )
        return None

    previous_version = (previous_plan.get("revision") or {}).get("version")
    if not isinstance(previous_version, int) or version != previous_version + 1:
        return _plan_error(
            request,
            "a revised plan must increment revision.version by exactly one.",
        )
    previous_requirements = (
        previous_plan.get("intent", {}).get("requirements") or []
    )
    if any(item not in requirements for item in previous_requirements):
        return _plan_error(
            request,
            "a revision cannot remove or rewrite existing user requirements.",
        )

    actual_changes = {
        field
        for field in _REVISION_FIELDS
        if previous_plan.get(field) != arguments.get(field)
    }
    if not actual_changes:
        return None
    missing_changes = actual_changes - reported_changes
    if missing_changes:
        return _plan_error(
            request,
            f"revision.changed_fields omits: {sorted(missing_changes)}.",
        )
    if "evidence" in actual_changes and not revision.get("evidence_changes"):
        return _plan_error(
            request,
            "evidence changes must be described in revision.evidence_changes.",
        )
    return None


class AnswerMiddleware(AgentMiddleware[BenchmarkDeepAgentState, None, Any]):
    """End the graph once a prepared answer is available."""

    state_schema = BenchmarkDeepAgentState

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: BenchmarkDeepAgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        del runtime
        if state.get("answer") is not None:
            return {"jump_to": "end"}
        prepared_answer = state.get("prepared_answer")
        if prepared_answer is not None:
            return {
                "answer": prepared_answer,
                "jump_to": "end",
            }
        return None


class CustomSystemPromptMiddleware(AgentMiddleware[Any, None, Any]):
    """Replace SDK system prompts after tool injection."""

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self.prompt = prompt.strip()

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        if not self.prompt:
            return handler(request)
        prompt = self.prompt
        try:
            boundary_text = _decision_evidence_boundary_text(
                request.state,
                _discovery_state(request.messages, request.state),
            )
        except Exception:
            boundary_text = ""
        if boundary_text and "</evidence_boundary>" not in prompt:
            prompt = f"{prompt}\n\n{boundary_text}"
        return handler(
            request.override(system_message=SystemMessage(content=prompt.strip()))
        )


class PlanningMiddleware(AgentMiddleware[BenchmarkDeepAgentState, None, Any]):
    """Enforce Discovery -> Plan -> Todos and allow revisions."""

    state_schema = BenchmarkDeepAgentState
    tools = [analyze_plan_tool]

    def before_model(
        self,
        state: BenchmarkDeepAgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        del runtime
        discovery = _discovery_state(state.get("messages", []), state)
        contract = _evidence_contract_payload(state, discovery)
        if state.get("evidence_contract") == contract:
            return None
        return {"evidence_contract": contract}

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        discovery = _discovery_state(request.messages, request.state)
        allowed_tools_for_response: set[str] | None = None
        forced_tool_for_response: str | None = None
        if request.state.get("analysis_plan") is None:
            messages = request.state.get("messages")
            if isinstance(messages, list) and _last_narrative_extraction_incomplete(
                messages,
            ):
                allowed_tools = {"extract_narrative_records"}
                tool_choice = "extract_narrative_records"
            elif _answer_candidate_has_complete_columns(
                request.state.get("answer_candidate"),
            ):
                allowed_tools = {"analyze_plan"}
                tool_choice = "analyze_plan"
            elif _pre_plan_needs_narrative_extraction(
                request.state,
            ) or _pre_plan_observed_narrative_hint_needs_extraction(
                request.state,
                discovery,
            ):
                allowed_tools = {"extract_narrative_records"}
                tool_choice = "extract_narrative_records"
            elif _pre_plan_execution_gate_active(
                state=request.state,
                discovery=discovery,
            ):
                allowed_tools = {"analyze_plan"}
                tool_choice = "analyze_plan"
            else:
                allowed_tools, tool_choice = discovery.tool_policy()
            allowed_tools_for_response = set(allowed_tools)
            forced_tool_for_response = tool_choice if isinstance(tool_choice, str) else None
            request = request.override(
                tools=[
                    item
                    for item in request.tools
                    if tool_name(item) in allowed_tools
                ],
                tool_choice=tool_choice,
            )
        elif request.state.get("answer_candidate") is not None:
            allowed_tools_for_response = set(_ANSWER_CANDIDATE_RECOVERY_TOOLS)
            request = request.override(
                tools=[
                    item
                    for item in request.tools
                    if tool_name(item) in _ANSWER_CANDIDATE_RECOVERY_TOOLS
                ],
                tool_choice=None,
            )
        elif not request.state.get("todos"):
            todo_tools = [
                item for item in request.tools if tool_name(item) == "write_todos"
            ]
            allowed_tools_for_response = {"write_todos"}
            forced_tool_for_response = "write_todos"
            request = request.override(tools=todo_tools, tool_choice="write_todos")
        elif _plan_needs_initial_narrative_extraction(request.state):
            allowed_tools_for_response = {"extract_narrative_records"}
            forced_tool_for_response = "extract_narrative_records"
            request = request.override(
                tools=[
                    item
                    for item in request.tools
                    if tool_name(item) == "extract_narrative_records"
                ],
                tool_choice="extract_narrative_records",
            )
        request = _inject_decision_evidence_boundary(request, discovery)
        response = handler(request)
        if allowed_tools_for_response is not None:
            response = _constrain_model_tool_calls(
                response,
                allowed_tool_names=allowed_tools_for_response,
                forced_tool_name=forced_tool_for_response,
            )
            response = _retry_missing_forced_tool_call(
                request,
                handler,
                response,
                forced_tool_name=forced_tool_for_response,
            )
            response = _constrain_model_tool_calls(
                response,
                allowed_tool_names=allowed_tools_for_response,
                forced_tool_name=forced_tool_for_response,
            )
        return _retry_invalid_tool_call(request, handler, response)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        current_tool_name = str(request.tool_call.get("name") or "")
        if current_tool_name in {"analyze_plan", "extract_narrative_records"}:
            request = _normalize_tool_call_arguments(request)
        if current_tool_name == "extract_narrative_records":
            request = _strip_extract_narrative_protocol_markup(request)
        if _tool_arguments_contain_protocol_markup(request.tool_call.get("args")):
            return _tool_error(
                request,
                (
                    "Tool arguments contain protocol markup remnants. "
                    "Reissue exactly one tool call with valid JSON arguments."
                ),
            )
        plan = request.state.get("analysis_plan")
        discovery = (
            _discovery_state(request.state["messages"], request.state)
            if plan is None or current_tool_name == "analyze_plan"
            else None
        )
        if current_tool_name == "extract_narrative_records":
            request = _canonicalize_extract_narrative_arguments(
                request,
                discovery or _discovery_state(request.state["messages"], request.state),
            )

        if plan is None and discovery is not None:
            if current_tool_name not in _SOURCE_DISCOVERY_TOOLS | {"analyze_plan"}:
                return _tool_error(
                    request,
                    "Only discovery tools are available before analyze_plan.",
                )
            if (
                current_tool_name != "analyze_plan"
                and _pre_plan_execution_gate_active(
                    state=request.state,
                    discovery=discovery,
                )
                and not (
                    current_tool_name == "extract_narrative_records"
                    and (
                        _pre_plan_needs_narrative_extraction(request.state)
                        or _pre_plan_observed_narrative_hint_needs_extraction(
                            request.state,
                            discovery,
                        )
                    )
                )
            ):
                return _tool_error(
                    request,
                    (
                        "Observed evidence is sufficient for an execution contract. "
                        "Call analyze_plan before running more discovery or execution tools."
                    ),
                )
        if current_tool_name == "analyze_plan":
            assert discovery is not None
            if not (discovery.knowledge_ready and discovery.context_ready):
                return _tool_error(
                    request,
                    (
                        "Complete discovery before analyze_plan: inspect the minimum "
                        "required independent data sources."
                    ),
            )
            request = _canonicalize_plan_quotes(request, discovery)
            request = _canonicalize_empty_authoritative_knowledge(request, discovery)
            request = _canonicalize_missing_intent_requirements(request)
            request = _canonicalize_unsupported_operations(request)
            request = _canonicalize_authorized_plan_operations(request)
            request = _canonicalize_distribution_output_columns(request)
            request = _canonicalize_direct_source_projection_plan(request)
            request = _canonicalize_semantic_source_bindings(request, discovery)
            request = _canonicalize_output_columns_from_valid_field_bindings(
                request,
                discovery,
            )
            request = _canonicalize_output_columns_from_section_field_facts(
                request,
                discovery,
            )
            request = _canonicalize_single_preserve_output_from_field_fact(
                request,
                discovery,
            )
            request = _canonicalize_duplicate_output_columns(request)
            request = _canonicalize_unrequested_measure_output_columns(request)
            request = _canonicalize_selector_output_columns(request)
            request = _canonicalize_unrequested_key_output_columns(request)
            request = _canonicalize_unrequested_knowledge_output_columns(
                request,
                discovery,
            )
            request = _canonicalize_preserve_expected_row_count(request)
            request = _canonicalize_transform_output_policy(request)
            request = _canonicalize_preserve_output_policy(request)
            request = _canonicalize_unbacked_sort_keys(request)
            request = _canonicalize_sort_null_policy(request)
            request = _canonicalize_plan_steps(request)
            request = _canonicalize_execution_supporting_fields(request)
            request = _canonicalize_unsatisfied_authoritative_source_hints(
                request,
                discovery,
            )
            request = _canonicalize_revision(request)
            request = _canonicalize_non_authoritative_knowledge(request)
            contract_error = _validate_plan_contract(request, discovery)
            if contract_error is not None:
                return contract_error
        if current_tool_name == "write_todos" and plan is not None:
            normalized_todos = _normalize_write_todos_arguments(request, plan)
            if isinstance(normalized_todos, ToolMessage):
                return normalized_todos
            request = normalized_todos
        if (
            plan is not None
            and not request.state.get("todos")
            and current_tool_name != "write_todos"
            and not (
                request.state.get("answer_candidate") is not None
                and current_tool_name in _ANSWER_CANDIDATE_RECOVERY_TOOLS
            )
        ):
            return _tool_error(
                request,
                "Call write_todos successfully before using any other tool.",
            )
        result = handler(request)
        if current_tool_name == "analyze_plan":
            return _promote_answer_candidate_after_plan(request, result)
        return result


class DisabledToolGuardMiddleware(AgentMiddleware[Any, None, Any]):
    """Reject built-in tools excluded from the benchmark harness."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        current_tool_name = str(request.tool_call.get("name") or "")
        if current_tool_name in DISABLED_BUILTIN_TOOLS:
            return _tool_error(
                request,
                (
                    f"Tool `{current_tool_name}` is disabled in this benchmark "
                    "agent. Use the exposed data tools instead; for large JSON "
                    "or CSV sources, call the original structured reader again "
                    "with pagination rather than using read_file."
                ),
            )
        return handler(request)


def workspace_permissions() -> list[FilesystemPermission]:
    """Allow read-only context access; scratch is managed by the backend."""

    context_paths = ["/context/**"]
    return [
        FilesystemPermission(operations=["read"], paths=context_paths, mode="allow"),
        FilesystemPermission(operations=["write"], paths=context_paths, mode="deny"),
    ]
