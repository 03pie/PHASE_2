from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
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
_CONTEXT_PATH_PATTERN = re.compile(r"""["'](/context/[^"']+)["']""")
_INJECTED_KNOWLEDGE_PATTERN = re.compile(
    r"<context_knowledge>\s*(.*?)\s*</context_knowledge>",
    re.DOTALL,
)
_CJK_SEQUENCE_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
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
_OPERATION_REQUIREMENT_TYPES: dict[str, frozenset[str]] = {
    "filter": frozenset({"filter", "time_range"}),
    "aggregate": frozenset({"calculation", "grouping"}),
    "derive": frozenset({"calculation"}),
    "sort": frozenset({"ordering"}),
    "limit": frozenset({"limit", "ordering"}),
    "deduplicate": frozenset({"deduplication"}),
    "reshape": frozenset({"reshape"}),
}
_AGGREGATE_SELECTOR_PATTERN = re.compile(
    r"(?i)\b(count|how many|number of|sum|total|average|avg|mean|"
    r"maximum|max|minimum|min|highest|lowest|largest|smallest|top|bottom)\b|"
    r"(多少|几个|几位|数量|总数|合计|平均|最大|最小|最高|最低|最多|最少|前\d+|后\d+)"
)
_DERIVE_PATTERN = re.compile(
    r"(?i)\b(calculate|compute|derive|ratio|rate|share|percent|percentage)\b|"
    r"(计算|占比|比例|比率|百分比|增速|增长率)"
)
_SPECIFIC_FILTER_PATTERN = re.compile(r"\d|[A-Z]{2,}|\b[A-Z]\b")
_GENERIC_SCOPE_TERMS = frozenset(
    {
        "我国",
        "中国",
        "全国",
        "国内",
        "记录",
        "数据",
        "数据记录",
        "这些数据",
        "这些记录",
    }
)
_GENERIC_SEMANTIC_TOKENS = frozenset(
    {"and", "data", "record", "records", "value", "values", "other"}
)
_IDENTITY_OUTPUT_PATTERN = re.compile(
    r"(?i)\b(which|who|name|code|ticker|symbol|abbr|abbreviation|identifier)\b|"
    r"(哪|哪个|谁|名称|姓名|名字|简称|代码|编号|证券代码|股票简称)"
)
_TIME_OUTPUT_PATTERN = re.compile(
    r"(?i)\b(date|time|timestamp|year|month|quarter|period|when)\b|"
    r"(日期|时间|时间戳|年份|年度|月份|季度|报告期|哪年|何时)"
)
_TIME_CONTEXT_FIELD_PATTERN = re.compile(
    r"(?i)(date|time|timestamp|year|month|quarter|period|reportperiod|enddate)"
)
_IDENTITY_CONTEXT_FIELD_PATTERN = re.compile(
    r"(?i)(name|code|abbr|id|identifier|symbol|ticker|province|region|area|"
    r"secuabbr|secucode|stockcode)"
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


def _tool_error(request: ToolCallRequest, content: str) -> ToolMessage:
    return ToolMessage(
        content=content,
        name=str(request.tool_call.get("name") or ""),
        tool_call_id=str(request.tool_call.get("id") or ""),
        status="error",
    )


def _decode_json_like_argument(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in {"{", "["}:
            try:
                return _decode_json_like_argument(json.loads(stripped))
            except json.JSONDecodeError:
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


def _normalize_tool_call_arguments(request: ToolCallRequest) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    normalized_arguments = _decode_json_like_argument(arguments)
    if not isinstance(normalized_arguments, Mapping):
        return request
    if str(request.tool_call.get("name") or "") == "analyze_plan":
        normalized_arguments = dict(normalized_arguments)
        execution_spec = normalized_arguments.get("execution_spec")
        if execution_spec is not None and not isinstance(execution_spec, Mapping):
            normalized_arguments.pop("execution_spec", None)
    if normalized_arguments is arguments:
        return request
    return request.override(
        tool_call={
            **request.tool_call,
            "args": dict(normalized_arguments),
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
    output_spec = analysis_plan.get("output_spec") or {}
    expected_columns = []
    if isinstance(output_spec, Mapping):
        expected_columns = [
            column
            for column in output_spec.get("columns") or []
            if isinstance(column, Mapping)
        ]
    if expected_columns and len(columns) != len(expected_columns):
        return None
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


def _observed_narrative_sources_by_logical(
    state: Mapping[str, Any],
) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}

    def add_source(logical_name: str, path: str) -> None:
        normalized_name = _normalized_quote_text(logical_name)
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
        logical_name = str(source.get("logical_name") or "")
        if not logical_name or not path:
            continue
        add_source(logical_name, path)

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
            logical_name = filename.rsplit(".", 1)[0]
            add_source(logical_name, path)
    return sources


def _observed_sources_by_logical(
    state: Mapping[str, Any],
) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}
    for source in _state_observed_sources(state):
        path = str(source.get("path") or "").replace("\\", "/")
        if not path:
            continue
        logical_names = {
            _normalized_quote_text(str(source.get("logical_name") or "")),
            _normalized_quote_text(str(source.get("table") or "")),
        }
        path_tail = path.rsplit("::", 1)[-1].rsplit("/", 1)[-1]
        logical_names.add(_normalized_quote_text(path_tail.rsplit(".", 1)[0]))
        for logical_name in {name for name in logical_names if name}:
            sources.setdefault(logical_name, [])
            if path not in sources[logical_name]:
                sources[logical_name].append(path)
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


def _canonicalize_execution_supporting_fields(
    request: ToolCallRequest,
) -> ToolCallRequest:
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    output_spec = arguments.get("output_spec")
    execution_spec = arguments.get("execution_spec")
    if not isinstance(output_spec, Mapping) or not isinstance(execution_spec, Mapping):
        return request

    output_field_names = {
        str(column.get("name") or "").casefold()
        for column in output_spec.get("columns") or []
        if isinstance(column, Mapping)
    }
    for column in output_spec.get("columns") or []:
        if not isinstance(column, Mapping):
            continue
        output_field_names.update(
            str(field or "").casefold()
            for field in column.get("source_fields") or []
            if str(field or "").strip()
        )
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
            str(field.get("name") or "").casefold(),
            *(
                str(item or "").casefold()
                for item in field.get("source_fields") or []
                if str(item or "").strip()
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
                "execution_spec": updated_execution_spec,
            },
        }
    )


def _question_requested_output_texts(state: Mapping[str, Any]) -> set[str]:
    question_structure = state.get("question_structure")
    if not isinstance(question_structure, Mapping):
        return set()
    texts: set[str] = set()
    output = question_structure.get("output")
    if isinstance(output, Mapping):
        texts.update(
            str(item or "").strip()
            for item in output.get("requested_columns") or []
            if str(item or "").strip()
        )
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


def _column_texts(column: Mapping[str, Any]) -> set[str]:
    texts = {str(column.get("name") or "").strip()}
    texts.update(
        str(field or "").strip()
        for field in column.get("source_fields") or []
        if str(field or "").strip()
    )
    return {text for text in texts if text}


def _column_matches_requested_output(
    column: Mapping[str, Any],
    requested_texts: set[str],
) -> bool:
    column_texts = _column_texts(column)
    if not column_texts or not requested_texts:
        return False
    for column_text in column_texts:
        normalized_column = _normalized_quote_text(column_text)
        column_tokens = _semantic_overlap_tokens(column_text)
        for requested_text in requested_texts:
            normalized_requested = _normalized_quote_text(requested_text)
            if not normalized_column or not normalized_requested:
                continue
            if normalized_column == normalized_requested:
                return True
            if normalized_column in normalized_requested or normalized_requested in normalized_column:
                return True
            if column_tokens & _semantic_overlap_tokens(requested_text):
                return True
    return False


def _should_keep_key_output_column(
    column: Mapping[str, Any],
    *,
    requested_texts: set[str],
    original_request: str,
) -> bool:
    role = str(column.get("role") or "")
    column_blob = " ".join(_column_texts(column))
    looks_time_context = bool(_TIME_CONTEXT_FIELD_PATTERN.search(column_blob))
    looks_identity_context = bool(_IDENTITY_CONTEXT_FIELD_PATTERN.search(column_blob))
    if role == "time_key" or (
        role == "output_column" and looks_time_context
    ):
        return bool(_TIME_OUTPUT_PATTERN.search(original_request)) or (
            _column_matches_requested_output(column, requested_texts)
        )
    if role in {"entity_key", "record_key"} or (
        role == "output_column" and looks_identity_context
    ):
        return bool(_IDENTITY_OUTPUT_PATTERN.search(original_request)) or (
            _column_matches_requested_output(column, requested_texts)
        )
    return True


def _looks_like_unrequested_key_output_column(column: Mapping[str, Any]) -> bool:
    column_blob = " ".join(_column_texts(column))
    return bool(
        _TIME_CONTEXT_FIELD_PATTERN.search(column_blob)
        or _IDENTITY_CONTEXT_FIELD_PATTERN.search(column_blob)
    )


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
    original_request = str(request.state.get("original_request") or "")
    kept_columns: list[Any] = []
    demoted_columns: list[Mapping[str, Any]] = []
    for column in output_columns:
        if not isinstance(column, Mapping):
            kept_columns.append(column)
            continue
        role = str(column.get("role") or "")
        if (
            role in _KEY_OUTPUT_ROLES | {"output_column"}
            or _looks_like_unrequested_key_output_column(column)
        ) and not _should_keep_key_output_column(
            column,
            requested_texts=requested_texts,
            original_request=original_request,
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


def _source_paths_from_execution_spec(execution_spec: Any) -> set[str]:
    if not isinstance(execution_spec, Mapping):
        return set()
    return {
        str(source.get("path") or "").replace("\\", "/")
        for source in execution_spec.get("sources") or []
        if isinstance(source, Mapping) and str(source.get("path") or "").strip()
    }


def _canonicalize_semantic_source_bindings(
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

    intent = arguments.get("intent")
    output_spec = arguments.get("output_spec")
    if not isinstance(intent, Mapping) or not isinstance(output_spec, Mapping):
        return request
    requirements = intent.get("requirements") or []
    output_columns = output_spec.get("columns") or []
    if not isinstance(output_columns, list):
        return request
    original_output_source_fields = {
        _normalized_quote_text(str(field or ""))
        for column in output_columns
        if isinstance(column, Mapping)
        for field in column.get("source_fields") or []
        if str(field or "").strip()
    }
    if not output_columns:
        return request

    requested_sources = {
        str(source.get("path") or "").replace("\\", "/")
        for source in evidence.get("context_sources") or []
        if isinstance(source, Mapping) and str(source.get("path") or "").strip()
    }
    execution_spec = arguments.get("execution_spec")
    execution_sources = _source_paths_from_execution_spec(execution_spec)
    plan_sources = requested_sources | execution_sources
    if not plan_sources:
        return request

    narrative_sources = _observed_narrative_sources_by_logical(request.state)
    observed_sources_by_logical = _observed_sources_by_logical(request.state)
    if not narrative_sources and not observed_sources_by_logical:
        return request

    def narrative_paths_for_table(logical_table: str) -> list[str]:
        logical_name = _normalized_quote_text(logical_table)
        paths = narrative_sources.get(logical_name, [])
        if paths:
            return paths
        return [
            path
            for path in observed_sources_by_logical.get(logical_name, [])
            if path.lower().endswith((".md", ".markdown", ".txt", ".pdf"))
        ]

    original_request = str(request.state.get("original_request") or "")
    knowledge_facts = parse_knowledge_content(discovery.knowledge_content)
    knowledge_facts_by_id = {str(fact.fact_id): fact for fact in knowledge_facts}
    output_field_aliases = {
        _normalized_quote_text(str(column.get("name") or ""))
        for column in output_columns
        if isinstance(column, Mapping)
        and str(column.get("name") or "").strip()
    }
    output_field_aliases.update(original_output_source_fields)
    execution_spec_mapping = execution_spec if isinstance(execution_spec, Mapping) else {}
    binding_field_aliases = {
        _normalized_quote_text(str(binding.get("source_field") or ""))
        for binding in execution_spec_mapping.get("source_bindings") or []
        if isinstance(binding, Mapping)
        and str(binding.get("source_field") or "").strip()
    }
    cited_field_facts: list[Any] = []
    for rule in evidence.get("knowledge_rules") or []:
        if not isinstance(rule, Mapping):
            continue
        fact = knowledge_facts_by_id.get(str(rule.get("fact_id") or ""))
        if (
            fact is not None
            and fact.kind == "field"
            and fact.logical_table
            and fact.logical_field
        ):
            cited_field_facts.append(fact)
    cited_target_facts = [
        fact
        for fact in cited_field_facts
        if len(cited_field_facts) == 1
        or _normalized_quote_text(fact.logical_field or "") in output_field_aliases
        or _normalized_quote_text(fact.logical_field or "") in binding_field_aliases
    ]

    target_facts: list[Any] = []
    seen_fact_ids: set[str] = set()
    for fact in [*cited_target_facts, *knowledge_facts]:
        if not (
            fact.kind == "field"
            and fact.logical_table
            and fact.logical_field
            and (
                str(fact.fact_id) in {
                    str(item.fact_id) for item in cited_target_facts
                }
                or _fact_targets_request(
                fact=fact,
                original_request=original_request,
                requirements=requirements,
                )
            )
        ):
            continue
        observed_paths = narrative_paths_for_table(fact.logical_table)
        if not observed_paths:
            continue
        if str(fact.fact_id) in seen_fact_ids:
            continue
        target_facts.append(fact)
        seen_fact_ids.add(str(fact.fact_id))

    bindings: list[dict[str, Any]] = []
    for fact in target_facts:
        observed_paths = narrative_paths_for_table(fact.logical_table)
        selected_paths = [
            path
            for path in observed_paths
            if path in plan_sources
        ] or list(observed_paths)
        if not selected_paths:
            continue
        bindings.append(
            {
                "fact_id": fact.fact_id,
                "logical_table": fact.logical_table,
                "source_field": fact.logical_field,
                "source_paths": selected_paths,
            }
        )

    if not bindings:
        return request

    updated_evidence = dict(evidence)
    updated_context_sources = [
        dict(source)
        for source in evidence.get("context_sources") or []
        if isinstance(source, Mapping)
    ]
    existing_context_paths = {
        str(source.get("path") or "").replace("\\", "/")
        for source in updated_context_sources
        if str(source.get("path") or "").strip()
    }
    for binding in bindings:
        logical_table = str(binding.get("logical_table") or "")
        source_field = str(binding.get("source_field") or "")
        for path in binding.get("source_paths") or []:
            source_path = str(path or "").replace("\\", "/")
            if not source_path or source_path in existing_context_paths:
                continue
            updated_context_sources.append(
                {
                    "path": source_path,
                    "observations": [
                        (
                            "Observed narrative source for "
                            f"{logical_table}.{source_field}."
                        )
                    ],
                }
            )
            existing_context_paths.add(source_path)
    updated_evidence["context_sources"] = updated_context_sources

    updated_knowledge_rules = [
        dict(rule)
        for rule in evidence.get("knowledge_rules") or []
        if isinstance(rule, Mapping)
    ]
    existing_rule_fact_ids = {
        str(rule.get("fact_id") or "")
        for rule in updated_knowledge_rules
        if str(rule.get("fact_id") or "").strip()
    }
    for fact in target_facts:
        if str(fact.fact_id) in existing_rule_fact_ids:
            continue
        updated_knowledge_rules.append(
            {
                "rule_type": "semantic",
                "source_path": fact.source_path,
                "quote": fact.quote,
                "fact_id": fact.fact_id,
            }
        )
        existing_rule_fact_ids.add(str(fact.fact_id))
    updated_evidence["knowledge_rules"] = updated_knowledge_rules

    updated_output_spec = dict(output_spec)
    canonical_fields = [
        str(binding.get("source_field") or "")
        for binding in bindings
        if str(binding.get("source_field") or "").strip()
    ]
    updated_columns: list[Any] = []
    used_column_indexes: set[int] = set()
    for field in canonical_fields:
        normalized_field = _normalized_quote_text(field)
        selected_index: int | None = None
        for index, column in enumerate(output_columns):
            if index in used_column_indexes or not isinstance(column, Mapping):
                continue
            source_fields = {
                _normalized_quote_text(str(item or ""))
                for item in column.get("source_fields") or []
                if str(item or "").strip()
            }
            if normalized_field in source_fields:
                selected_index = index
                break
        if selected_index is None:
            for index, column in enumerate(output_columns):
                if index in used_column_indexes or not isinstance(column, Mapping):
                    continue
                if str(column.get("role") or "") in {
                    "measure",
                    "calculation",
                    "output_column",
                }:
                    selected_index = index
                    break
        if selected_index is None:
            for index, column in enumerate(output_columns):
                if index not in used_column_indexes and isinstance(column, Mapping):
                    selected_index = index
                    break
        if selected_index is None:
            updated_columns.append({"name": field, "source_fields": [field]})
            continue
        used_column_indexes.add(selected_index)
        selected_column = dict(output_columns[selected_index])
        selected_column["source_fields"] = [field]
        if not str(selected_column.get("name") or "").strip():
            selected_column["name"] = field
        updated_columns.append(selected_column)
    updated_output_spec["columns"] = updated_columns
    if "expected_row_count" in updated_output_spec:
        updated_output_spec["expected_row_count"] = None

    updated_execution_spec = (
        dict(execution_spec) if isinstance(execution_spec, Mapping) else {}
    )
    updated_execution_spec.setdefault("sources", [])
    updated_execution_spec.setdefault("supporting_fields", [])
    updated_execution_spec.setdefault("operations", [])
    existing_source_paths = {
        str(source.get("path") or "").replace("\\", "/")
        for source in updated_execution_spec.get("sources") or []
        if isinstance(source, Mapping) and str(source.get("path") or "").strip()
    }
    updated_sources = [
        dict(source)
        for source in updated_execution_spec.get("sources") or []
        if isinstance(source, Mapping)
    ]
    for binding in bindings:
        for path in binding.get("source_paths") or []:
            source_path = str(path or "").replace("\\", "/")
            if not source_path or source_path in existing_source_paths:
                continue
            updated_sources.append(
                {
                    "path": source_path,
                    "source_type": "doc",
                    "table_or_path": source_path,
                }
            )
            existing_source_paths.add(source_path)
    updated_execution_spec["sources"] = updated_sources
    derived_fields = {
        _normalized_quote_text(str(binding.get("source_field") or ""))
        for binding in bindings
        if str(binding.get("source_field") or "").strip()
    }
    replaced_fields = set(original_output_source_fields) | derived_fields
    existing_bindings = [
        dict(item)
        for item in updated_execution_spec.get("source_bindings") or []
        if isinstance(item, Mapping)
        and _normalized_quote_text(str(item.get("source_field") or ""))
        not in replaced_fields
    ]
    existing_keys = {
        (
            str(item.get("fact_id") or ""),
            str(item.get("logical_table") or ""),
            str(item.get("source_field") or ""),
        )
        for item in existing_bindings
    }
    for binding in bindings:
        key = (
            str(binding.get("fact_id") or ""),
            str(binding.get("logical_table") or ""),
            str(binding.get("source_field") or ""),
        )
        if key not in existing_keys:
            existing_bindings.append(binding)
            existing_keys.add(key)
    updated_execution_spec["source_bindings"] = existing_bindings

    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "evidence": updated_evidence,
                "output_spec": updated_output_spec,
                "execution_spec": updated_execution_spec,
            },
        }
    )


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


def _semantic_overlap_tokens(value: str) -> set[str]:
    normalized = _normalized_quote_text(value)
    tokens = {
        token
        for token in re.findall(r"[0-9a-z_]{3,}", normalized)
        if token not in {"the", "and", "from", "with"}
    }
    for sequence in _CJK_SEQUENCE_PATTERN.findall(normalized):
        if len(sequence) < 2:
            continue
        tokens.update(
            sequence[index : index + 2]
            for index in range(0, len(sequence) - 1)
        )
    return _singularized_tokens(tokens)


def _split_identifier_tokens(value: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    spaced = re.sub(r"[_\W]+", " ", spaced, flags=re.UNICODE)
    tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", spaced)
    }
    collapsed = "".join(tokens)
    tokens.update(
        token
        for token in _semantic_overlap_tokens(collapsed)
        if token.isascii()
    )
    return _singularized_tokens(tokens)


def _singularized_tokens(tokens: set[str]) -> set[str]:
    normalized = set(tokens)
    for token in list(tokens):
        if len(token) > 3 and token.endswith("ies"):
            normalized.add(f"{token[:-3]}y")
        elif len(token) > 3 and token.endswith("s"):
            normalized.add(token[:-1])
    return normalized


def _canonical_semantic_concepts(tokens: set[str]) -> set[str]:
    concepts: set[str] = set()
    for token in tokens:
        normalized = token.casefold()
        if len(normalized) > 3 and normalized.endswith("ies"):
            normalized = f"{normalized[:-3]}y"
        elif len(normalized) > 3 and normalized.endswith("s"):
            normalized = normalized[:-1]
        if normalized and normalized not in _GENERIC_SEMANTIC_TOKENS:
            concepts.add(normalized)
    return concepts


def _fact_matches_target_requirement(
    *,
    fact: Any,
    target_requirement_blob: str,
) -> bool:
    target_tokens = _canonical_semantic_concepts(
        _semantic_overlap_tokens(target_requirement_blob)
        | _split_identifier_tokens(target_requirement_blob)
    )
    field_tokens = _canonical_semantic_concepts(
        _split_identifier_tokens(str(fact.logical_field or ""))
    )
    quote_tokens = _canonical_semantic_concepts(
        _semantic_overlap_tokens(str(fact.quote or ""))
    )
    field_overlap = field_tokens & target_tokens
    quote_overlap = quote_tokens & target_tokens
    normalized_target = _normalized_quote_text(target_requirement_blob).replace(" ", "")
    normalized_field = _normalized_quote_text(str(fact.logical_field or "")).replace(" ", "")
    field_substring_overlap = {
        token
        for token in target_tokens
        if len(token) >= 4 and token.isascii() and token in normalized_field
    }
    if normalized_field and normalized_field in normalized_target:
        return True
    if len(field_overlap) >= 2:
        return True
    if len(field_substring_overlap) >= 2:
        return True
    if field_substring_overlap and len(quote_overlap) >= 2:
        return True
    return bool(field_overlap) and len(quote_overlap) >= 2


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
    conditions = question_structure.get("conditions")
    if isinstance(conditions, Mapping):
        for container_name, items in conditions.items():
            if not isinstance(items, list):
                continue
            requirement_type = _condition_requirement_type(str(container_name))
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                quote = str(item.get("quote") or "").strip()
                if quote:
                    pairs.append((quote, requirement_type))
    return pairs


def _requirement_types_for_quote(
    *,
    quote: str,
    arguments: Mapping[str, Any],
    state: Mapping[str, Any],
) -> set[str]:
    stripped = quote.strip()
    if not stripped:
        return set()
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
    pairs.extend(_quote_type_pairs_from_question_structure(state.get("question_structure")))
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


def _is_generic_scope_quote(quote: str) -> bool:
    normalized = _normalized_quote_text(quote)
    if not normalized:
        return True
    compact = normalized.replace(" ", "")
    if compact in _GENERIC_SCOPE_TERMS:
        return True
    tokens = set(normalized.split())
    return bool(tokens) and tokens.issubset(_GENERIC_SCOPE_TERMS)


def _quote_has_specific_filter_value(quote: str) -> bool:
    stripped = quote.strip()
    if not stripped or _is_generic_scope_quote(stripped):
        return False
    if _SPECIFIC_FILTER_PATTERN.search(stripped):
        return True
    normalized = _normalized_quote_text(stripped)
    if any(term in stripped for term in ("型", "类", "为", "等于", "代码", "学历")):
        return True
    return 0 < len(normalized.replace(" ", "")) <= 4


def _user_quote_authorizes_operation(
    *,
    operation: str,
    quote: str,
    arguments: Mapping[str, Any],
    state: Mapping[str, Any],
) -> bool:
    operation_name = operation.casefold()
    requirement_types = _requirement_types_for_quote(
        quote=quote,
        arguments=arguments,
        state=state,
    )
    if requirement_types & _OPERATION_REQUIREMENT_TYPES.get(operation_name, frozenset()):
        return True
    if operation_name == "filter":
        return "entity" in requirement_types and _quote_has_specific_filter_value(quote)
    if operation_name in {"aggregate", "sort", "limit"}:
        return bool(_AGGREGATE_SELECTOR_PATTERN.search(quote))
    if operation_name == "derive":
        return bool(_DERIVE_PATTERN.search(quote))
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


def _fact_targets_request(
    *,
    fact: Any,
    original_request: str,
    requirements: list[Any],
) -> bool:
    target_requirement_blob = " ".join(
        " ".join(str(item.get(key) or "") for key in ("quote", "statement"))
        for item in requirements
        if isinstance(item, Mapping)
        and str(item.get("requirement_type") or "")
        in {"measure", "calculation", "output_column"}
    )
    if target_requirement_blob.strip():
        field_blob = " ".join(
            str(item or "")
            for item in (fact.logical_field, fact.quote)
        )
        request_time_terms = _time_overlap_terms(target_requirement_blob)
        fact_time_terms = _time_overlap_terms(field_blob)
        if request_time_terms and not fact_time_terms:
            return False
        if request_time_terms and fact_time_terms and not (
            request_time_terms & fact_time_terms
        ):
            return False
        return _fact_matches_target_requirement(
            fact=fact,
            target_requirement_blob=target_requirement_blob,
        )

    request_blob = " ".join(
        [
            original_request,
            *(
                " ".join(
                    str(item.get(key) or "")
                    for key in ("quote", "statement")
                )
                for item in requirements
                if isinstance(item, Mapping)
            ),
        ]
    )
    fact_blob = " ".join(
        str(item or "")
        for item in (fact.logical_field, fact.logical_table, fact.quote)
    )
    request_time_terms = _time_overlap_terms(request_blob)
    fact_time_terms = _time_overlap_terms(fact_blob)
    if request_time_terms and not fact_time_terms:
        return False
    if request_time_terms and fact_time_terms and not (
        request_time_terms & fact_time_terms
    ):
        return False
    if not _fact_matches_target_requirement(
        fact=fact,
        target_requirement_blob=request_blob,
    ):
        return False
    request_concepts = _canonical_semantic_concepts(
        _semantic_overlap_tokens(request_blob) | _split_identifier_tokens(request_blob)
    )
    table_concepts = _canonical_semantic_concepts(
        _semantic_overlap_tokens(str(fact.logical_table or ""))
        | _split_identifier_tokens(str(fact.logical_table or ""))
    )
    return True


def _fact_has_time_terms(fact: Any) -> bool:
    fact_blob = " ".join(
        str(item or "")
        for item in (fact.logical_field, fact.logical_table, fact.quote)
    )
    return bool(_time_overlap_terms(fact_blob))


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
    if quote and quote in knowledge_content:
        return quote
    normalized_quote = _normalized_quote_text(quote)
    if not normalized_quote:
        return None
    matches = [
        line.strip()
        for line in knowledge_content.splitlines()
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
            if fact.kind == "field" and normalized_rule.get("rule_type") != "semantic":
                normalized_rule["rule_type"] = "semantic"
                changed = True
        elif str(normalized_rule.get("rule_type") or "") == "field":
            normalized_rule["rule_type"] = "semantic"
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
    normalized_output_spec = arguments.get("output_spec")
    if isinstance(normalized_output_spec, Mapping):
        updated_output_spec = dict(normalized_output_spec)
        normalized_transformations = []
        for transformation in updated_output_spec.get("transformations") or []:
            if not isinstance(transformation, Mapping):
                normalized_transformations.append(transformation)
                continue
            normalized_transformation = dict(transformation)
            authorization = transformation.get("authorization")
            if isinstance(authorization, Mapping) and str(authorization.get("source") or "") == "knowledge":
                quote = str(authorization.get("quote") or "").strip()
                canonical_quote = _canonical_knowledge_quote(
                    quote,
                    discovery.knowledge_content,
                )
                if canonical_quote is None:
                    canonical_quote = quote_for_fact_ids(
                        transformation.get("authorization_fact_ids")
                    )
                if canonical_quote is not None and canonical_quote != quote:
                    normalized_transformation["authorization"] = {
                        **dict(authorization),
                        "quote": canonical_quote,
                    }
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
            normalized_operation = dict(operation)
            authorization = operation.get("authorization")
            if isinstance(authorization, Mapping) and str(authorization.get("source") or "") == "knowledge":
                quote = str(authorization.get("quote") or "").strip()
                canonical_quote = _canonical_knowledge_quote(
                    quote,
                    discovery.knowledge_content,
                )
                if canonical_quote is None:
                    canonical_quote = quote_for_fact_ids(
                        operation.get("authorization_fact_ids")
                    )
                if canonical_quote is not None and canonical_quote != quote:
                    normalized_operation["authorization"] = {
                        **dict(authorization),
                        "quote": canonical_quote,
                    }
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
                if knowledge_available and injected_knowledge is None:
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
                if observed_content and injected_knowledge is None:
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

    return _DiscoveryState(
        knowledge_present=knowledge_present,
        knowledge_checked=knowledge_checked,
        knowledge_available=knowledge_available,
        knowledge_content="\n".join(knowledge_contents),
        context_sources=frozenset(context_sources),
        needs_cross_validation=needs_cross_validation,
    )


def _plan_error(
    request: ToolCallRequest,
    content: str,
) -> ToolMessage:
    return _tool_error(request, f"Invalid analysis plan: {content}")


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
    evidence = arguments.get("evidence") or {}
    if not isinstance(evidence, Mapping):
        return _plan_error(request, "evidence must be a JSON object.")
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
    semantic_field_facts = [
        fact
        for fact in knowledge_facts
        if fact.kind == "field" and fact.logical_field
    ]
    semantic_field_facts_by_quote: dict[str, list[Any]] = {}
    for fact in semantic_field_facts:
        semantic_field_facts_by_quote.setdefault(fact.quote, []).append(fact)
    narrative_sources = _observed_narrative_sources_by_logical(request.state)
    observed_sources_by_logical = _observed_sources_by_logical(request.state)
    if knowledge_status in {"unavailable", "invalid"}:
        observed_fact_sources = sorted(
            {
                path
                for fact in knowledge_facts
                for path in narrative_sources.get(
                    _normalized_quote_text(fact.logical_table or ""),
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
    rule_quotes_by_type: dict[str, set[str]] = {}
    cited_semantic_field_facts: list[Any] = []
    for rule in knowledge_rules:
        if not isinstance(rule, dict):
            continue
        source_path = str(rule.get("source_path") or "").replace("\\", "/")
        quote = str(rule.get("quote") or "").strip()
        rule_type = str(rule.get("rule_type") or "")
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
                if (
                    rule_type == "semantic"
                    and fact.kind == "field"
                    and fact.logical_field
                ):
                    cited_semantic_field_facts.append(fact)
        elif rule_type == "semantic":
            cited_semantic_field_facts.extend(
                semantic_field_facts_by_quote.get(quote)
                or _field_facts_for_knowledge_quote(
                    quote=quote,
                    field_facts=semantic_field_facts,
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
        rule_quotes_by_type.setdefault(rule_type, set()).add(quote)

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
        for types in requirement_types_by_quote.values()
        for requirement_type in types
    }
    missing_operation_requirements: list[str] = []
    if required_types & {"filter", "time_range"} and "filter" not in declared_operations:
        missing_operation_requirements.append("filter")
    if "grouping" in required_types and "aggregate" not in declared_operations:
        missing_operation_requirements.append("aggregate")
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
    execution_sources = _source_paths_from_execution_spec(execution_spec)
    if knowledge_status == "authoritative":
        output_source_fields = {
            _normalized_quote_text(str(field or ""))
            for column in output_columns
            if isinstance(column, Mapping)
            for field in column.get("source_fields") or []
            if str(field or "").strip()
        }
        binding_source_fields = {
            _normalized_quote_text(str(binding.get("source_field") or ""))
            for binding in execution_spec.get("source_bindings") or []
            if isinstance(binding, Mapping)
            and str(binding.get("source_field") or "").strip()
        }
        cited_fact_ids = {str(fact.fact_id) for fact in cited_semantic_field_facts}
        explicit_cited_target_facts = [
            fact
            for fact in cited_semantic_field_facts
            if len(cited_semantic_field_facts) == 1
            or _normalized_quote_text(fact.logical_field or "") in output_source_fields
            or _normalized_quote_text(fact.logical_field or "") in binding_source_fields
        ]
        automatic_target_field_facts = [
            fact
            for fact in semantic_field_facts
            if _fact_targets_request(
                fact=fact,
                original_request=original_request,
                requirements=requirements,
            )
        ]
        request_target_field_facts: list[Any] = []
        seen_request_target_fact_ids: set[str] = set()
        for fact in [*explicit_cited_target_facts, *automatic_target_field_facts]:
            if str(fact.fact_id) in seen_request_target_fact_ids:
                continue
            request_target_field_facts.append(fact)
            seen_request_target_fact_ids.add(str(fact.fact_id))
        undiscovered_target_fields = sorted(
            {
                f"{fact.logical_table}.{fact.logical_field}"
                for fact in request_target_field_facts
                if fact.logical_table
                and _fact_has_time_terms(fact)
                if not observed_sources_by_logical.get(
                    _normalized_quote_text(fact.logical_table or ""),
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
                    "inspect the logical source first: "
                    f"{undiscovered_target_fields}."
                ),
            )
        observed_target_field_facts = [
            fact
            for fact in request_target_field_facts
            if observed_sources_by_logical.get(
                _normalized_quote_text(fact.logical_table or ""),
                [],
            )
        ]
        uncited_target_fields = sorted(
            {
                f"{fact.logical_table}.{fact.logical_field}"
                for fact in observed_target_field_facts
                if str(fact.fact_id) not in cited_fact_ids
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
        missing_logical_fields = sorted(
            {
                str(fact.logical_field)
                for fact in target_field_facts
                if _normalized_quote_text(fact.logical_field or "")
                not in output_source_fields
            }
        )
        if missing_logical_fields:
            return _plan_error(
                request,
                (
                    "output columns authorized by knowledge field facts must "
                    "cite those logical fields in source_fields; do not replace "
                    f"them with semantic-neighbor fields: {missing_logical_fields}."
                ),
            )
        plan_sources = requested_sources | execution_sources
        missing_source_bindings = sorted(
            {
                f"{fact.logical_table}.{fact.logical_field}"
                for fact in target_field_facts
                if narrative_sources.get(
                    _normalized_quote_text(fact.logical_table or ""),
                    [],
                )
                if not (
                    set(
                        narrative_sources.get(
                            _normalized_quote_text(fact.logical_table or ""),
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
        del operation
        if not quote or quote not in original_request:
            return (
                "user authorization must cite an exact quote from "
                "original_request."
            )
        return None

    def knowledge_authorization_error(operation: str, quote: str) -> str | None:
        del operation
        authorized_quotes = {
            quote
            for quotes in rule_quotes_by_type.values()
            for quote in quotes
        }
        if not quote or quote not in authorized_quotes:
            return (
                "knowledge authorization must cite an observed knowledge quote."
            )
        return None

    def fact_authorizes_operation(operation: str, fact_id: str) -> bool:
        del operation
        return fact_id in knowledge_facts_by_id

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
        return handler(
            request.override(system_message=SystemMessage(content=self.prompt))
        )


class PlanningMiddleware(AgentMiddleware[BenchmarkDeepAgentState, None, Any]):
    """Enforce Discovery -> Plan -> Todos and allow revisions."""

    state_schema = BenchmarkDeepAgentState
    tools = [analyze_plan_tool]

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        discovery = _discovery_state(request.messages, request.state)
        if request.state.get("analysis_plan") is None:
            allowed_tools, tool_choice = discovery.tool_policy()
            request = request.override(
                tools=[
                    item
                    for item in request.tools
                    if tool_name(item) in allowed_tools
                ],
                tool_choice=tool_choice,
            )
        elif request.state.get("answer_candidate") is not None:
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
            request = request.override(tools=todo_tools, tool_choice="write_todos")
        response = handler(request)
        return _retry_invalid_tool_call(request, handler, response)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        current_tool_name = str(request.tool_call.get("name") or "")
        if current_tool_name == "analyze_plan":
            request = _normalize_tool_call_arguments(request)
        plan = request.state.get("analysis_plan")
        discovery = (
            _discovery_state(request.state["messages"], request.state)
            if plan is None or current_tool_name == "analyze_plan"
            else None
        )

        if plan is None and discovery is not None:
            if current_tool_name not in _SOURCE_DISCOVERY_TOOLS | {"analyze_plan"}:
                return _tool_error(
                    request,
                    "Only discovery tools are available before analyze_plan.",
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
            request = _canonicalize_unsupported_operations(request)
            request = _canonicalize_unrequested_key_output_columns(request)
            request = _canonicalize_preserve_expected_row_count(request)
            request = _canonicalize_transform_output_policy(request)
            request = _canonicalize_preserve_output_policy(request)
            request = _canonicalize_plan_steps(request)
            request = _canonicalize_execution_supporting_fields(request)
            request = _canonicalize_revision(request)
            request = _canonicalize_semantic_source_bindings(request, discovery)
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
