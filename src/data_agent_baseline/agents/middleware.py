from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
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

_SOURCE_DISCOVERY_TOOLS = frozenset(
    {
        "execute_python",
        "grep_file",
        "inspect_sqlite",
        "query_schema",
        "read_csv",
        "read_doc",
        "read_json",
    }
)
_CONTEXT_PATH_PATTERN = re.compile(r"""["'](/context/[^"']+)["']""")
_INJECTED_KNOWLEDGE_PATTERN = re.compile(
    r"<context_knowledge>\s*(.*?)\s*</context_knowledge>",
    re.DOTALL,
)
_USER_TRANSFORMATION_REQUIREMENT_TYPES = {
    "filter": frozenset({"filter"}),
    "aggregate": frozenset({"calculation"}),
    "derive": frozenset({"calculation"}),
    "sort": frozenset({"ordering"}),
    "limit": frozenset({"limit"}),
    "deduplicate": frozenset({"deduplication"}),
    "reshape": frozenset({"reshape"}),
}
_KNOWLEDGE_TRANSFORMATION_RULE_TYPES = {
    "filter": frozenset({"filter"}),
    "aggregate": frozenset({"calculation"}),
    "derive": frozenset({"calculation"}),
    "sort": frozenset({"output"}),
    "limit": frozenset({"output"}),
    "deduplicate": frozenset({"output"}),
    "reshape": frozenset({"output"}),
}
_KNOWLEDGE_FACT_KINDS_BY_OPERATION = {
    "filter": frozenset({"filter_rule", "example_query"}),
    "aggregate": frozenset({"calculation", "example_query", "output_rule"}),
    "derive": frozenset({"calculation", "example_query"}),
    "sort": frozenset({"ordering_rule", "example_query", "output_rule"}),
    "limit": frozenset({"ordering_rule", "example_query", "output_rule"}),
    "deduplicate": frozenset({"output_rule", "example_query"}),
    "reshape": frozenset({"output_rule", "example_query"}),
}
_OPERATION_CONDITION_KEYS = {
    "filter": "filters",
    "aggregate": "calculations",
    "derive": "calculations",
    "sort": "orderings",
    "limit": "limits",
    "deduplicate": "deduplications",
    "reshape": "reshapes",
}
_ACTION_CLAIM_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "filter": (
        re.compile(
            r"\b(?:can|could|would|should|will|need\s+to|i\s+will|i\s+should)\s+"
            r"select\s+(?:representative|matching|only|national|source\s+)?"
            r"(?:rows?|records?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bselect\s+(?:representative|matching|only|national|source\s+)"
            r"(?:rows?|records?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:filter|select|keep|retain|return)\s+(?:only\s+)?"
            r"(?:the\s+)?(?:rows?|records?|source\s+rows?)?\s*"
            r"(?:where|with|matching|for)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\breturn\s+only\s+(?:the\s+)?(?:rows?|records?)\b", re.IGNORECASE),
        re.compile(r"(?:筛选|过滤|只保留|仅保留|只返回|仅返回).{0,40}(?:记录|行|数据)"),
    ),
    "aggregate": (
        re.compile(
            r"\b(?:can|could|would|should|will|need\s+to|i\s+will|i\s+should)\s+"
            r"(?:aggregate|sum|total|roll\s*up)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:calculate|compute|derive|produce|create|return|report)\s+"
            r"(?:the\s+)?(?:sum|total|aggregate)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:calculate|compute)\s+(?:the\s+)?sum\s+"
            r"(?:across|of|from|over)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:sum|aggregate|total|roll\s*up)\s+(?:all\s+)?"
            r"(?:source\s+)?(?:rows|records|values|provinces|provincial)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bgroup\s+by\b", re.IGNORECASE),
        re.compile(r"(?:计算|求取|求和|汇总|加总|合计|聚合).{0,40}(?:记录|行|值|省|地区|全国|总量)"),
    ),
    "derive": (
        re.compile(
            r"\b(?:calculate|compute|derive|create|produce)\s+"
            r"(?:(?:a|an|the)\s+)?(?:ratio|rate|share|percentage|growth|"
            r"difference|delta|change|index|indicator|derived)\b",
            re.IGNORECASE,
        ),
        re.compile(r"(?:派生|计算|生成).{0,40}(?:比例|占比|率|增速|增长|差值|变化|指标|新列)"),
    ),
    "sort": (
        re.compile(
            r"\b(?:sort|order|rank)\s+(?:the\s+)?"
            r"(?:rows|records|values|data|result|results)?\s*"
            r"(?:by|ascending|descending|from)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\border\s+by\b", re.IGNORECASE),
        re.compile(r"(?:排序|按.{1,40}排序|降序|升序|排名)"),
    ),
    "limit": (
        re.compile(
            r"\b(?:limit|keep|return|select|take)\s+(?:the\s+)?"
            r"(?:top|bottom|first|last|latest|earliest)\s+\d+\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:top|bottom)\s+\d+\b", re.IGNORECASE),
        re.compile(r"(?:前|后|最新|最早)\s*\d+\s*(?:条|个|行|年)"),
    ),
    "deduplicate": (
        re.compile(r"\b(?:deduplicate|de-duplicate|remove\s+duplicates|distinct)\b", re.IGNORECASE),
        re.compile(r"(?:去重|删除重复|唯一值)"),
    ),
    "reshape": (
        re.compile(
            r"\b(?:pivot|unpivot|transpose|reshape|wide\s+format|long\s+format)\b",
            re.IGNORECASE,
        ),
        re.compile(r"(?:透视|转置|宽表|长表|重塑)"),
    ),
}
_OUTPUT_COLUMN_REQUIREMENT_TYPES = frozenset(
    {
        "calculation",
        "entity",
        "grouping",
        "measure",
        "output_column",
    }
)
_CONTEXT_COLUMN_ROLES = frozenset({"entity_key", "record_key", "time_key"})
_REVISION_FIELDS = (
    "intent",
    "output_spec",
    "evidence",
    "steps",
    "delegation_candidates",
)
_TODO_STATUSES = frozenset({"pending", "in_progress", "completed"})
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
        """根据已掌握的信息决定下一轮可见工具及是否强制调用。"""

        if not self.context_ready:
            return set(_SOURCE_DISCOVERY_TOOLS), None
        return {"analyze_plan"}, "analyze_plan"


def tool_name(value: Any) -> str:
    """兼容 LangChain 工具对象和字典形式的工具定义。"""

    if isinstance(value, dict):
        return str(value.get("name") or value.get("function", {}).get("name") or "")
    return str(getattr(value, "name", ""))


def _invalid_tool_name(response: ModelResponse[Any]) -> str | None:
    """识别只有格式错误工具调用、没有可执行调用的模型响应。"""

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
    """让模型纠正一次工具参数格式，避免图把解析失败误判为正常结束。"""

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
    if normalized_arguments is arguments or not isinstance(normalized_arguments, Mapping):
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

    observed_counts = _observed_source_row_counts(request.state["messages"])
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


def _canonical_requirements_from_question_structure(
    state: Mapping[str, Any],
) -> list[dict[str, str]] | None:
    if not state.get("question_structure_enforced"):
        return None
    structure = state.get("question_structure")
    if not isinstance(structure, Mapping):
        return None

    question_text = str(
        structure.get("original_question") or state.get("original_request") or ""
    )
    requirements: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def append_requirement(
        *,
        quote: Any,
        requirement_type: str | None,
        statement: Any,
    ) -> None:
        quote_text = str(quote or "").strip()
        statement_text = str(statement or "").strip()
        if not quote_text or not requirement_type:
            return
        if question_text and quote_text not in question_text:
            return
        key = (quote_text, requirement_type)
        if key in seen:
            return
        seen.add(key)
        requirements.append(
            {
                "statement": statement_text or f"{requirement_type}: {quote_text}",
                "requirement_type": requirement_type,
                "quote": quote_text,
            }
        )

    target_type_map = {
        "entity": "entity",
        "measure": "measure",
        "record_set": "output",
    }
    for target in structure.get("targets") or []:
        if not isinstance(target, Mapping):
            continue
        requirement_type = target_type_map.get(str(target.get("target_type") or ""))
        append_requirement(
            quote=target.get("quote"),
            requirement_type=requirement_type,
            statement=target.get("description") or target.get("name"),
        )

    constraint_type_map = {
        "entity": "entity",
        "filter": "filter",
        "geography": "entity",
        "grouping": "grouping",
        "limit": "limit",
        "ordering": "ordering",
        "output_shape": "output",
        "scope": "entity",
        "time_range": "time_range",
    }
    for constraint in structure.get("target_constraints") or []:
        if not isinstance(constraint, Mapping):
            continue
        constraint_type = str(constraint.get("constraint_type") or "")
        requirement_type = constraint_type_map.get(constraint_type)
        statement_parts = [
            constraint_type,
            str(constraint.get("value") or "").strip(),
        ]
        append_requirement(
            quote=constraint.get("quote"),
            requirement_type=requirement_type,
            statement=": ".join(part for part in statement_parts if part),
        )

    condition_type_map = {
        "calculations": "calculation",
        "filters": "filter",
        "groupings": "grouping",
        "limits": "limit",
        "orderings": "ordering",
        "output_columns": "output_column",
        "time_ranges": "time_range",
    }
    conditions = structure.get("conditions") or {}
    if isinstance(conditions, Mapping):
        for key, requirement_type in condition_type_map.items():
            for item in conditions.get(key) or []:
                if isinstance(item, Mapping):
                    quote = item.get("quote")
                    statement = (
                        item.get("description")
                        or item.get("statement")
                        or item.get("value")
                        or key
                    )
                else:
                    quote = item
                    statement = key
                append_requirement(
                    quote=quote,
                    requirement_type=requirement_type,
                    statement=statement,
                )

    operator_type_map = {
        "aggregate": "calculation",
        "derive": "calculation",
        "sort": "ordering",
        "limit": "limit",
    }
    for operator in structure.get("intent_operators") or []:
        if not isinstance(operator, Mapping):
            continue
        operation = str(operator.get("operation") or "")
        requirement_type = operator_type_map.get(operation)
        append_requirement(
            quote=operator.get("quote"),
            requirement_type=requirement_type,
            statement=f"{operation}: {operator.get('operator_type') or operator.get('quote')}",
        )
        if operation == "aggregate" and str(operator.get("operator_type") or "") == "distribution":
            append_requirement(
                quote=operator.get("quote"),
                requirement_type="grouping",
                statement=f"{operation}: distribution group key",
            )

    return requirements or None


def _canonicalize_intent_requirements(request: ToolCallRequest) -> ToolCallRequest:
    canonical_requirements = _canonical_requirements_from_question_structure(
        request.state
    )
    if not canonical_requirements:
        return request
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, Mapping):
        return request
    intent = arguments.get("intent")
    if not isinstance(intent, Mapping):
        return request
    if intent.get("requirements") == canonical_requirements:
        return request
    return request.override(
        tool_call={
            **request.tool_call,
            "args": {
                **dict(arguments),
                "intent": {
                    **dict(intent),
                    "requirements": canonical_requirements,
                },
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
    """仅忽略 Markdown 和标点差异，用于定位唯一的原始知识行。"""

    return " ".join(
        re.sub(r"[^\w\u4e00-\u9fff]+", " ", value, flags=re.UNICODE)
        .casefold()
        .split()
    )


def _quote_tokens(value: str) -> set[str]:
    return {
        token
        for token in _normalized_quote_text(value).split()
        if len(token) > 1 or re.search(r"[\u4e00-\u9fff]", token)
    }


def _canonical_knowledge_quote(
    quote: str,
    knowledge_content: str | None,
) -> str | None:
    """把格式被简化的引用还原为 knowledge 中唯一存在的原始行。"""

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

    quote_tokens = _quote_tokens(quote)
    if len(quote_tokens) < 3:
        return None
    fuzzy_matches: list[tuple[float, str]] = []
    for line in knowledge_content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        normalized_line = _normalized_quote_text(stripped)
        line_tokens = _quote_tokens(stripped)
        if not line_tokens:
            continue
        token_recall = len(quote_tokens & line_tokens) / len(quote_tokens)
        sequence_ratio = SequenceMatcher(
            None,
            normalized_quote,
            normalized_line,
        ).ratio()
        score = max(token_recall, sequence_ratio)
        if score >= 0.82:
            fuzzy_matches.append((score, stripped))

    if not fuzzy_matches:
        return None
    best_score = max(score for score, _ in fuzzy_matches)
    best_matches = [line for score, line in fuzzy_matches if score == best_score]
    return best_matches[0] if len(best_matches) == 1 else None



def _canonicalize_plan_quotes(
    request: ToolCallRequest,
    discovery: _DiscoveryState,
) -> ToolCallRequest:
    """将可唯一定位的 knowledge 引用替换成真实原文后再校验和落盘。"""

    raw_arguments = request.tool_call.get("args") or {}
    if not isinstance(raw_arguments, Mapping):
        return request
    arguments = dict(raw_arguments)
    raw_evidence = arguments.get("evidence") or {}
    if not isinstance(raw_evidence, Mapping):
        return request
    evidence = dict(raw_evidence)
    changed = False

    knowledge_facts_by_id = {
        fact.fact_id: fact for fact in parse_knowledge_content(discovery.knowledge_content)
    }
    normalized_rules: list[Any] = []
    for rule in evidence.get("knowledge_rules") or []:
        if not isinstance(rule, dict):
            normalized_rules.append(rule)
            continue
        normalized_rule = dict(rule)
        fact_id = str(rule.get("fact_id") or "").strip()
        if fact_id and not normalized_rule.get("quote"):
            fact = knowledge_facts_by_id.get(fact_id)
            if fact is not None:
                normalized_rule["quote"] = fact.quote
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
        normalized_rules.append(normalized_rule)
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
    """从成功的探索调用中提取实际检查过的上下文来源。"""

    if tool in {
        "grep_file",
        "inspect_sqlite",
        "read_csv",
        "read_doc",
        "read_json",
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


def _discovery_state(messages: list[BaseMessage]) -> _DiscoveryState:
    """汇总计划前已经检查的 knowledge 和独立数据来源。"""

    tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}
    injected_knowledge = _injected_knowledge_content(messages)
    knowledge_present = injected_knowledge is not None
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
                    tool_calls[tool_call_id] = (
                        str(tool_call.get("name") or ""),
                        tool_call.get("args") or {},
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
                str(evidence.get("knowledge_status") or "") != "authoritative"
            )
            continue

        if (
            current_tool_name in _SOURCE_DISCOVERY_TOOLS
            and getattr(message, "status", "success") != "error"
        ):
            context_sources.update(_context_sources(current_tool_name, arguments))

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


def _plan_rejection_payload(
    *,
    code: str,
    json_path: str,
    operation: str,
    offending_text: str,
    field_text: str,
    facts: list[str],
    required_fix: str,
    intent_reconsideration: Mapping[str, Any],
) -> str:
    return json.dumps(
        {
            "code": code,
            "json_path": json_path,
            "operation": operation,
            "offending_text": offending_text,
            "field_text": field_text,
            "facts": facts,
            "intent_reconsideration": dict(intent_reconsideration),
            "required_fix": required_fix,
        },
        ensure_ascii=False,
    )


def _free_text_plan_fields(arguments: Mapping[str, Any]) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    evidence = arguments.get("evidence") or {}
    if isinstance(evidence, Mapping):
        inference = str(evidence.get("cross_validated_inference") or "").strip()
        if inference:
            fields.append(("/evidence/cross_validated_inference", inference))

    intent = arguments.get("intent") or {}
    if isinstance(intent, Mapping):
        for index, item in enumerate(intent.get("unresolved") or []):
            text = str(item or "").strip()
            if text:
                fields.append((f"/intent/unresolved/{index}", text))

    for index, item in enumerate(arguments.get("steps") or []):
        text = str(item or "").strip()
        if text:
            fields.append((f"/steps/{index}", text))

    return fields


def _detect_action_claim(text: str) -> tuple[str, str] | None:
    for operation, patterns in _ACTION_CLAIM_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match is not None:
                return operation, match.group(0)
    return None


def _operation_is_authorized(
    operation: str,
    *,
    requirement_types_by_quote: Mapping[str, set[str]],
    rule_quotes_by_type: Mapping[str, set[str]],
    declared_operations: set[str],
) -> bool:
    if operation in declared_operations:
        return True

    user_types = _USER_TRANSFORMATION_REQUIREMENT_TYPES.get(operation, frozenset())
    if any(
        requirement_type in user_types
        for requirement_types in requirement_types_by_quote.values()
        for requirement_type in requirement_types
    ):
        return True

    rule_types = _KNOWLEDGE_TRANSFORMATION_RULE_TYPES.get(operation, frozenset())
    return any(rule_quotes_by_type.get(rule_type) for rule_type in rule_types)


def _authorization_facts(
    *,
    arguments: Mapping[str, Any],
    operation: str,
    json_path: str,
    offending_text: str,
    original_request: str,
    requirement_types_by_quote: Mapping[str, set[str]],
    rule_quotes_by_type: Mapping[str, set[str]],
    declared_operations: set[str],
    question_conditions: Mapping[str, list[Any]] | None,
) -> list[str]:
    user_types = _USER_TRANSFORMATION_REQUIREMENT_TYPES.get(operation, frozenset())
    rule_types = _KNOWLEDGE_TRANSFORMATION_RULE_TYPES.get(operation, frozenset())
    facts = [
        f"Original request text: {original_request!r}.",
        (
            "Plan intent quotes from the original request are "
            f"{sorted(requirement_types_by_quote)}."
        ),
        (
            f"{json_path} contains an execution claim for {operation!r}: "
            f"{offending_text!r}."
        ),
        (
            "Observed user requirement types are "
            f"{sorted({item for values in requirement_types_by_quote.values() for item in values})}; "
            f"{operation!r} requires one of {sorted(user_types)}."
        ),
        (
            "Observed knowledge rule types are "
            f"{sorted(rule_quotes_by_type)}; {operation!r} requires one of "
            f"{sorted(rule_types)}."
        ),
        (
            "Declared authorized transformations are "
            f"{sorted(declared_operations)}."
        ),
    ]
    evidence = arguments.get("evidence") or {}
    if isinstance(evidence, Mapping):
        for source in evidence.get("context_sources") or []:
            if not isinstance(source, Mapping):
                continue
            path = str(source.get("path") or "").strip()
            observations = [
                str(item).strip()
                for item in source.get("observations") or []
                if str(item).strip()
            ]
            if path and observations:
                facts.append(
                    f"Observed source facts from {path}: "
                    + " | ".join(observations[:4])
                )
    condition_key = _OPERATION_CONDITION_KEYS.get(operation)
    if question_conditions is not None and condition_key is not None:
        facts.append(
            "question_structure.conditions."
            f"{condition_key} has "
            f"{len(question_conditions.get(condition_key, []))} item(s)."
        )
    if question_conditions is not None:
        facts.append(
            "question_structure captured explicit transformation conditions as "
            + json.dumps(
                {
                    key: len(question_conditions.get(key, []))
                    for key in (
                        "filters",
                        "calculations",
                        "orderings",
                        "limits",
                        "groupings",
                        "output_columns",
                    )
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "."
        )
    return facts


def _intent_reconsideration_payload(
    *,
    operation: str,
    json_path: str,
    offending_text: str,
    original_request: str,
    requirement_types_by_quote: Mapping[str, set[str]],
    question_conditions: Mapping[str, list[Any]] | None,
) -> dict[str, Any]:
    condition_key = _OPERATION_CONDITION_KEYS.get(operation)
    condition_count = (
        len(question_conditions.get(condition_key, []))
        if question_conditions is not None and condition_key is not None
        else None
    )
    return {
        "issue": (
            "The plan is using an observed data capability as if it were part of "
            "the user's intent."
        ),
        "trigger": {
            "json_path": json_path,
            "operation": operation,
            "text": offending_text,
        },
        "source_text_to_reinterpret": {
            "original_request": original_request,
            "requirement_quotes": [
                {"quote": quote, "requirement_type": requirement_type}
                for quote, requirement_types in requirement_types_by_quote.items()
                for requirement_type in sorted(requirement_types)
            ],
        },
        "fact_based_questions": [
            (
                "Did the user explicitly ask for this operation, or did it arise "
                "only because the observed source grain/scope differs from the "
                "wording of the question?"
            ),
            (
                "Is there an existing source row or field that satisfies the "
                "requested scope without deriving, filtering, sorting, limiting, "
                "deduplicating, or reshaping rows?"
            ),
            (
                "If the source grain does not exactly match the requested scope, "
                "should the plan preserve source rows and record the scope "
                "mismatch instead of synthesizing a new answer?"
            ),
        ],
        "condition_count": condition_count,
        "allowed_revisions": [
            (
                "Keep the original user requirement quotes, but revise the "
                "interpretation to a source-record retrieval plan when the "
                "question asks for records and no transformation is authorized."
            ),
            (
                "Move a transformation into output_spec.transformations only if "
                "it can cite an explicit user quote or an observed knowledge rule "
                "that authorizes that operation."
            ),
            (
                "When the data scope/grain is mismatched, state the mismatch in "
                "intent.unresolved and evidence as a fact, then preserve source "
                "row grain."
            ),
        ],
    }


def _validate_free_text_action_claims(
    request: ToolCallRequest,
    *,
    arguments: Mapping[str, Any],
    requirement_types_by_quote: Mapping[str, set[str]],
    rule_quotes_by_type: Mapping[str, set[str]],
    declared_operations: set[str],
    question_conditions: Mapping[str, list[Any]] | None,
) -> ToolMessage | None:
    original_request = str(request.state.get("original_request") or "")
    for json_path, text in _free_text_plan_fields(arguments):
        action_claim = _detect_action_claim(text)
        if action_claim is None:
            continue
        operation, offending_text = action_claim
        if _operation_is_authorized(
            operation,
            requirement_types_by_quote=requirement_types_by_quote,
            rule_quotes_by_type=rule_quotes_by_type,
            declared_operations=declared_operations,
        ):
            continue

        facts = _authorization_facts(
            arguments=arguments,
            operation=operation,
            json_path=json_path,
            offending_text=offending_text,
            original_request=original_request,
            requirement_types_by_quote=requirement_types_by_quote,
            rule_quotes_by_type=rule_quotes_by_type,
            declared_operations=declared_operations,
            question_conditions=question_conditions,
        )
        return _plan_error(
            request,
            _plan_rejection_payload(
                code="UNAUTHORIZED_PLAN_ACTION",
                json_path=json_path,
                operation=operation,
                offending_text=offending_text,
                field_text=text,
                facts=facts,
                intent_reconsideration=_intent_reconsideration_payload(
                    operation=operation,
                    json_path=json_path,
                    offending_text=offending_text,
                    original_request=original_request,
                    requirement_types_by_quote=requirement_types_by_quote,
                    question_conditions=question_conditions,
                ),
                required_fix=(
                    "Reconsider the intent against the observed facts. If the "
                    "operation was inferred only to reconcile a source grain/scope "
                    "mismatch, revise the plan to preserve source rows and explain "
                    "the mismatch. If the operation is genuinely intended, declare "
                    "it in output_spec.transformations with a valid user or "
                    "knowledge authorization."
                ),
            ),
        )

    return None


def _question_structure_conditions(state: Mapping[str, Any]) -> dict[str, list[Any]] | None:
    if not state.get("question_structure_enforced"):
        return None
    structure = state.get("question_structure")
    if not isinstance(structure, Mapping):
        return None
    conditions = structure.get("conditions")
    if not isinstance(conditions, Mapping):
        return {}
    return {
        str(key): value
        for key, value in conditions.items()
        if isinstance(value, list)
    }


def _question_structure_target_count(state: Mapping[str, Any]) -> int | None:
    if not state.get("question_structure_enforced"):
        return None
    structure = state.get("question_structure")
    if not isinstance(structure, Mapping):
        return None
    targets = structure.get("targets")
    return len(targets) if isinstance(targets, list) else 0


def _quoted_requirement_support(state: Mapping[str, Any]) -> set[tuple[str, str]] | None:
    """Return quote/type pairs produced by the isolated question structure."""

    if not state.get("question_structure_enforced"):
        return None
    structure = state.get("question_structure")
    if not isinstance(structure, Mapping):
        return None

    question_text = str(
        structure.get("original_question") or state.get("original_request") or ""
    )
    supported: set[tuple[str, str]] = set()

    target_type_map = {
        "entity": "entity",
        "measure": "measure",
        "record_set": "output",
    }
    for target in structure.get("targets") or []:
        if not isinstance(target, Mapping):
            continue
        quote = str(target.get("quote") or "").strip()
        requirement_type = target_type_map.get(str(target.get("target_type") or ""))
        if quote and requirement_type and (not question_text or quote in question_text):
            supported.add((quote, requirement_type))

    constraint_type_map = {
        "entity": "entity",
        "filter": "filter",
        "geography": "entity",
        "grouping": "grouping",
        "limit": "limit",
        "ordering": "ordering",
        "output_shape": "output",
        "scope": "entity",
        "time_range": "time_range",
    }
    for constraint in structure.get("target_constraints") or []:
        if not isinstance(constraint, Mapping):
            continue
        quote = str(constraint.get("quote") or "").strip()
        requirement_type = constraint_type_map.get(
            str(constraint.get("constraint_type") or "")
        )
        if quote and requirement_type and (not question_text or quote in question_text):
            supported.add((quote, requirement_type))

    condition_type_map = {
        "calculations": "calculation",
        "filters": "filter",
        "groupings": "grouping",
        "limits": "limit",
        "orderings": "ordering",
        "output_columns": "output_column",
        "time_ranges": "time_range",
    }
    conditions = structure.get("conditions") or {}
    if isinstance(conditions, Mapping):
        for key, requirement_type in condition_type_map.items():
            for item in conditions.get(key) or []:
                quote = (
                    str(item.get("quote") or "").strip()
                    if isinstance(item, Mapping)
                    else str(item or "").strip()
                )
                if quote and (not question_text or quote in question_text):
                    supported.add((quote, requirement_type))

    operator_type_map = {
        "aggregate": "calculation",
        "derive": "calculation",
        "sort": "ordering",
        "limit": "limit",
    }
    for operator in structure.get("intent_operators") or []:
        if not isinstance(operator, Mapping):
            continue
        quote = str(operator.get("quote") or "").strip()
        operation = str(operator.get("operation") or "")
        requirement_type = operator_type_map.get(operation)
        if quote and requirement_type and (not question_text or quote in question_text):
            supported.add((quote, requirement_type))
        if (
            quote
            and operation == "aggregate"
            and str(operator.get("operator_type") or "") == "distribution"
            and (not question_text or quote in question_text)
        ):
            supported.add((quote, "grouping"))

    return supported


def _question_structure_context_capacity(
    state: Mapping[str, Any],
) -> dict[str, int] | None:
    """Return how many source-row context columns the isolated question supports."""

    if not state.get("question_structure_enforced"):
        return None
    structure = state.get("question_structure")
    if not isinstance(structure, Mapping):
        return None

    capacities = {role: 0 for role in _CONTEXT_COLUMN_ROLES}
    constraints = structure.get("target_constraints") or []
    for constraint in constraints:
        if not isinstance(constraint, Mapping):
            continue
        constraint_type = str(constraint.get("constraint_type") or "")
        if constraint_type == "time_range":
            capacities["time_key"] += 1
        if constraint_type in {"entity", "geography", "scope"}:
            capacities["entity_key"] += 1

    conditions = structure.get("conditions") or {}
    if isinstance(conditions, Mapping):
        capacities["time_key"] += len(conditions.get("time_ranges") or [])
        capacities["entity_key"] += len(conditions.get("filters") or [])
        capacities["entity_key"] += len(conditions.get("groupings") or [])

    output = structure.get("output") or {}
    if isinstance(output, Mapping):
        if str(output.get("row_grain_hint") or "") == "source_records":
            capacities["record_key"] += 1
        if str(output.get("preserve_source_rows") or "") == "true":
            capacities["record_key"] += 1

    return capacities


def _column_role(column: Any) -> str:
    return str(column.get("role") or "") if isinstance(column, Mapping) else ""


def _validate_plan_contract(
    request: ToolCallRequest,
    discovery: _DiscoveryState,
) -> ToolMessage | None:
    """用可核验的引用和状态约束计划，不推断自然语言语义。"""

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
    knowledge_rules = evidence.get("knowledge_rules") or []
    rule_quotes_by_type: dict[str, set[str]] = {}
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
                return _plan_error(
                    request,
                    f"knowledge rule cites unknown KnowledgeFact.fact_id {fact_id!r}.",
                )
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
    target_output_columns = output_columns
    if isinstance(execution_spec, Mapping):
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
    output_column_capacity = sum(
        1
        for requirement_types in requirement_types_by_quote.values()
        for requirement_type in requirement_types
        if requirement_type in _OUTPUT_COLUMN_REQUIREMENT_TYPES
    )
    if output_column_capacity > 0 and len(target_output_columns) > output_column_capacity:
        return _plan_error(
            request,
            (
                "output_spec.columns is final answer only. Move selector, "
                "filter, join, or context fields to execution_spec.supporting_fields; "
                f"got {len(target_output_columns)} final columns but "
                f"{output_column_capacity} answer-bearing requirement(s)."
            ),
        )
    if output_column_capacity > 0 and not target_output_columns:
        return _plan_error(
            request,
            "at least one non-context output column must satisfy the requested target.",
        )
    question_conditions = _question_structure_conditions(request.state)
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

    supported_requirements = _quoted_requirement_support(request.state)

    def user_authorization_error(operation: str, quote: str) -> str | None:
        requirement_types = requirement_types_by_quote.get(quote, set())
        allowed_types = _USER_TRANSFORMATION_REQUIREMENT_TYPES.get(
            operation, frozenset()
        )
        if not (requirement_types & allowed_types):
            return (
                f"user authorization for {operation!r} must cite an explicit "
                f"requirement typed as one of {sorted(allowed_types)}."
            )
        if supported_requirements is not None and not any(
            (quote, requirement_type) in supported_requirements
            for requirement_type in requirement_types & allowed_types
        ):
            return (
                f"user authorization for {operation!r} must be supported "
                "by the isolated question structure with the same quote "
                "and requirement type."
            )
        return None

    def knowledge_authorization_error(operation: str, quote: str) -> str | None:
        allowed_rule_types = _KNOWLEDGE_TRANSFORMATION_RULE_TYPES.get(
            operation, frozenset()
        )
        authorized_quotes = set().union(
            *(
                rule_quotes_by_type.get(rule_type, set())
                for rule_type in allowed_rule_types
            )
        )
        if quote not in authorized_quotes:
            return (
                f"knowledge authorization for {operation!r} must cite an "
                f"observed rule typed as one of {sorted(allowed_rule_types)}."
            )
        return None

    def fact_authorizes_operation(operation: str, fact_id: str) -> bool:
        fact = knowledge_facts_by_id.get(fact_id)
        if fact is None:
            return False
        allowed_kinds = _KNOWLEDGE_FACT_KINDS_BY_OPERATION.get(
            operation, frozenset()
        )
        fact_operations = {
            item.strip()
            for item in str(fact.operation or "").split(",")
            if item.strip()
        }
        return fact.kind in allowed_kinds or operation in fact_operations

    for transformation in transformations:
        if not isinstance(transformation, dict):
            return _plan_error(request, "transformations must be structured objects.")
        operation = str(transformation.get("operation") or "")
        authorization = transformation.get("authorization") or {}
        if not isinstance(authorization, Mapping):
            continue
        source = str(authorization.get("source") or "")
        quote = str(authorization.get("quote") or "").strip()
        if source == "user":
            if error_message := user_authorization_error(operation, quote):
                return _plan_error(request, error_message)
        if source == "knowledge":
            if error_message := knowledge_authorization_error(operation, quote):
                return _plan_error(request, error_message)
        if source not in {"user", "knowledge"}:
            return _plan_error(
                request,
                "context evidence cannot authorize a transformation.",
            )

    execution_operations: set[str] = set()
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
            if not authorized:
                return _plan_error(
                    request,
                    (
                        f"execution_spec.operations[{index}] for {operation!r} "
                        "requires an exact user quote, an observed knowledge rule, "
                        "or a valid KnowledgeFact.fact_id."
                    ),
                )
            execution_operations.add(operation)

    declared_operations = {
        str(transformation.get("operation") or "")
        for transformation in transformations
        if isinstance(transformation, Mapping)
        and str(transformation.get("operation") or "")
    } | execution_operations
    free_text_error = _validate_free_text_action_claims(
        request,
        arguments=arguments,
        requirement_types_by_quote=requirement_types_by_quote,
        rule_quotes_by_type=rule_quotes_by_type,
        declared_operations=declared_operations,
        question_conditions=question_conditions,
    )
    if free_text_error is not None:
        return free_text_error

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
    """计算结果进入状态后直接完成，不再要求模型复述完整表格。"""

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
    """在工具注入后替换系统提示词，避免 SDK 默认提示词进入模型请求。"""

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
    """强制执行 Discovery -> Plan -> Todos，并允许执行中修订计划。"""

    state_schema = BenchmarkDeepAgentState
    tools = [analyze_plan_tool]

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        discovery = _discovery_state(request.messages)
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
            _discovery_state(request.state["messages"])
            if plan is None or current_tool_name == "analyze_plan"
            else None
        )

        if plan is None and discovery is not None:
            if current_tool_name not in _SOURCE_DISCOVERY_TOOLS | {"analyze_plan"}:
                return _tool_error(
                    request,
                    "Only discovery tools are available before analyze_plan.",
                )
            if current_tool_name in _SOURCE_DISCOVERY_TOOLS:
                if discovery.context_ready:
                    return _tool_error(
                        request,
                        (
                            "Discovery is already sufficient for the current "
                            "question_structure and context evidence. Call "
                            "analyze_plan instead of exploring another source."
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
            request = _canonicalize_intent_requirements(request)
            request = _canonicalize_plan_quotes(request, discovery)
            request = _canonicalize_preserve_expected_row_count(request)
            request = _canonicalize_preserve_output_policy(request)
            request = _canonicalize_plan_steps(request)
            request = _canonicalize_revision(request)
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
        ):
            return _tool_error(
                request,
                "Call write_todos successfully before using any other tool.",
            )
        return handler(request)


class DisabledToolGuardMiddleware(AgentMiddleware[Any, None, Any]):
    """拒绝执行官方 profile 已排除的内置工具。"""

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
    """上下文只读；scratch 由后端默认规则管理。"""

    context_paths = ["/context/**"]
    return [
        FilesystemPermission(operations=["read"], paths=context_paths, mode="allow"),
        FilesystemPermission(operations=["write"], paths=context_paths, mode="deny"),
    ]
