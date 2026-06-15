from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from data_agent_baseline.agents.deep_state import BenchmarkDeepAgentState
from data_agent_baseline.tools.analyze_plan import analyze_plan_tool

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
_OUTPUT_COLUMN_REQUIREMENT_TYPES = frozenset(
    {
        "calculation",
        "measure",
        "output_column",
    }
)
_TRANSFORMATION_CONDITION_KEYS = {
    "aggregate": "calculations",
    "derive": "calculations",
    "filter": "filters",
    "sort": "orderings",
    "limit": "limits",
}
_EXPLICIT_REQUIREMENT_PATTERNS = {
    "calculation": re.compile(
        (
            r"(求和|合计|汇总|加总|总和|总额|平均|均值|最大|最高|最小|最低|"
            r"增长率|同比|环比|占比|比例|计算|算出|sum|average|mean|max|min|"
            r"total|calculate|aggregate|growth|ratio)"
        ),
        re.IGNORECASE,
    ),
    "ordering": re.compile(
        r"(排序|升序|降序|顺序|从高到低|从低到高|最高|最低|top|order|sort|ascending|descending)",
        re.IGNORECASE,
    ),
    "output_column": re.compile(
        (
            r"(列|字段|维度|包含|包括|显示|输出|返回|日期|年份|时间|省份|地区|城市|"
            r"column|field|dimension|include|show|return|date|year|province|region|city)"
        ),
        re.IGNORECASE,
    ),
}
_DISCOVERY_CALCULATION_INFERENCE_PATTERN = re.compile(
    (
        r"(sum\s*\(|\.sum\s*\(|groupby\s*\(|group\s+by|aggregate|aggregation|"
        r"calculate|calculation|average|mean\s*\(|max\s*\(|min\s*\(|ratio|"
        r"growth|total\s+(?:row|value|amount|sum|gdp|record)|"
        r"求和|合计|汇总|加总|总和|总额|平均|均值|最大|最小|增长率|同比|环比|"
        r"占比|比例|计算|算出|全国总计|全国合计|国家总计|国家合计)"
    ),
    re.IGNORECASE,
)
_DISCOVERY_ORDERING_INFERENCE_PATTERN = re.compile(
    r"(\bsort(?:ed)?\s*\(|\.sort\s*\(|order\s+by|ascending|descending|排序|升序|降序)",
    re.IGNORECASE,
)
_DISCOVERY_LIMIT_INFERENCE_PATTERN = re.compile(
    r"(\blimit\s+\d+|\btop\s+\d+|head\s*\(\s*\d+|tail\s*\(\s*\d+|前\s*\d+\s*个|后\s*\d+\s*个)",
    re.IGNORECASE,
)
_DISCOVERY_FILTER_INFERENCE_PATTERN = re.compile(
    r"(\bwhere\b|\.query\s*\(|filter\s*\(|筛选|过滤|条件)",
    re.IGNORECASE,
)
_DISCOVERY_DIMENSION_INFERENCE_PATTERN = re.compile(
    (
        r"((unique|distinct|values|list|enumerate|print)\W{0,40}"
        r"(province|region|city|date|year|enddate|time|省份|地区|城市|日期|年份|时间))|"
        r"((province|region|city|date|year|enddate|time|省份|地区|城市|日期|年份|时间)"
        r"\W{0,40}(unique|distinct|values|list|enumerate|print))"
    ),
    re.IGNORECASE,
)
_REVISION_FIELDS = (
    "intent",
    "output_spec",
    "evidence",
    "steps",
    "delegation_candidates",
)
_UNSUPPORTED_MEDIA_SUFFIXES = frozenset(
    {
        ".aac",
        ".avi",
        ".bmp",
        ".flac",
        ".gif",
        ".jpeg",
        ".jpg",
        ".m4a",
        ".mov",
        ".mp3",
        ".mp4",
        ".ogg",
        ".pdf",
        ".png",
        ".tif",
        ".tiff",
        ".wav",
        ".webp",
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
        required = 2 if not self.knowledge_available or self.needs_cross_validation else 1
        return len(self.context_sources) >= required

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


def _unsupported_media_read_error(request: ToolCallRequest) -> ToolMessage | None:
    if str(request.tool_call.get("name") or "") != "read_file":
        return None
    arguments = request.tool_call.get("args") or {}
    if not isinstance(arguments, Mapping):
        return None
    file_path = str(
        arguments.get("file_path")
        or arguments.get("path")
        or ""
    ).replace("\\", "/")
    suffix = PurePosixPath(file_path).suffix.casefold()
    if suffix not in _UNSUPPORTED_MEDIA_SUFFIXES:
        return None
    return _tool_error(
        request,
        (
            f"Direct read_file for {suffix or 'binary'} files is disabled for "
            "this model endpoint because it returns multimodal file blocks. "
            "Use read_doc for text/PDF files, or specialized structured-data "
            "tools for CSV, JSON, and SQLite sources instead."
        ),
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
    return matches[0] if len(matches) == 1 else None


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
    normalized_rules: list[Any] = []
    changed = False
    for rule in evidence.get("knowledge_rules") or []:
        if not isinstance(rule, dict):
            normalized_rules.append(rule)
            continue
        normalized_rule = dict(rule)
        quote = str(rule.get("quote") or "").strip()
        canonical_quote = _canonical_knowledge_quote(
            quote,
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


def _argument_text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        values: list[str] = []
        for item in value.values():
            values.extend(_argument_text_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_argument_text_values(item))
        return values
    return []


def _question_structure_has_condition(
    conditions: Mapping[str, list[Any]],
    *keys: str,
) -> bool:
    return any(bool(conditions.get(key)) for key in keys)


def _discovery_question_structure_error(
    request: ToolCallRequest,
    current_tool_name: str,
) -> ToolMessage | None:
    """Reject exploratory calls that add unstated operations to the question."""

    if current_tool_name not in _SOURCE_DISCOVERY_TOOLS:
        return None
    conditions = _question_structure_conditions(request.state)
    if conditions is None:
        return None
    arguments = request.tool_call.get("args") or {}
    if not isinstance(arguments, Mapping):
        return None
    argument_text = "\n".join(_argument_text_values(arguments))
    if not argument_text:
        return None

    if (
        not _question_structure_has_condition(conditions, "calculations")
        and _DISCOVERY_CALCULATION_INFERENCE_PATTERN.search(argument_text)
    ):
        return _tool_error(
            request,
            (
                "Discovery call rejected by question_structure: "
                "conditions.calculations is empty, but the tool arguments/code "
                "attempt to explore an aggregate, derived calculation, or total. "
                "Inspect direct target fields only, or call analyze_plan once the "
                "source is clear."
            ),
        )
    if (
        not _question_structure_has_condition(conditions, "orderings")
        and _DISCOVERY_ORDERING_INFERENCE_PATTERN.search(argument_text)
    ):
        return _tool_error(
            request,
            (
                "Discovery call rejected by question_structure: "
                "conditions.orderings is empty, but the tool arguments/code "
                "attempt to explore sorting or ordered ranking."
            ),
        )
    if (
        not _question_structure_has_condition(conditions, "limits")
        and _DISCOVERY_LIMIT_INFERENCE_PATTERN.search(argument_text)
    ):
        return _tool_error(
            request,
            (
                "Discovery call rejected by question_structure: "
                "conditions.limits is empty, but the tool arguments/code attempt "
                "to explore a top/limit slice."
            ),
        )
    if (
        not _question_structure_has_condition(conditions, "filters")
        and _DISCOVERY_FILTER_INFERENCE_PATTERN.search(argument_text)
    ):
        return _tool_error(
            request,
            (
                "Discovery call rejected by question_structure: "
                "conditions.filters is empty, but the tool arguments/code attempt "
                "to explore a filtered subset."
            ),
        )
    if (
        not _question_structure_has_condition(
            conditions,
            "output_columns",
            "groupings",
            "time_ranges",
            "filters",
        )
        and _DISCOVERY_DIMENSION_INFERENCE_PATTERN.search(argument_text)
    ):
        return _tool_error(
            request,
            (
                "Discovery call rejected by question_structure: no output, "
                "grouping, time, or filter condition authorizes enumerating "
                "helper dimensions such as geography or date values."
            ),
        )
    return None


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
    requirement_types_by_quote = {
        str(item.get("quote") or "").strip(): str(
            item.get("requirement_type") or ""
        )
        for item in requirements
        if isinstance(item, dict)
    }
    if not requirement_types_by_quote or any(
        not quote or quote not in original_request
        for quote in requirement_types_by_quote
    ):
        return _plan_error(
            request,
            "every intent requirement quote must occur verbatim in original_request.",
        )
    for quote, requirement_type in requirement_types_by_quote.items():
        explicit_pattern = _EXPLICIT_REQUIREMENT_PATTERNS.get(requirement_type)
        if explicit_pattern is None or explicit_pattern.search(quote):
            continue
        return _plan_error(
            request,
            (
                f"{requirement_type} requirements must quote an explicit "
                f"{requirement_type} instruction from the original request; "
                f"{quote!r} is not sufficient."
            ),
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

    knowledge_rules = evidence.get("knowledge_rules") or []
    rule_quotes_by_type: dict[str, set[str]] = {}
    for rule in knowledge_rules:
        if not isinstance(rule, dict):
            continue
        source_path = str(rule.get("source_path") or "").replace("\\", "/")
        quote = str(rule.get("quote") or "").strip()
        rule_type = str(rule.get("rule_type") or "")
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
    output_column_capacity = sum(
        1
        for requirement_type in requirement_types_by_quote.values()
        if requirement_type in _OUTPUT_COLUMN_REQUIREMENT_TYPES
    )
    if len(output_columns) > output_column_capacity:
        return _plan_error(
            request,
            (
                "each output column must be backed by a distinct measure, "
                "calculation, or explicit output_column requirement; got "
                f"{len(output_columns)} columns but {output_column_capacity} "
                "output-bearing requirements."
            ),
        )
    question_conditions = _question_structure_conditions(request.state)
    question_target_count = _question_structure_target_count(request.state)
    if question_conditions is not None and question_target_count is not None:
        explicit_output_columns = len(question_conditions.get("output_columns", []))
        question_output_capacity = question_target_count + explicit_output_columns
        if len(output_columns) > question_output_capacity:
            return _plan_error(
                request,
                (
                    "output_spec.columns exceeds the isolated question structure: "
                    f"{len(output_columns)} columns requested by the plan, but the "
                    f"question structure supports {question_target_count} target "
                    f"column(s) plus {explicit_output_columns} explicit output "
                    "column(s)."
                ),
            )
    transformations = output_spec.get("transformations") or []
    row_policy = str(output_spec.get("row_policy") or "")
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
            requirement_type = requirement_types_by_quote.get(quote)
            allowed_types = _USER_TRANSFORMATION_REQUIREMENT_TYPES.get(
                operation, frozenset()
            )
            if requirement_type not in allowed_types:
                return _plan_error(
                    request,
                    (
                        f"user authorization for {operation!r} must cite an explicit "
                        f"requirement typed as one of {sorted(allowed_types)}."
                    ),
                )
            condition_key = _TRANSFORMATION_CONDITION_KEYS.get(operation)
            if (
                question_conditions is not None
                and condition_key is not None
                and not question_conditions.get(condition_key)
            ):
                return _plan_error(
                    request,
                    (
                        f"user authorization for {operation!r} conflicts with the "
                        "isolated question structure: "
                        f"conditions.{condition_key} is empty."
                    ),
                )
        if source == "knowledge":
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
                return _plan_error(
                    request,
                    (
                        f"knowledge authorization for {operation!r} must cite an "
                        f"observed rule typed as one of {sorted(allowed_rule_types)}."
                    ),
                )
        if source not in {"user", "knowledge"}:
            return _plan_error(
                request,
                "context evidence cannot authorize a transformation.",
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
        return _plan_error(request, "a revision must make an actual plan change.")
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
                structure_error = _discovery_question_structure_error(
                    request,
                    current_tool_name,
                )
                if structure_error is not None:
                    return structure_error
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
            request = _canonicalize_plan_quotes(request, discovery)
            contract_error = _validate_plan_contract(request, discovery)
            if contract_error is not None:
                return contract_error
        if current_tool_name == "write_todos" and plan is not None:
            todos = request.tool_call.get("args", {}).get("todos") or []
            todo_contents = [
                str(todo.get("content") or "").strip()
                for todo in todos
                if isinstance(todo, dict)
            ]
            if todo_contents != plan.get("steps"):
                return _tool_error(
                    request,
                    "write_todos contents must exactly match analysis_plan.steps.",
                )
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


class TextOnlyReadFileMiddleware(AgentMiddleware[Any, None, Any]):
    """Return text errors for media files instead of multimodal content blocks."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        unsupported_media_error = _unsupported_media_read_error(request)
        if unsupported_media_error is not None:
            return unsupported_media_error
        return handler(request)


class HideUnavailableToolsMiddleware(AgentMiddleware[Any, None, Any]):
    """隐藏基准环境不允许模型调用的通用文件和 shell 工具。"""

    hidden_tools = frozenset(
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

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        filtered_tools = [
            item for item in request.tools if tool_name(item) not in self.hidden_tools
        ]
        return handler(request.override(tools=filtered_tools))


def workspace_permissions() -> list[FilesystemPermission]:
    """上下文只读；scratch 由后端默认规则管理。"""

    context_paths = ["/context/**"]
    return [
        FilesystemPermission(operations=["read"], paths=context_paths, mode="allow"),
        FilesystemPermission(operations=["write"], paths=context_paths, mode="deny"),
    ]
