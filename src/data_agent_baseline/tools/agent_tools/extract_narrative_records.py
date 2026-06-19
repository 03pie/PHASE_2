from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from data_agent_baseline.agents.deep_state import BenchmarkDeepAgentState
from data_agent_baseline.agents.semantic_layer import parse_knowledge_content
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools._helpers import (
    DOC_SUFFIXES,
    error,
    extract_pdf_text,
    resolve_context_path,
    virtual_path,
)
from data_agent_baseline.tools.answer import (
    answer_value_hash,
    validate_prepared_answer,
)
from data_agent_baseline.tools.observed_sources import (
    merge_observed_sources,
    sample_hash,
)

_CJK_TIME_RE = re.compile(
    r"[\u8fd1\u7b2c]?(?:\d+|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u4e24])+[\u5e74\u6708\u65e5\u5b63\u5468\u5929]"
)
_EN_TIME_RE = re.compile(
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)[-_\s]?(?:year|month|week|day|quarter)s?",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_VALUE_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_RECORD_RE = re.compile(
    r"(?:\u6863\u6848|\u6218\u7565\u5355\u5143|archive)\s*\d+",
    re.IGNORECASE,
)
_RECORD_KEY_RE = re.compile(
    r"(?:\u6863\u6848|\u6218\u7565\u5355\u5143|\u6761\u76ee|\u8bb0\u5f55|archive)\s*`?(?P<id>\d+)`?",
    re.IGNORECASE,
)
_MISSING_RE = re.compile(
    "|".join(
        [
            "\u7f3a\u5931",
            "nan",
            "\u65e0\u6cd5",
            "\u4e0d\u8db3",
            "\u672a\u6ee1",
            "\u672a\u6709\u8bb0\u5f55",
            "\u672a\u63d0\u4f9b",
            "\u672a\u77e5",
            "\u65e0\u6cd5\u8bc4\u4f30",
            "\u65e0\u6cd5\u786e\u5b9a",
            "\u65e0\u6cd5\u83b7\u53d6",
            "missing",
            "unknown",
            "unavailable",
            "not available",
        ]
    ),
    re.IGNORECASE,
)
_ANNUALIZED_RE = re.compile(
    r"(?:\u5e74\u5316|annualized|annual)",
    re.IGNORECASE,
)
_SECTION_SWITCH_RE = re.compile(
    r"(?:\u6210\u7acb\u4ee5\u6765|since inception)",
    re.IGNORECASE,
)
_RETURN_TERMS = (
    "\u56de\u62a5\u7387",
    "\u6536\u76ca\u7387",
    "return rate",
    "return",
    "rate",
)
_FINAL_VALUE_RE = re.compile(
    (
        r"(?:(?:final|audited|official|confirmed|reconciled)"
        r"|(?:\u6700\u7ec8|\u786e\u8ba4|\u6838\u5b9e|\u51c6\u786e|"
        r"\u4fee\u6b63(?:\u4e3a)?|\u7cbe\u786e\u4fee\u6b63(?:\u4e3a)?))"
        r"[^0-9+-]{0,100}"
        r"(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)"
    ),
    re.IGNORECASE,
)
_PROVISIONAL_VALUE_RE = re.compile(
    r"(?:initial|initially|preliminary|estimated|estimate|logged|"
    r"tentative|provisional|early|earlier|\u521d\u6b65|\u6682\u4f30|"
    r"\u539f\u5148|\u65e9\u671f)",
    re.IGNORECASE,
)
_BEFORE_MARKER_RE = re.compile(
    r"(?:before|prior to|pre[-\s]?|"
    r"\u4ea4\u6613\u524d|\u8f6c\u8ba9\u524d|"
    r"\u672c\u6b21\u4ea4\u6613\u524d|\u8be5\u7b14\u4ea4\u6613\u53d1\u751f\u524d|"
    r"\u6b64\u6b21\u4ea4\u6613\u524d|\u5728[^。；;]{0,30}?\u524d)",
    re.IGNORECASE,
)
_AFTER_MARKER_RE = re.compile(
    r"(?:after|following|post[-\s]?|"
    r"\u4ea4\u6613\u5b8c\u6210\u540e|\u4ea4\u6613\u540e|"
    r"\u8f6c\u8ba9\u540e|\u5b8c\u6210\u540e|\u4e4b\u540e)",
    re.IGNORECASE,
)
_SHARE_VALUE_RE = re.compile(
    r"(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)\s*(?:\u80a1|shares?)",
    re.IGNORECASE,
)
_PERCENT_VALUE_RE = re.compile(
    r"(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def _candidate_payload(
    *,
    columns: list[str],
    rows: list[list[Any]],
    audit: dict[str, Any] | None,
    validation_error: str,
) -> dict[str, Any]:
    raw_paths = audit.get("source_paths") if isinstance(audit, dict) else []
    return {
        "columns": columns,
        "rows": rows,
        "audit": audit,
        "column_count": len(columns),
        "row_count": len(rows),
        "code_context_paths": [
            str(path).replace("\\", "/")
            for path in raw_paths
            if str(path).strip()
        ],
        "validation_error": validation_error,
    }


def _value_is_empty(value: Any) -> bool:
    return value is None or value == ""


def _knowledge_quote_for_field(
    *,
    analysis_plan: Mapping[str, Any],
    source_field: str,
) -> str:
    normalized_field = source_field.casefold()
    evidence = analysis_plan.get("evidence")
    if isinstance(evidence, Mapping):
        for rule in evidence.get("knowledge_rules") or []:
            if not isinstance(rule, Mapping):
                continue
            quote = str(rule.get("quote") or "")
            if normalized_field and normalized_field in quote.casefold():
                return quote
    return ""


def _source_binding_paths(
    *,
    analysis_plan: Mapping[str, Any],
    source_field: str,
) -> set[str]:
    execution_spec = analysis_plan.get("execution_spec")
    if not isinstance(execution_spec, Mapping):
        return set()
    normalized_field = source_field.casefold()
    paths: set[str] = set()
    for binding in execution_spec.get("source_bindings") or []:
        if not isinstance(binding, Mapping):
            continue
        if str(binding.get("source_field") or "").casefold() != normalized_field:
            continue
        for path in binding.get("source_paths") or []:
            normalized = str(path or "").replace("\\", "/")
            if normalized:
                paths.add(normalized)
    return paths


def _time_terms(*values: str) -> list[str]:
    terms: list[str] = []
    for value in values:
        for term in _CJK_TIME_RE.findall(value or ""):
            normalized = term.lstrip("\u8fd1\u7b2c")
            if normalized and normalized not in terms:
                terms.append(normalized)
        for term in _EN_TIME_RE.findall(value or ""):
            normalized = term.replace("-", "").replace("_", "").replace(" ", "").casefold()
            if normalized and normalized not in terms:
                terms.append(normalized)
    return terms


def _compact_text(value: str) -> str:
    return re.sub(r"[-_\s]+", "", value.casefold())


def _line_mentions_target(line: str, time_terms: list[str]) -> bool:
    compact_line = _compact_text(line)
    if time_terms and not any(_compact_text(term) in compact_line for term in time_terms):
        return False
    lowered = line.casefold()
    return any(term in lowered for term in _RETURN_TERMS)


def _infer_window(
    lines: list[str],
    *,
    time_terms: list[str],
    start_line: int | None,
    end_line: int | None,
) -> tuple[int, int]:
    if start_line is not None:
        start_index = max(0, start_line - 1)
    else:
        start_index = 0
        for index, line in enumerate(lines):
            if _RECORD_RE.search(line) and _line_mentions_target(line, time_terms):
                start_index = index
                break
    if end_line is not None:
        end_index = min(len(lines), end_line)
    else:
        end_index = len(lines)
        for index in range(start_index + 1, len(lines)):
            line = lines[index]
            if _SECTION_SWITCH_RE.search(line) and not _line_mentions_target(
                line,
                time_terms,
            ):
                end_index = index
                break
    return start_index, end_index


def _target_segment(line: str, time_terms: list[str]) -> str:
    lowered = line.casefold()
    positions = [
        lowered.find(term)
        for term in time_terms
        if term and lowered.find(term) >= 0
    ]
    if not positions:
        positions = [
            lowered.find(term)
            for term in _RETURN_TERMS
            if term and lowered.find(term) >= 0
        ]
    if not positions:
        return ""
    segment = line[min(positions) :]
    annualized = _ANNUALIZED_RE.search(segment)
    if annualized is not None and annualized.start() > 0:
        segment = segment[: annualized.start()]
    return segment


def _extract_value(line: str, time_terms: list[str]) -> str:
    segment = _target_segment(line, time_terms)
    if not segment:
        return ""
    if _MISSING_RE.search(segment):
        return ""
    numbers = _NUMBER_RE.findall(segment)
    return numbers[-1] if numbers else ""


def _coerce_number(value: str) -> float | str:
    cleaned = value.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def _coerce_percentage_ratio(value: str) -> float | str:
    coerced = _coerce_number(value)
    if isinstance(coerced, float):
        return coerced / 100
    return coerced


def _nearby_field_value(line: str, match: re.Match[str]) -> Any:
    field_end = match.end()
    start = max(0, match.start() - 180)
    end = min(len(line), field_end + 220)
    window = line[start:end]
    relative_start = match.start() - start
    relative_end = field_end - start

    before_numbers = list(_VALUE_NUMBER_RE.finditer(window[:relative_start]))
    if before_numbers:
        last = before_numbers[-1]
        bridge = window[last.end() : relative_start]
        if relative_start - last.end() <= 90 and not re.search(r"[.;。；]", bridge):
            return _coerce_number(last.group())

    after_text = window[relative_end:]
    final_match = _FINAL_VALUE_RE.search(after_text)
    after_numbers = list(_VALUE_NUMBER_RE.finditer(after_text))
    if after_numbers and after_numbers[0].start() <= 120:
        if final_match is not None:
            before_final = after_text[: final_match.start()]
            if _PROVISIONAL_VALUE_RE.search(before_final):
                return _coerce_number(final_match.group("value"))
        return _coerce_number(after_numbers[0].group())

    if final_match is not None:
        return _coerce_number(final_match.group("value"))

    return ""


def _nearby_value_after_anchor(line: str, match: re.Match[str]) -> Any:
    field_end = match.end()
    end = min(len(line), field_end + 220)
    after_text = line[field_end:end]
    final_match = _FINAL_VALUE_RE.search(after_text)
    after_numbers = list(_VALUE_NUMBER_RE.finditer(after_text))
    if after_numbers and after_numbers[0].start() <= 120:
        if final_match is not None:
            before_final = after_text[: final_match.start()]
            bridge_after_first = after_text[
                after_numbers[0].end() : final_match.start()
            ]
            revision_bridge = re.search(
                r"(?:but|however|after|revalu|revis|correct|"
                r"\u4f46|\u7ecf\u8fc7|\u91cd\u65b0\u4f30\u503c|"
                r"\u4fee\u6b63|\u66f4\u6b63)",
                bridge_after_first,
                re.IGNORECASE,
            )
            if (
                _PROVISIONAL_VALUE_RE.search(before_final) or revision_bridge
            ) and not re.search(r"[.;。；]", bridge_after_first):
                return _coerce_number(final_match.group("value"))
        return _coerce_number(after_numbers[0].group())
    if final_match is not None:
        return _coerce_number(final_match.group("value"))
    return ""


def _extract_field_rows(
    lines: list[str],
    *,
    source_field: str,
    start_line: int | None,
    end_line: int | None,
    max_records: int,
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    field = source_field.strip()
    if not field:
        return [], []
    field_pattern = re.compile(re.escape(field), re.IGNORECASE)
    start_index = max(0, start_line - 1) if start_line is not None else 0
    end_index = min(len(lines), end_line) if end_line is not None else len(lines)
    rows: list[list[Any]] = []
    evidence: list[dict[str, Any]] = []
    for index in range(start_index, end_index):
        line = lines[index]
        matches = list(field_pattern.finditer(line))
        if not matches:
            continue
        context_line = " ".join(lines[index : min(len(lines), index + 3)])
        line_values: list[Any] = []
        for match in matches:
            value = _nearby_field_value(line, match)
            if value == "":
                value = _nearby_field_value(context_line, match)
            if value == "":
                continue
            rows.append([value])
            line_values.append(value)
            if len(rows) >= max_records:
                break
        if line_values:
            evidence.append(
                {
                    "line_number": index + 1,
                    "record_count": len(line_values),
                    "value": line_values[-1],
                    "content": context_line,
                }
            )
        if len(rows) >= max_records:
            break
    return rows, evidence


def _observed_field_name(lines: list[str], source_field: str) -> str:
    field = source_field.strip()
    if not field:
        return source_field
    pattern = re.compile(re.escape(field), re.IGNORECASE)
    for line in lines:
        match = pattern.search(line)
        if match is not None:
            return match.group(0)
    return source_field


def _extract_rows(
    lines: list[str],
    *,
    source_field: str,
    knowledge_quote: str,
    start_line: int | None,
    end_line: int | None,
    max_records: int,
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    field_rows, field_evidence = _extract_field_rows(
        lines,
        source_field=source_field,
        start_line=start_line,
        end_line=end_line,
        max_records=max_records,
    )
    if field_rows:
        return field_rows, field_evidence

    aliases = _field_aliases(source_field, None)
    start_index = max(0, start_line - 1) if start_line is not None else 0
    end_index = min(len(lines), end_line) if end_line is not None else len(lines)
    rows: list[list[Any]] = []
    evidence: list[dict[str, Any]] = []
    for index in range(start_index, end_index):
        line = lines[index]
        extracted = _extract_multi_field_value(
            line,
            source_field=source_field,
            aliases=aliases,
        )
        if extracted is None:
            continue
        raw_value, normalized_value = extracted
        rows.append([raw_value])
        evidence.append(
            {
                "line_number": index + 1,
                "record_count": 1,
                "value": raw_value,
                "normalized_value": normalized_value,
                "content": line,
            }
        )
        if len(rows) >= max_records:
            break
    if rows:
        return rows, evidence

    time_terms = _time_terms(source_field, knowledge_quote)
    if not time_terms:
        time_terms = _time_terms(knowledge_quote)
    start_index, end_index = _infer_window(
        lines,
        time_terms=time_terms,
        start_line=start_line,
        end_line=end_line,
    )
    rows: list[list[Any]] = []
    evidence: list[dict[str, Any]] = []
    for index in range(start_index, end_index):
        line = lines[index]
        matches = list(_RECORD_RE.finditer(line))
        if not matches:
            continue
        value = _extract_value(line, time_terms)
        for _match in matches:
            rows.append([value])
            if len(rows) >= max_records:
                break
        evidence.append(
            {
                "line_number": index + 1,
                "record_count": len(matches),
                "value": value,
                "content": line,
            }
        )
        if len(rows) >= max_records:
            break
    return rows, evidence


def _default_aliases_for_field(source_field: str) -> list[str]:
    field = source_field.strip()
    if not field:
        return []
    aliases = [field]
    token_aliases = {
        "id": ["id", "\u7f16\u53f7", "\u7f16\u7801", "\u8bc6\u522b\u7801", "\u6807\u8bc6\u7b26"],
        "code": ["code", "\u7f16\u7801", "\u8bc6\u522b\u7801", "\u6807\u8bc6\u7b26"],
        "personal": ["personal", "\u4e2a\u4eba", "\u4eba\u4e8b", "\u5185\u90e8"],
        "total": ["total"],
        "fund": ["fund"],
        "net": ["net"],
        "value": ["value"],
        "nv": ["nv", "net value", "\u51c0\u503c", "\u8d44\u4ea7\u51c0\u503c"],
        "aum": [
            "aum",
            "assets under management",
            "\u8d44\u4ea7\u7ba1\u7406\u89c4\u6a21",
            "\u8d44\u4ea7\u89c4\u6a21",
            "\u7ba1\u7406\u89c4\u6a21",
        ],
        "asset": ["asset", "assets", "\u8d44\u4ea7"],
        "scale": ["scale", "\u89c4\u6a21"],
        "amount": ["amount", "\u6570\u91cf", "\u91d1\u989d"],
        "quantity": ["quantity", "\u6570\u91cf"],
        "percent": ["percent", "percentage", "%", "\u767e\u5206\u6bd4", "\u6bd4\u4f8b"],
        "pct": ["pct", "%", "\u767e\u5206\u6bd4", "\u6bd4\u4f8b"],
    }
    token_text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", field)
    tokens = {
        token.casefold()
        for token in re.split(r"[^A-Za-z0-9\u3400-\u9fff]+", token_text)
        if token.strip()
    }
    compact_field = re.sub(r"[^A-Za-z0-9]+", "", field).casefold()
    tokens.update(
        token
        for token in token_aliases
        if len(token) >= 2 and token in compact_field
    )
    specific_aliases: list[str] = []
    if {"personal", "code"} <= tokens:
        specific_aliases.extend(
            [
                "PersonalCode",
                "\u4e2a\u4eba\u8bc6\u522b\u7801",
                "\u5185\u90e8\u8bc6\u522b\u7801",
                "\u4eba\u4e8b\u7f16\u7801",
            ]
        )
    if "total" in tokens and ({"nv", "aum", "net", "value", "asset"} & tokens):
        specific_aliases.extend(
            [
                "\u603b\u8d44\u4ea7\u51c0\u503c",
                "\u8d44\u4ea7\u603b\u89c4\u6a21",
                "\u603b\u51c0\u503c",
                "\u7ba1\u7406\u89c4\u6a21",
                "TotalAUM",
            ]
        )
    aliases.extend(specific_aliases)
    broad_component_tokens: set[str] = set()
    if {"personal", "code"} <= tokens:
        broad_component_tokens.add("personal")
    for token in tokens:
        if token in broad_component_tokens:
            continue
        aliases.extend(token_aliases.get(token, []))
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _field_aliases(
    source_field: str,
    field_aliases: Mapping[str, Any] | None,
) -> list[str]:
    aliases = _default_aliases_for_field(source_field)
    if isinstance(field_aliases, Mapping):
        for key, values in field_aliases.items():
            if str(key).casefold() != source_field.casefold():
                continue
            if isinstance(values, list):
                aliases.extend(str(value) for value in values if str(value).strip())
            elif str(values).strip():
                aliases.append(str(values))
    return list(dict.fromkeys(alias.strip() for alias in aliases if alias.strip()))


def _field_descriptor(source_field: str, aliases: list[str]) -> str:
    return " ".join([source_field, *aliases]).casefold()


def _is_percentage_field(source_field: str, aliases: list[str]) -> bool:
    descriptor = _field_descriptor(source_field, aliases)
    return any(
        token in descriptor
        for token in (
            "pct",
            "percent",
            "percentage",
            "ratio",
            "\u6bd4\u4f8b",
            "\u767e\u5206\u6bd4",
        )
    )


def _is_quantity_field(source_field: str, aliases: list[str]) -> bool:
    descriptor = _field_descriptor(source_field, aliases)
    return any(
        token in descriptor
        for token in (
            "sum",
            "qty",
            "quantity",
            "amount",
            "share",
            "shares",
            "\u6570\u91cf",
            "\u6301\u80a1",
            "\u80a1",
        )
    )


def _is_before_field(source_field: str, aliases: list[str]) -> bool:
    descriptor = _field_descriptor(source_field, aliases)
    return any(
        token in descriptor
        for token in (
            "before",
            "prior",
            "pre",
            "\u524d",
        )
    )


def _before_segment(line: str) -> str:
    before_match = _BEFORE_MARKER_RE.search(line)
    if before_match is None:
        return ""
    segment = line[before_match.start() :]
    after_match = _AFTER_MARKER_RE.search(segment)
    if after_match is not None and after_match.start() > 0:
        segment = segment[: after_match.start()]
    return segment


def _extract_before_after_value(
    line: str,
    *,
    source_field: str,
    aliases: list[str],
) -> tuple[Any, Any] | None:
    if not _is_before_field(source_field, aliases):
        return None
    segment = _before_segment(line)
    if not segment:
        return None
    if _MISSING_RE.search(segment):
        return "", ""
    if _is_percentage_field(source_field, aliases):
        match = _PERCENT_VALUE_RE.search(segment)
        if match is None:
            return None
        value = _coerce_percentage_ratio(match.group("value"))
        return value, value
    if _is_quantity_field(source_field, aliases):
        match = _SHARE_VALUE_RE.search(segment)
        if match is None:
            return None
        value = _coerce_number(match.group("value"))
        return value, value
    return None


def _extract_generic_near_alias(line: str, aliases: list[str]) -> Any | None:
    for alias in aliases:
        match = re.search(re.escape(alias), line, re.IGNORECASE)
        if match is None:
            continue
        value = _nearby_value_after_anchor(line, match)
        if value != "":
            return value
    return None


def _extract_multi_field_value(
    line: str,
    *,
    source_field: str,
    aliases: list[str],
) -> tuple[Any, Any] | None:
    before_after_value = _extract_before_after_value(
        line,
        source_field=source_field,
        aliases=aliases,
    )
    if before_after_value is not None:
        return before_after_value
    value = _extract_generic_near_alias(line, aliases)
    if value is not None:
        return value, value
    return None


def _line_matches_record_anchor(line: str, record_anchor: str | None) -> bool:
    if not record_anchor:
        return True
    match = re.search(re.escape(record_anchor), line, re.IGNORECASE)
    if match is None:
        return False
    before_match = _BEFORE_MARKER_RE.search(line)
    if before_match is not None:
        if match.start() <= before_match.start():
            return True
        return record_anchor.casefold() in before_match.group(0).casefold()
    return match.start() <= 120


def _record_key_from_line(line: str) -> str | None:
    match = _RECORD_KEY_RE.search(line)
    if match is None:
        return None
    return match.group("id")


def _extraction_cache_key(
    *,
    path: str,
    columns: list[str],
    record_anchor: str | None,
) -> str:
    field_key = ",".join(column.casefold() for column in columns)
    anchor_key = str(record_anchor or "").casefold()
    return f"{path}|{field_key}|{anchor_key}"


def _extraction_records_from_rows(
    *,
    columns: list[str],
    rows: list[list[Any]],
    line_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        evidence = (
            line_evidence[row_index]
            if row_index < len(line_evidence)
            and isinstance(line_evidence[row_index], Mapping)
            else {}
        )
        record_key = str(evidence.get("record_key") or "").strip()
        line_number = evidence.get("line_number")
        if not record_key:
            if isinstance(line_number, int):
                record_key = f"line:{line_number}"
            else:
                record_key = f"row:{row_index + 1}"
        values = {
            str(field): value
            for field, value in (
                evidence.get("values", {}).items()
                if isinstance(evidence.get("values"), Mapping)
                else []
            )
        }
        for field_index, field in enumerate(columns):
            if field_index < len(row) and field not in values:
                values[field] = row[field_index]
        record: dict[str, Any] = {
            "record_key": record_key,
            "values": values,
            "matched_fields": [
                str(field)
                for field in (evidence.get("matched_fields") or [])
                if str(field).strip()
            ],
        }
        if isinstance(line_number, int):
            record["line_number"] = line_number
        normalized_values = evidence.get("normalized_values")
        if isinstance(normalized_values, Mapping):
            record["normalized_values"] = dict(normalized_values)
        content = str(evidence.get("content") or "").strip()
        if content:
            record["content"] = content[:400]
        records.append(record)
    return records


def _cache_entries_for_extraction(
    *,
    state: Mapping[str, Any],
    path: str,
    columns: list[str],
    record_anchor: str | None,
) -> list[dict[str, Any]]:
    cache_key = _extraction_cache_key(
        path=path,
        columns=columns,
        record_anchor=record_anchor,
    )
    entries = state.get("narrative_extraction_cache")
    if not isinstance(entries, list):
        return []
    records: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("cache_key") or "") != cache_key:
            continue
        for record in entry.get("records") or []:
            if isinstance(record, Mapping):
                records.append(dict(record))
    return records


def _cache_window_end_for_extraction(
    *,
    state: Mapping[str, Any],
    path: str,
    columns: list[str],
    record_anchor: str | None,
) -> int | None:
    cache_key = _extraction_cache_key(
        path=path,
        columns=columns,
        record_anchor=record_anchor,
    )
    entries = state.get("narrative_extraction_cache")
    if not isinstance(entries, list):
        return None
    ends = [
        int(entry["observed_line_end"])
        for entry in entries
        if isinstance(entry, Mapping)
        and str(entry.get("cache_key") or "") == cache_key
        and isinstance(entry.get("observed_line_end"), int)
    ]
    if not ends:
        return None
    return max(ends)


def _max_cached_line_number(records: list[dict[str, Any]]) -> int | None:
    line_numbers = [
        int(record["line_number"])
        for record in records
        if isinstance(record.get("line_number"), int)
    ]
    if not line_numbers:
        return None
    return max(line_numbers)


def _merge_extraction_records(
    *,
    columns: list[str],
    current_records: list[dict[str, Any]],
    cached_records: list[dict[str, Any]],
    max_records: int,
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    current_keys = [
        str(record.get("record_key") or "").strip()
        for record in current_records
        if str(record.get("record_key") or "").strip()
    ]
    if not current_keys:
        return [], []
    current_key_set = set(current_keys)
    merged: dict[str, dict[str, Any]] = {}
    for record in [
        *[
            record
            for record in cached_records
            if str(record.get("record_key") or "").strip() in current_key_set
        ],
        *current_records,
    ]:
        record_key = str(record.get("record_key") or "").strip()
        if not record_key:
            continue
        target = merged.setdefault(
            record_key,
            {
                "record_key": record_key,
                "values": {field: "" for field in columns},
                "matched_fields": [],
            },
        )
        values = target.setdefault("values", {})
        if not isinstance(values, dict):
            values = {field: "" for field in columns}
            target["values"] = values
        for field, value in (record.get("values") or {}).items():
            field_name = str(field)
            if field_name not in columns:
                continue
            if _value_is_empty(values.get(field_name)) and not _value_is_empty(value):
                values[field_name] = value
        matched_fields = set(target.get("matched_fields") or [])
        matched_fields.update(str(field) for field in record.get("matched_fields") or [])
        matched_fields.update(
            field for field in columns if not _value_is_empty(values.get(field))
        )
        target["matched_fields"] = sorted(matched_fields)
        normalized = target.setdefault("normalized_values", {})
        if not isinstance(normalized, dict):
            normalized = {}
            target["normalized_values"] = normalized
        for field, value in (record.get("normalized_values") or {}).items():
            if field in columns and field not in normalized:
                normalized[field] = value
        line_number = record.get("line_number")
        if isinstance(line_number, int):
            existing_line = target.get("line_number")
            target["line_number"] = (
                min(existing_line, line_number)
                if isinstance(existing_line, int)
                else line_number
            )
        content = str(record.get("content") or "").strip()
        if content and not target.get("content"):
            target["content"] = content[:1200]

    rows: list[list[Any]] = []
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record_key in current_keys:
        if record_key in seen:
            continue
        seen.add(record_key)
        record = merged.get(record_key)
        if not record:
            continue
        values = record.get("values")
        if not isinstance(values, Mapping):
            continue
        row = [values.get(field, "") for field in columns]
        if not any(not _value_is_empty(value) for value in row):
            continue
        rows.append(row)
        evidence.append(
            {
                "line_number": record.get("line_number"),
                "record_key": record_key,
                "values": {field: row[index] for index, field in enumerate(columns)},
                "normalized_values": record.get("normalized_values") or {},
                "matched_fields": record.get("matched_fields") or [],
                "content": str(record.get("content") or "")[:1200],
            }
        )
        if len(rows) >= max_records:
            break
    return rows, evidence


def _extraction_cache_entry(
    *,
    path: str,
    columns: list[str],
    record_anchor: str | None,
    rows: list[list[Any]],
    line_evidence: list[dict[str, Any]],
    observed_line_start: int | None = None,
    observed_line_end: int | None = None,
) -> dict[str, Any]:
    records = _extraction_records_from_rows(
        columns=columns,
        rows=rows,
        line_evidence=line_evidence,
    )
    return {
        "cache_key": _extraction_cache_key(
            path=path,
            columns=columns,
            record_anchor=record_anchor,
        ),
        "path": path,
        "columns": columns,
        "record_anchor": record_anchor,
        "records": records,
        **(
            {"observed_line_start": observed_line_start}
            if observed_line_start is not None
            else {}
        ),
        **(
            {"observed_line_end": observed_line_end}
            if observed_line_end is not None
            else {}
        ),
    }


def _stats_for_multi_field_rows(
    *,
    columns: list[str],
    rows: list[list[Any]],
    record_anchor: str | None,
) -> dict[str, Any]:
    return {
        "fields": columns,
        "missing_counts": {
            field: sum(
                1
                for row in rows
                if field_index >= len(row) or _value_is_empty(row[field_index])
            )
            for field_index, field in enumerate(columns)
        },
        "record_anchor": record_anchor,
    }


def _record_segments(
    lines: list[str],
    *,
    start_index: int,
    end_index: int,
    record_anchor: str | None,
) -> list[tuple[str, int, list[str]]]:
    segments: list[tuple[str, int, list[str]]] = []
    current_key: str | None = None
    current_key_is_synthetic = False
    current_start = start_index
    current_lines: list[str] = []
    for index in range(start_index, end_index):
        line = lines[index]
        key = _record_key_from_line(line)
        if key is None and record_anchor and _line_matches_record_anchor(line, record_anchor):
            if current_key is None or current_key_is_synthetic:
                key = f"line:{index + 1}"
        if key is not None:
            if current_key is not None and current_lines:
                segments.append((current_key, current_start, current_lines))
            current_key = key
            current_key_is_synthetic = key.startswith("line:")
            current_start = index
            current_lines = [line]
            continue
        if current_key is not None:
            current_lines.append(line)
    if current_key is not None and current_lines:
        segments.append((current_key, current_start, current_lines))
    return segments


def _extract_multi_field_rows(
    lines: list[str],
    *,
    source_fields: list[str],
    field_aliases: Mapping[str, Any] | None,
    record_anchor: str | None,
    start_line: int | None,
    end_line: int | None,
    max_records: int,
) -> tuple[list[list[Any]], list[dict[str, Any]], dict[str, Any]]:
    start_index = max(0, start_line - 1) if start_line is not None else 0
    end_index = min(len(lines), end_line) if end_line is not None else len(lines)
    aliases_by_field = {
        field: _field_aliases(field, field_aliases)
        for field in source_fields
    }
    segments = _record_segments(
        lines,
        start_index=start_index,
        end_index=end_index,
        record_anchor=record_anchor,
    )
    if segments:
        values_by_record: dict[str, list[Any]] = {}
        normalized_by_record: dict[str, dict[str, Any]] = {}
        evidence_by_record: dict[str, dict[str, Any]] = {}
        missing_counts = {field: 0 for field in source_fields}
        for record_key, segment_start, segment_lines in segments:
            context_line = " ".join(segment_lines)
            values = values_by_record.setdefault(record_key, [""] * len(source_fields))
            normalized_values = normalized_by_record.setdefault(record_key, {})
            matched_fields: list[str] = []
            for field_index, field in enumerate(source_fields):
                if values[field_index] != "":
                    continue
                extracted = _extract_multi_field_value(
                    context_line,
                    source_field=field,
                    aliases=aliases_by_field[field],
                )
                if extracted is None:
                    continue
                raw_value, normalized_value = extracted
                values[field_index] = raw_value
                normalized_values[field] = normalized_value
                matched_fields.append(field)
            if not matched_fields:
                continue
            existing = evidence_by_record.get(record_key)
            evidence_by_record[record_key] = {
                "line_number": (
                    min(existing["line_number"], segment_start + 1)
                    if isinstance(existing, Mapping)
                    and isinstance(existing.get("line_number"), int)
                    else segment_start + 1
                ),
                "record_key": record_key,
                "values": {
                    field: values[field_index]
                    for field_index, field in enumerate(source_fields)
                },
                "normalized_values": dict(normalized_values),
                "matched_fields": sorted(
                    set(
                        [
                            *(
                                existing.get("matched_fields", [])
                                if isinstance(existing, Mapping)
                                else []
                            ),
                            *matched_fields,
                        ]
                    )
                ),
                "content": (
                    f"{existing.get('content', '')} {context_line}".strip()[:1200]
                    if isinstance(existing, Mapping)
                    else context_line[:1200]
                ),
            }
        rows: list[list[Any]] = []
        evidence: list[dict[str, Any]] = []
        for record_key, values in values_by_record.items():
            if not any(value != "" for value in values):
                continue
            for field_index, field in enumerate(source_fields):
                if values[field_index] == "":
                    missing_counts[field] += 1
            rows.append(values)
            if record_key in evidence_by_record:
                evidence.append(evidence_by_record[record_key])
            if len(rows) >= max_records:
                break
        stats = {
            "fields": source_fields,
            "missing_counts": missing_counts,
            "record_anchor": record_anchor,
        }
        return rows, evidence, stats

    rows: list[list[Any]] = []
    evidence: list[dict[str, Any]] = []
    missing_counts = {field: 0 for field in source_fields}
    for index in range(start_index, end_index):
        line = lines[index]
        context_line = " ".join(lines[index : min(len(lines), index + 2)])
        if record_anchor and not _line_matches_record_anchor(line, record_anchor):
            continue
        values: list[Any] = []
        normalized_values: dict[str, Any] = {}
        matched_any = False
        for field in source_fields:
            extracted = _extract_multi_field_value(
                line,
                source_field=field,
                aliases=aliases_by_field[field],
            )
            if extracted is None and context_line != line:
                extracted = _extract_multi_field_value(
                    context_line,
                    source_field=field,
                    aliases=aliases_by_field[field],
                )
            if extracted is None:
                values.append("")
                missing_counts[field] += 1
                continue
            raw_value, normalized_value = extracted
            values.append(raw_value)
            if raw_value == "":
                missing_counts[field] += 1
            normalized_values[field] = normalized_value
            matched_any = True
        if not matched_any:
            continue
        rows.append(values)
        evidence.append(
            {
                "line_number": index + 1,
                "values": {
                    field: values[field_index]
                    for field_index, field in enumerate(source_fields)
                },
                "normalized_values": normalized_values,
                "content": context_line,
            }
        )
        if len(rows) >= max_records:
            break
    stats = {
        "fields": source_fields,
        "missing_counts": missing_counts,
        "record_anchor": record_anchor,
    }
    return rows, evidence, stats


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            return [str(item).strip() for item in decoded if str(item).strip()]
    return [item.strip() for item in re.split(r"[,;|]", text) if item.strip()]


def _coerce_field_aliases(value: Any) -> dict[str, list[str]] | None:
    if value is None or isinstance(value, Mapping):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, Mapping):
        return None
    aliases: dict[str, list[str]] = {}
    for key, item in decoded.items():
        if isinstance(item, list):
            aliases[str(key)] = [str(alias) for alias in item if str(alias).strip()]
        elif str(item).strip():
            aliases[str(key)] = [str(item)]
    return aliases


def _plan_operation_items(analysis_plan: Mapping[str, Any]) -> list[Any]:
    operations: list[Any] = []
    output_spec = analysis_plan.get("output_spec")
    if isinstance(output_spec, Mapping):
        transformations = output_spec.get("transformations")
        if isinstance(transformations, list):
            operations.extend(
                dict(item)
                for item in transformations
                if isinstance(item, Mapping)
            )
    execution_spec = analysis_plan.get("execution_spec")
    if isinstance(execution_spec, Mapping):
        execution_operations = execution_spec.get("operations")
        if isinstance(execution_operations, list):
            operations.extend(
                dict(item)
                for item in execution_operations
                if isinstance(item, Mapping)
            )
    return operations


def create_extract_narrative_records_tool(
    workspace: Path,
    config: Any,
) -> BaseTool:
    """Create a source-bound narrative record extractor."""

    context_root = (workspace / "context").resolve()

    @tool("extract_narrative_records", description=load_tool_prompt("extract_narrative_records"))
    def extract_narrative_records(
        source_path: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict[str, Any], InjectedState],
        source_field: str | None = None,
        source_fields: list[str] | None = None,
        record_anchor: str | None = None,
        field_aliases: dict[str, list[str]] | None = None,
        column: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        max_records: int = 500,
    ) -> Command[BenchmarkDeepAgentState] | ToolMessage:
        """Extract a source-bound metric column from a narrative document."""

        resolved, path_error = resolve_context_path(
            context_root,
            source_path,
            allowed_suffixes=DOC_SUFFIXES,
        )
        if path_error:
            return error(
                name="extract_narrative_records",
                tool_call_id=tool_call_id,
                message=path_error,
                max_output_bytes=config.max_output_bytes,
            )
        if max_records < 1:
            return error(
                name="extract_narrative_records",
                tool_call_id=tool_call_id,
                message="max_records must be >= 1.",
                max_output_bytes=config.max_output_bytes,
            )

        assert resolved is not None
        virtual_source = virtual_path(resolved, context_root)
        coerced_source_fields = _coerce_string_list(source_fields)
        coerced_field_aliases = _coerce_field_aliases(field_aliases)
        requested_fields = [
            str(field).strip()
            for field in (
                coerced_source_fields
                or ([source_field] if source_field else [])
            )
            if str(field).strip()
        ]
        if not requested_fields:
            return error(
                name="extract_narrative_records",
                tool_call_id=tool_call_id,
                message="source_field or source_fields must be provided.",
                max_output_bytes=config.max_output_bytes,
            )
        analysis_plan = state.get("analysis_plan")
        if isinstance(analysis_plan, Mapping):
            for requested_field in requested_fields:
                bound_paths = _source_binding_paths(
                    analysis_plan=analysis_plan,
                    source_field=requested_field,
                )
                if bound_paths and virtual_source not in bound_paths:
                    return error(
                        name="extract_narrative_records",
                        tool_call_id=tool_call_id,
                        message=(
                            "source_path must satisfy the active source binding for "
                            f"{requested_field}: {sorted(bound_paths)}."
                        ),
                        max_output_bytes=config.max_output_bytes,
                    )
        else:
            analysis_plan = {}

        columns_for_cache = [str(field) for field in requested_fields]
        cached_records_before = (
            _cache_entries_for_extraction(
                state=state,
                path=virtual_source,
                columns=columns_for_cache,
                record_anchor=record_anchor if len(columns_for_cache) > 1 else None,
            )
            if len(columns_for_cache) > 1
            else []
        )
        effective_start_line = start_line
        window_adjustment: dict[str, Any] | None = None
        cached_window_end = (
            _cache_window_end_for_extraction(
                state=state,
                path=virtual_source,
                columns=columns_for_cache,
                record_anchor=record_anchor if len(columns_for_cache) > 1 else None,
            )
            if len(columns_for_cache) > 1
            else None
        )
        cached_max_line = cached_window_end or _max_cached_line_number(
            cached_records_before
        )
        if (
            cached_max_line is not None
            and isinstance(start_line, int)
            and end_line is None
            and start_line > cached_max_line + 1
        ):
            effective_start_line = cached_max_line + 1
            window_adjustment = {
                "reason": "filled_gap_after_cached_extraction_window",
                "requested_start_line": start_line,
                "effective_start_line": effective_start_line,
                "cached_max_line": cached_max_line,
            }

        try:
            if resolved.suffix.lower() == ".pdf":
                text = extract_pdf_text(resolved)
            else:
                text = resolved.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            knowledge_quote = _knowledge_quote_for_field(
                analysis_plan=analysis_plan,
                source_field=requested_fields[0],
            )
            if not knowledge_quote:
                knowledge_path = context_root / "knowledge.md"
                if knowledge_path.exists():
                    for fact in parse_knowledge_content(
                        knowledge_path.read_text(encoding="utf-8", errors="replace")
                    ):
                        if str(fact.field_key or "").casefold() == requested_fields[0].casefold():
                            knowledge_quote = fact.quote
                            break
            extraction_stats: dict[str, Any] = {}
            if len(requested_fields) == 1 and not coerced_source_fields:
                rows, line_evidence = _extract_rows(
                    lines,
                    source_field=requested_fields[0],
                    knowledge_quote=knowledge_quote,
                    start_line=effective_start_line,
                    end_line=end_line,
                    max_records=max_records,
                )
            else:
                rows, line_evidence, extraction_stats = _extract_multi_field_rows(
                    lines,
                    source_fields=requested_fields,
                    field_aliases=coerced_field_aliases,
                    record_anchor=record_anchor,
                    start_line=effective_start_line,
                    end_line=end_line,
                    max_records=max_records,
                )
        except Exception as exc:
            return error(
                name="extract_narrative_records",
                tool_call_id=tool_call_id,
                message=f"Failed to extract narrative records: {exc}",
                max_output_bytes=config.max_output_bytes,
            )

        if len(requested_fields) == 1 and not coerced_source_fields:
            observed_field_name = _observed_field_name(lines, requested_fields[0])
            columns = [str(column or observed_field_name or requested_fields[0])]
        else:
            columns = [str(field) for field in requested_fields]
            cached_records = _cache_entries_for_extraction(
                state=state,
                path=virtual_source,
                columns=columns,
                record_anchor=record_anchor,
            )
            if cached_records:
                current_records = _extraction_records_from_rows(
                    columns=columns,
                    rows=rows,
                    line_evidence=line_evidence,
                )
                merged_rows, merged_evidence = _merge_extraction_records(
                    columns=columns,
                    current_records=current_records,
                    cached_records=cached_records,
                    max_records=max_records,
                )
                if merged_rows:
                    rows = merged_rows
                    line_evidence = merged_evidence
                    extraction_stats = _stats_for_multi_field_rows(
                        columns=columns,
                        rows=rows,
                        record_anchor=record_anchor,
                    )
        cache_update = [
            _extraction_cache_entry(
                path=virtual_source,
                columns=columns,
                record_anchor=record_anchor if len(columns) > 1 else None,
                rows=rows,
                line_evidence=line_evidence,
                observed_line_start=effective_start_line,
                observed_line_end=(
                    end_line
                    if isinstance(end_line, int)
                    else (
                        len(lines)
                        if end_line is None
                        else None
                    )
                ),
            )
        ]
        field_non_empty_counts = {
            field: sum(
                1
                for row in rows
                if isinstance(row, list)
                and field_index < len(row)
                and not _value_is_empty(row[field_index])
            )
            for field_index, field in enumerate(columns)
        }
        observed_field_names = [
            field
            for field in columns
            if field_non_empty_counts.get(field, 0) > 0
        ]
        field_line_counts = {
            field: sum(
                1
                for evidence in line_evidence
                if isinstance(evidence, Mapping)
                and (
                    field in (evidence.get("matched_fields") or [])
                    or (
                        isinstance(evidence.get("values"), Mapping)
                        and not _value_is_empty(evidence["values"].get(field))
                    )
                )
            )
            for field in columns
        }
        incomplete_fields = [
            field
            for field in columns
            if field_non_empty_counts.get(field, 0) == 0
        ]
        candidate_is_complete = bool(rows) and not incomplete_fields
        plan_operations = (
            _plan_operation_items(analysis_plan)
            if isinstance(analysis_plan, Mapping)
            else []
        )
        audit_operations = plan_operations or [
            {
                "operation": "extract_narrative_records",
                "description": (
                    "Extract source-bound narrative records without additional "
                    "semantic transformation."
                ),
            }
        ]
        audit = {
            "source_paths": [virtual_source],
            "operations": audit_operations,
            "tool_actions": [
                {
                    "tool": "extract_narrative_records",
                    "fields": requested_fields,
                }
            ],
            "output_row_count": len(rows),
            "output_hash": answer_value_hash(columns, rows),
            "audit_origin": "extract_narrative_records",
        }

        message_payload = {
            "status": "extracted",
            "path": virtual_source,
            "column": columns[0] if len(columns) == 1 else None,
            "columns": columns,
            "row_count": len(rows),
            "non_empty_count": sum(
                1 for row in rows if row and any(value != "" for value in row)
            ),
            "field_non_empty_counts": field_non_empty_counts,
            "line_evidence": line_evidence[:20],
        }
        if window_adjustment is not None:
            message_payload["window_adjustment"] = window_adjustment
        if extraction_stats:
            message_payload["extraction_stats"] = extraction_stats
        if incomplete_fields:
            message_payload["incomplete_fields"] = incomplete_fields
            message_payload["recommended_next_actions"] = [
                {
                    "tool": "read_doc",
                    "reason": (
                        "inspect adjacent or later slices before forming an "
                        "execution plan"
                    ),
                },
                {
                    "tool": "extract_narrative_records",
                    "reason": (
                        "retry with an expanded or later line window; omit "
                        "end_line when a field appears outside the current slice"
                    ),
                },
            ]

        prepared_answer = None
        answer_error = "extract_narrative_records requires a successful analysis_plan first."
        has_analysis_plan = isinstance(analysis_plan, dict) and bool(analysis_plan)
        if has_analysis_plan:
            prepared_answer, answer_error = validate_prepared_answer(
                columns,
                rows,
                analysis_plan,
                audit,
            )
        observed_sources = merge_observed_sources(
            state.get("observed_sources"),
            [
                {
                    "path": virtual_source,
                    "source_type": "doc",
                    "source_name_hint": resolved.stem,
                    "line_count": len(lines),
                    "fields": observed_field_names,
                    "extracted_fields": requested_fields,
                    "field_evidence": [
                        {
                            "field": field,
                            "evidence_type": "narrative_extraction",
                            "line_evidence_count": field_line_counts.get(field, 0),
                        }
                        for field in requested_fields
                    ],
                    "matched_lines": line_evidence[:20],
                    "sample_hash": sample_hash(rows[:50]),
                    "observed_by": "extract_narrative_records",
                }
            ],
        )
        if not candidate_is_complete:
            message_payload["status"] = "extracted_incomplete"
            message_payload["validation_error"] = (
                "extracted rows have no non-empty values for requested fields: "
                f"{incomplete_fields}."
            )
            return Command(
                update={
                    "observed_sources": observed_sources,
                    "narrative_extraction_cache": cache_update,
                    "messages": [
                        ToolMessage(
                            content=json.dumps(message_payload, ensure_ascii=False),
                            name="extract_narrative_records",
                            tool_call_id=tool_call_id,
                            status="error" if has_analysis_plan else "success",
                        )
                    ],
                }
            )
        if prepared_answer is None:
            candidate = _candidate_payload(
                columns=columns,
                rows=rows,
                audit=audit,
                validation_error=answer_error,
            )
            message_payload["validation_error"] = answer_error
            message_payload["status"] = (
                "candidate_saved"
                if has_analysis_plan
                else "extracted_pending_analysis_plan"
            )
            return Command(
                update={
                    "observed_sources": observed_sources,
                    "narrative_extraction_cache": cache_update,
                    "answer_candidate": candidate,
                    "messages": [
                        ToolMessage(
                            content=json.dumps(message_payload, ensure_ascii=False),
                            name="extract_narrative_records",
                            tool_call_id=tool_call_id,
                            status="error" if has_analysis_plan else "success",
                        )
                    ],
                }
            )

        return Command(
            update={
                "observed_sources": observed_sources,
                "narrative_extraction_cache": cache_update,
                "prepared_answer": prepared_answer,
                "answer_candidate": None,
                "messages": [
                    ToolMessage(
                        content=json.dumps(message_payload, ensure_ascii=False),
                        name="extract_narrative_records",
                        tool_call_id=tool_call_id,
                        status="success",
                    )
                ],
            }
        )

    return extract_narrative_records
