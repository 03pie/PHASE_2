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
_RECORD_RE = re.compile(
    r"(?:\u6863\u6848|\u6218\u7565\u5355\u5143|archive)\s*\d+",
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
            "\u65e0\u6cd5\u8bc4\u4f30",
            "missing",
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


def _extract_rows(
    lines: list[str],
    *,
    source_field: str,
    knowledge_quote: str,
    start_line: int | None,
    end_line: int | None,
    max_records: int,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    time_terms = _time_terms(source_field, knowledge_quote)
    if not time_terms:
        time_terms = _time_terms(knowledge_quote)
    start_index, end_index = _infer_window(
        lines,
        time_terms=time_terms,
        start_line=start_line,
        end_line=end_line,
    )
    rows: list[list[str]] = []
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


def create_extract_narrative_records_tool(
    workspace: Path,
    config: Any,
) -> BaseTool:
    """Create a source-bound narrative record extractor."""

    context_root = (workspace / "context").resolve()

    @tool("extract_narrative_records", description=load_tool_prompt("extract_narrative_records"))
    def extract_narrative_records(
        source_path: str,
        source_field: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict[str, Any], InjectedState],
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
        analysis_plan = state.get("analysis_plan")
        if isinstance(analysis_plan, Mapping):
            bound_paths = _source_binding_paths(
                analysis_plan=analysis_plan,
                source_field=source_field,
            )
            if bound_paths and virtual_source not in bound_paths:
                return error(
                    name="extract_narrative_records",
                    tool_call_id=tool_call_id,
                    message=(
                        "source_path must satisfy the active source binding for "
                        f"{source_field}: {sorted(bound_paths)}."
                    ),
                    max_output_bytes=config.max_output_bytes,
                )
        else:
            analysis_plan = {}

        try:
            if resolved.suffix.lower() == ".pdf":
                text = extract_pdf_text(resolved)
            else:
                text = resolved.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            knowledge_quote = _knowledge_quote_for_field(
                analysis_plan=analysis_plan,
                source_field=source_field,
            )
            if not knowledge_quote:
                knowledge_path = context_root / "knowledge.md"
                if knowledge_path.exists():
                    for fact in parse_knowledge_content(
                        knowledge_path.read_text(encoding="utf-8", errors="replace")
                    ):
                        if str(fact.logical_field or "").casefold() == source_field.casefold():
                            knowledge_quote = fact.quote
                            break
            rows, line_evidence = _extract_rows(
                lines,
                source_field=source_field,
                knowledge_quote=knowledge_quote,
                start_line=start_line,
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

        columns = [str(column or source_field)]
        audit = {
            "source_paths": [virtual_source],
            "operations": [
                {
                    "operation": "extract_narrative_records",
                    "description": (
                        f"Extract {source_field} from source-bound narrative records."
                    ),
                }
            ],
            "output_row_count": len(rows),
            "output_hash": answer_value_hash(columns, rows),
        }

        message_payload = {
            "status": "extracted",
            "path": virtual_source,
            "column": columns[0],
            "row_count": len(rows),
            "non_empty_count": sum(1 for row in rows if row and row[0] != ""),
            "line_evidence": line_evidence[:20],
        }

        prepared_answer = None
        answer_error = "extract_narrative_records requires a successful analysis_plan first."
        if isinstance(analysis_plan, dict) and analysis_plan:
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
                    "logical_name": resolved.stem,
                    "line_count": len(lines),
                    "matched_lines": line_evidence[:20],
                    "sample_hash": sample_hash(rows[:50]),
                    "observed_by": "extract_narrative_records",
                }
            ],
        )
        if prepared_answer is None:
            candidate = _candidate_payload(
                columns=columns,
                rows=rows,
                audit=audit,
                validation_error=answer_error,
            )
            message_payload["validation_error"] = answer_error
            message_payload["status"] = "candidate_saved"
            return Command(
                update={
                    "observed_sources": observed_sources,
                    "answer_candidate": candidate,
                    "messages": [
                        ToolMessage(
                            content=json.dumps(message_payload, ensure_ascii=False),
                            name="extract_narrative_records",
                            tool_call_id=tool_call_id,
                            status="error",
                        )
                    ],
                }
            )

        return Command(
            update={
                "observed_sources": observed_sources,
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
