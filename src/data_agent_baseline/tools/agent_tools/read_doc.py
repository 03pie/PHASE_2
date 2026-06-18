from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.agents.semantic_layer import query_semantic_context
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools._helpers import (
    DOC_SUFFIXES,
    error,
    extract_pdf_text,
    resolve_context_path,
    success,
    virtual_path,
)
from data_agent_baseline.tools.observed_sources import (
    observed_sources_command,
    sample_hash,
)

_NARRATIVE_QUERY_HINT_PATTERN = re.compile(
    r"年|月|日|周|回报|收益|return|rate|rr",
    re.IGNORECASE,
)
_MAX_DOC_LINES_PER_READ = 240


def _read_strategy_payload(
    *,
    path: str,
    total_lines: int,
    returned_lines: int,
    start_line: int,
    effective_max_lines: int,
    next_start_line: int | None,
    previous_start_line: int | None,
) -> dict[str, Any]:
    large_file = total_lines > effective_max_lines
    actions: list[dict[str, Any]] = []
    if large_file:
        actions.append(
            {
                "tool": "grep_file",
                "reason": "locate relevant lines before reading more document slices",
                "args": {
                    "path": path,
                    "output_mode": "content",
                    "context_lines": 20,
                },
            }
        )
    if next_start_line is not None:
        actions.append(
            {
                "tool": "read_doc",
                "reason": "read the next adjacent slice",
                "args": {
                    "path": path,
                    "start_line": next_start_line,
                    "max_lines": effective_max_lines,
                },
            }
        )
    if previous_start_line is not None:
        actions.append(
            {
                "tool": "read_doc",
                "reason": "read the previous adjacent slice",
                "args": {
                    "path": path,
                    "start_line": previous_start_line,
                    "max_lines": effective_max_lines,
                },
            }
        )
    return {
        "large_file": large_file,
        "read_strategy": (
            "grep_file_to_line_anchors_then_read_doc_slices"
            if large_file
            else "single_slice_or_adjacent_paging"
        ),
        "recommended_next_actions": actions,
        "slice_window": {
            "start_line": start_line,
            "returned_lines": returned_lines,
            "max_lines": effective_max_lines,
        },
    }


def _collect_semantic_terms(value: Any, terms: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"quote", "value", "name", "description"}:
                text = str(item or "").strip()
                if 1 < len(text) <= 80 and text not in terms:
                    terms.append(text)
            _collect_semantic_terms(item, terms)
    elif isinstance(value, list):
        for item in value:
            _collect_semantic_terms(item, terms)


def _state_semantic_terms(state: Mapping[str, Any]) -> list[str]:
    terms: list[str] = []
    original_request = str(state.get("original_request") or "").strip()
    if original_request:
        terms.append(original_request)
    question_structure = state.get("question_structure")
    if isinstance(question_structure, Mapping):
        _collect_semantic_terms(question_structure, terms)
    filtered_terms = [
        term
        for term in terms
        if len(term) > 2 or _NARRATIVE_QUERY_HINT_PATTERN.search(term)
    ]
    filtered_terms.sort(
        key=lambda item: (
            0 if _NARRATIVE_QUERY_HINT_PATTERN.search(item) else 1,
            len(item),
            item,
        )
    )
    return filtered_terms[:8]


def _semantic_windows_for_doc(
    *,
    context_root: Path,
    source_path: str,
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for term in _state_semantic_terms(state):
        semantic = query_semantic_context(context_root, term, max_matches=10)
        for candidate in semantic.get("source_candidates") or []:
            if not isinstance(candidate, Mapping):
                continue
            if str(candidate.get("source_path") or "") != source_path:
                continue
            for line in candidate.get("line_evidence") or []:
                if not isinstance(line, Mapping):
                    continue
                line_number = line.get("line_number")
                if not isinstance(line_number, int):
                    continue
                key = (term, line_number)
                if key in seen:
                    continue
                seen.add(key)
                windows.append(
                    {
                        "query": term,
                        "line_number": line_number,
                        "content": line.get("content"),
                        **(
                            {"score": line.get("score")}
                            if isinstance(line.get("score"), int)
                            else {}
                        ),
                    }
                )
                if len(windows) >= 10:
                    return windows
    return windows


def create_read_doc_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """Create a text/PDF document reader that returns text-only tool output."""

    context_root = (workspace / "context").resolve()

    @tool("read_doc", description=load_tool_prompt("read_doc"))
    def read_doc(
        path: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict[str, Any], InjectedState],
        start_line: int = 0,
        max_lines: int = 120,
    ) -> Any:
        """Run the read_doc tool."""

        resolved, path_error = resolve_context_path(
            context_root,
            path,
            allowed_suffixes=DOC_SUFFIXES,
        )
        if path_error:
            return error(
                name="read_doc",
                tool_call_id=tool_call_id,
                message=path_error,
                max_output_bytes=config.max_output_bytes,
            )
        if start_line < 0 or max_lines < 1:
            return error(
                name="read_doc",
                tool_call_id=tool_call_id,
                message="start_line >= 0 and max_lines >= 1 required.",
                max_output_bytes=config.max_output_bytes,
            )

        try:
            assert resolved is not None
            effective_max_lines = min(max_lines, _MAX_DOC_LINES_PER_READ)
            if resolved.suffix.lower() == ".pdf":
                text = extract_pdf_text(resolved)
            else:
                text = resolved.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            slice_lines = lines[start_line : start_line + effective_max_lines]
            numbered = [
                f"{line_number:>6}->{line}"
                for line_number, line in enumerate(slice_lines, start=start_line + 1)
            ]
            headings = [
                {"line": index + 1, "text": line.strip()}
                for index, line in enumerate(lines)
                if line.startswith("#")
            ][:30]
            next_start_line = (
                start_line + len(slice_lines)
                if start_line + len(slice_lines) < len(lines)
                else None
            )
            previous_start_line = (
                max(0, start_line - effective_max_lines)
                if start_line > 0
                else None
            )
            payload = {
                "path": virtual_path(resolved, context_root),
                "content": "\n".join(numbered),
                "total_lines": len(lines),
                "returned_lines": len(slice_lines),
                "start_line": start_line + 1,
                "start_line_arg": start_line,
                "requested_max_lines": max_lines,
                "max_lines": effective_max_lines,
                "end_line": start_line + len(slice_lines),
                "next_start_line": next_start_line,
                "previous_start_line": previous_start_line,
                "has_more": start_line + len(slice_lines) < len(lines),
                "truncated": start_line > 0
                or start_line + len(slice_lines) < len(lines),
                "headings": headings,
                "line_numbering": (
                    "content prefixes are 1-based; start_line arguments are 0-based"
                ),
                **_read_strategy_payload(
                    path=virtual_path(resolved, context_root),
                    total_lines=len(lines),
                    returned_lines=len(slice_lines),
                    start_line=start_line,
                    effective_max_lines=effective_max_lines,
                    next_start_line=next_start_line,
                    previous_start_line=previous_start_line,
                ),
            }
            semantic_windows = _semantic_windows_for_doc(
                context_root=context_root,
                source_path=payload["path"],
                state=state,
            )
            if semantic_windows:
                payload["semantic_windows"] = semantic_windows
            message = success(
                name="read_doc",
                tool_call_id=tool_call_id,
                payload=payload,
                max_output_bytes=config.max_output_bytes,
            )
            return observed_sources_command(
                state=state,
                message=message,
                sources=[
                    {
                        "path": payload["path"],
                        "source_type": "doc",
                        "source_name_hint": resolved.stem,
                        "line_count": len(lines),
                        "observed_line_start": payload["start_line"],
                        "observed_line_count": payload["returned_lines"],
                        "truncated": payload["truncated"],
                        "semantic_query": (
                            semantic_windows[0]["query"] if semantic_windows else None
                        ),
                        "matched_lines": semantic_windows,
                        "headings": headings,
                        "sample_hash": sample_hash(slice_lines),
                        "observed_by": "read_doc",
                    }
                ],
            )
        except Exception as exc:
            return error(
                name="read_doc",
                tool_call_id=tool_call_id,
                message=f"Failed to read document: {exc}",
                max_output_bytes=config.max_output_bytes,
            )

    return read_doc
