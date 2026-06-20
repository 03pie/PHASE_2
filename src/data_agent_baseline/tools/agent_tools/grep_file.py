from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools._helpers import (
    apply_head_limit,
    collect_search_files,
    error,
    fuzzy_search,
    pattern_to_keywords,
    regex_search,
    resolve_context_path,
    success,
)
from data_agent_baseline.tools.observed_sources import (
    observed_sources_command,
    sample_hash,
)

_MAX_CONTENT_MATCHES_PER_PAGE = 50


def _source_type_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix in {".log", ".md", ".pdf", ".txt"}:
        return "doc"
    return "text"


def _grep_observed_sources(
    matches: list[dict[str, Any]],
    *,
    pattern: str,
    search_mode: str,
) -> list[dict[str, Any]]:
    by_file: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for match in matches:
        if match.get("is_separator"):
            continue
        file_path = str(match.get("file") or "").replace("\\", "/")
        if not file_path:
            continue
        counts[file_path] = counts.get(file_path, 0) + 1
        if len(by_file.setdefault(file_path, [])) >= 10:
            continue
        by_file[file_path].append(
            {
                "line_number": match.get("line_number"),
                "content": match.get("content"),
                **({"score": match.get("score")} if "score" in match else {}),
                **({"is_match": True} if match.get("is_match") else {}),
            }
        )
    return [
        {
            "path": file_path,
            "source_type": _source_type_for_path(file_path),
            "search_pattern": pattern,
            "search_mode": search_mode,
            "match_count": counts.get(file_path, len(matched_lines)),
            "matched_lines": matched_lines,
            "sample_hash": sample_hash(matched_lines),
            "observed_by": "grep_file",
        }
        for file_path, matched_lines in by_file.items()
        if matched_lines
    ]


def _read_doc_slices_for_matches(
    matches: list[dict[str, Any]],
    *,
    context_lines: int,
    max_slices: int = 10,
) -> list[dict[str, Any]]:
    slices: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    lead_lines = max(12, context_lines)
    window_lines = max(40, min(200, lead_lines * 2 + 40))
    for match in matches:
        if match.get("is_separator"):
            continue
        file_path = str(match.get("file") or "").replace("\\", "/")
        line_number = match.get("line_number")
        if not file_path or not isinstance(line_number, int):
            continue
        start_line = max(0, line_number - 1 - lead_lines)
        key = (file_path, start_line)
        if key in seen:
            continue
        seen.add(key)
        slices.append(
            {
                "path": file_path,
                "anchor_line": line_number,
                "start_line": start_line,
                "max_lines": window_lines,
            }
        )
        if len(slices) >= max_slices:
            break
    return slices


def _paging_payload(
    *,
    total_items: int,
    returned_items: int,
    offset: int,
    paging_unit: str,
) -> dict[str, Any]:
    next_offset = offset + returned_items
    if next_offset >= total_items:
        next_offset = None
    return {
        "paging_unit": paging_unit,
        "total_result_items": total_items,
        "offset": offset,
        "next_offset": next_offset,
        "has_more": next_offset is not None,
    }


def create_grep_file_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """Create an EVE-style grep_file search tool."""

    context_root = (workspace / "context").resolve()

    @tool("grep_file", description=load_tool_prompt("grep_file"))
    def grep_file(
        pattern: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict[str, Any], InjectedState],
        path: str = ".",
        include: str = "",
        output_mode: Literal["content", "files_with_matches"] = "files_with_matches",
        context_lines: int = 0,
        max_matches: int = 200,
        offset: int = 0,
    ) -> Any:
        """Run the grep_file tool."""

        if not pattern.strip():
            return error(
                name="grep_file",
                tool_call_id=tool_call_id,
                message="pattern is required.",
                max_output_bytes=config.max_output_bytes,
            )
        if max_matches < 0 or offset < 0 or context_lines < 0:
            return error(
                name="grep_file",
                tool_call_id=tool_call_id,
                message="max_matches, offset, and context_lines must be non-negative.",
                max_output_bytes=config.max_output_bytes,
            )

        resolved, path_error = resolve_context_path(
            context_root,
            path,
            allow_directory=True,
        )
        if path_error:
            return error(
                name="grep_file",
                tool_call_id=tool_call_id,
                message=path_error,
                max_output_bytes=config.max_output_bytes,
            )
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return error(
                name="grep_file",
                tool_call_id=tool_call_id,
                message=f"Invalid regex: {exc}",
                max_output_bytes=config.max_output_bytes,
            )

        assert resolved is not None
        files = collect_search_files(resolved, context_root=context_root, include=include)
        raw_matches = regex_search(files, compiled, context_lines=context_lines)
        search_mode = "regex"
        fuzzy_query = None
        if not raw_matches:
            fuzzy_query = pattern_to_keywords(pattern)
            if fuzzy_query:
                raw_matches = fuzzy_search(files, fuzzy_query, threshold=0.55)
                if raw_matches:
                    search_mode = "auto_fuzzy"

        payload: dict[str, Any]
        if output_mode == "files_with_matches":
            file_counts: dict[str, int] = {}
            for match in raw_matches:
                if not match.get("is_separator"):
                    file_name = str(match["file"])
                    file_counts[file_name] = file_counts.get(file_name, 0) + 1
            filenames = [
                filename
                for filename, _ in sorted(
                    file_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ]
            limited, applied_limit = apply_head_limit(filenames, max_matches, offset)
            payload = {
                "mode": "files_with_matches",
                "filenames": limited,
                "numFiles": len(limited),
                "files_searched": len(files),
                "search_mode": search_mode,
                "total_matches": sum(file_counts.values()),
                **_paging_payload(
                    total_items=len(filenames),
                    returned_items=len(limited),
                    offset=offset,
                    paging_unit="files",
                ),
            }
        else:
            effective_max_matches = min(max_matches, _MAX_CONTENT_MATCHES_PER_PAGE)
            limited, applied_limit = apply_head_limit(
                raw_matches,
                effective_max_matches,
                offset,
            )
            total_matches = sum(
                1 for match in raw_matches if not match.get("is_separator")
            )
            payload = {
                "mode": "content",
                "content": "\n".join(
                    f"{match['file']}:{match['line_number']}:{match['content']}"
                    for match in limited
                ),
                "numLines": len(limited),
                "filenames": list(
                    dict.fromkeys(str(match["file"]) for match in limited)
                ),
                "files_searched": len(files),
                "search_mode": search_mode,
                "total_matches": total_matches,
                **_paging_payload(
                    total_items=len(raw_matches),
                    returned_items=len(limited),
                    offset=offset,
                    paging_unit="matched_lines",
                ),
            }
            if max_matches > effective_max_matches:
                payload["requested_max_matches"] = max_matches
                payload["max_matches"] = effective_max_matches
                payload["match_page_capped"] = True
        read_doc_slices = _read_doc_slices_for_matches(
            raw_matches,
            context_lines=context_lines,
        )
        if read_doc_slices:
            payload["read_doc_slices"] = read_doc_slices
        if applied_limit is not None:
            payload["appliedLimit"] = applied_limit
        if offset:
            payload["appliedOffset"] = offset
        if fuzzy_query is not None:
            payload["fuzzy_query"] = fuzzy_query
        if not raw_matches:
            payload["hint"] = "No matches found. Try a broader pattern."

        message = success(
            name="grep_file",
            tool_call_id=tool_call_id,
            payload=payload,
            max_output_bytes=config.max_output_bytes,
        )
        sources = _grep_observed_sources(
            raw_matches,
            pattern=pattern,
            search_mode=search_mode,
        )
        if not sources:
            return message
        return observed_sources_command(
            state=state,
            message=message,
            sources=sources,
        )

    return grep_file
