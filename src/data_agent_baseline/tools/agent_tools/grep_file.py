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


def _source_type_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix in {".log", ".md", ".txt"}:
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
            }
        else:
            limited, applied_limit = apply_head_limit(raw_matches, max_matches, offset)
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
            }
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
