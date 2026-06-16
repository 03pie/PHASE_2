from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.tools import BaseTool, InjectedToolCallId, tool

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

def create_grep_file_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """Create an EVE-style grep_file search tool."""

    context_root = (workspace / "context").resolve()

    @tool("grep_file", description=load_tool_prompt("grep_file"))
    def grep_file(
        pattern: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
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

        return success(
            name="grep_file",
            tool_call_id=tool_call_id,
            payload=payload,
            max_output_bytes=config.max_output_bytes,
        )

    return grep_file
