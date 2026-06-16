from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools._helpers import (
    classify_file_role,
    format_size,
    success,
    virtual_path,
)

def create_list_context_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """Create a context inventory tool with EVE-style file role hints."""

    context_root = (workspace / "context").resolve()

    @tool("list_context", description=load_tool_prompt("list_context"))
    def list_context(
        tool_call_id: Annotated[str, InjectedToolCallId],
        max_files: int = 200,
    ) -> Any:
        """Run the list_context tool."""

        entries: list[dict[str, Any]] = []
        for path in sorted(context_root.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            rel = path.relative_to(context_root).as_posix()
            entries.append(
                {
                    "path": virtual_path(path, context_root),
                    "size": stat.st_size,
                    "size_human": format_size(stat.st_size),
                    "role": classify_file_role(rel, stat.st_size),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(
                        timespec="seconds"
                    ),
                }
            )

        max_files = max(1, max_files)
        return success(
            name="list_context",
            tool_call_id=tool_call_id,
            payload={
                "files": entries[:max_files],
                "total_files": len(entries),
                "truncated": len(entries) > max_files,
                "note": "The task prompt already contains the inventory.",
            },
            max_output_bytes=config.max_output_bytes,
        )

    return list_context
