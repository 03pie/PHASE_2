from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.tools._helpers import VIDEO_SUFFIXES, format_size, success, virtual_path


def create_list_videos_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """Create a video inventory tool for video-aware subagent workflows."""

    context_root = (workspace / "context").resolve()

    @tool("list_videos")
    def list_videos(
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Any:
        """List video files in context with basic metadata."""

        videos: list[dict[str, Any]] = []
        for path in sorted(context_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            stat = path.stat()
            videos.append(
                {
                    "path": virtual_path(path, context_root),
                    "size": stat.st_size,
                    "size_human": format_size(stat.st_size),
                }
            )
        return success(
            name="list_videos",
            tool_call_id=tool_call_id,
            payload={"videos": videos, "total": len(videos)},
            max_output_bytes=config.max_output_bytes,
        )

    return list_videos
