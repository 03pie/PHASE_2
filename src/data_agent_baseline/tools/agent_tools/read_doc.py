from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools._helpers import (
    DOC_SUFFIXES,
    error,
    extract_pdf_text,
    resolve_context_path,
    success,
    virtual_path,
)

def create_read_doc_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """Create a text/PDF document reader that returns text-only tool output."""

    context_root = (workspace / "context").resolve()

    @tool("read_doc", description=load_tool_prompt("read_doc"))
    def read_doc(
        path: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
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
            if resolved.suffix.lower() == ".pdf":
                text = extract_pdf_text(resolved)
            else:
                text = resolved.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            slice_lines = lines[start_line : start_line + max_lines]
            numbered = [
                f"{line_number:>6}->{line}"
                for line_number, line in enumerate(slice_lines, start=start_line + 1)
            ]
            headings = [
                {"line": index + 1, "text": line.strip()}
                for index, line in enumerate(lines)
                if line.startswith("#")
            ][:30]
            return success(
                name="read_doc",
                tool_call_id=tool_call_id,
                payload={
                    "path": virtual_path(resolved, context_root),
                    "content": "\n".join(numbered),
                    "total_lines": len(lines),
                    "returned_lines": len(slice_lines),
                    "start_line": start_line + 1,
                    "has_more": start_line + len(slice_lines) < len(lines),
                    "truncated": start_line > 0
                    or start_line + len(slice_lines) < len(lines),
                    "headings": headings,
                },
                max_output_bytes=config.max_output_bytes,
            )
        except Exception as exc:
            return error(
                name="read_doc",
                tool_call_id=tool_call_id,
                message=f"Failed to read document: {exc}",
                max_output_bytes=config.max_output_bytes,
            )

    return read_doc
