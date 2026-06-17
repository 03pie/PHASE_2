from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.agents.semantic_layer import query_semantic_context
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools._helpers import error, query_context_schema, success
from data_agent_baseline.tools.observed_sources import (
    observed_sources_command,
    sample_hash,
)


def _semantic_observed_sources(
    semantic_matches: dict[str, Any],
    *,
    field_text: str,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for candidate in semantic_matches.get("source_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("source_type") or "") != "doc":
            continue
        path = str(candidate.get("source_path") or "").replace("\\", "/")
        line_evidence = candidate.get("line_evidence")
        if not path or not isinstance(line_evidence, list) or not line_evidence:
            continue
        matched_lines = [
            {
                "line_number": item.get("line_number"),
                "content": item.get("content"),
                **(
                    {"score": item.get("score")}
                    if isinstance(item.get("score"), int)
                    else {}
                ),
            }
            for item in line_evidence
            if isinstance(item, dict)
        ][:10]
        sources.append(
            {
                "path": path,
                "source_type": "doc",
                "logical_name": candidate.get("logical_name"),
                "semantic_query": field_text,
                "semantic_confidence": candidate.get("confidence"),
                "matched_lines": matched_lines,
                "sample_hash": sample_hash(matched_lines),
                "observed_by": "query_schema",
            }
        )
    return sources


def create_query_schema_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """Create a lightweight field lookup tool across context data sources."""

    context_root = (workspace / "context").resolve()

    @tool("query_schema", description=load_tool_prompt("query_schema"))
    def query_schema(
        field: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict[str, Any], InjectedState],
        max_matches: int = 25,
    ) -> Any:
        """Run the query_schema tool."""

        field_text = field.strip()
        if not field_text:
            return error(
                name="query_schema",
                tool_call_id=tool_call_id,
                message="field must be non-empty.",
                max_output_bytes=config.max_output_bytes,
            )
        if max_matches < 1:
            return error(
                name="query_schema",
                tool_call_id=tool_call_id,
                message="max_matches must be positive.",
                max_output_bytes=config.max_output_bytes,
            )

        matches = query_context_schema(
            context_root,
            field_text,
            max_matches=max_matches,
        )
        semantic_matches = query_semantic_context(
            context_root,
            field_text,
            max_matches=max_matches,
        )
        message = success(
            name="query_schema",
            tool_call_id=tool_call_id,
            payload={
                "field": field_text,
                "matches": matches,
                "match_count": len(matches),
                "source_candidates": semantic_matches["source_candidates"],
                "logical_bindings": semantic_matches["logical_bindings"],
                "binding_issues": semantic_matches["binding_issues"],
                "knowledge_facts": semantic_matches["knowledge_facts"],
                "hint": "Inspect the reported source before relying on a field.",
            },
            max_output_bytes=config.max_output_bytes,
        )
        sources = _semantic_observed_sources(
            semantic_matches,
            field_text=field_text,
        )
        if not sources:
            return message
        return observed_sources_command(
            state=state,
            message=message,
            sources=sources,
        )

    return query_schema
