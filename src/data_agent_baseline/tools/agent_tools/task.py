from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from data_agent_baseline.prompts.loader import load_tool_prompt

_TASK_PROMPT_ENTRY_DESCRIPTION_LIMIT = 160


def task_tool_description_overrides() -> dict[str, str]:
    """Return DeepAgents profile overrides for the injected task tool."""

    return {"task": load_tool_prompt("task")}


def _compact_text(value: Any, *, limit: int = _TASK_PROMPT_ENTRY_DESCRIPTION_LIMIT) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def format_task_prompt_entry(subagents: Sequence[Mapping[str, Any]]) -> str | None:
    """Render the task tool entry shown in the dynamic tool section."""

    if not subagents:
        return None
    agent_descriptions = []
    for subagent in subagents:
        name = str(subagent.get("name") or "").strip()
        description = _compact_text(subagent.get("description"))
        if name:
            agent_descriptions.append(
                f"`{name}`" + (f" - {description}" if description else "")
            )
    available = "; ".join(agent_descriptions) if agent_descriptions else "see tool schema"
    return load_tool_prompt("task_prompt_entry").format(available_agents=available)
