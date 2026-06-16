from __future__ import annotations

from typing import Any

from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware

from data_agent_baseline.prompts.loader import load_tool_prompt

WRITE_TODOS_EXCLUDED_MIDDLEWARE = frozenset({TodoListMiddleware})


class BenchmarkTodoListMiddleware(TodoListMiddleware):
    """Todo middleware whose model-facing prompt is owned by tool_prompts."""

    def __init__(self) -> None:
        super().__init__(
            system_prompt="",
            tool_description=load_tool_prompt("write_todos"),
        )


def build_write_todos_middleware() -> list[AgentMiddleware[Any, Any, Any]]:
    """Create middleware instances that replace SDK-injected todo defaults."""

    return [BenchmarkTodoListMiddleware()]


def create_write_todos_tools() -> list[Any]:
    """Return write_todos tool instances for dynamic prompt rendering."""

    return list(BenchmarkTodoListMiddleware().tools)
