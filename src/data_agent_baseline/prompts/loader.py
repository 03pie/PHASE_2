from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from importlib.resources import files
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask

_PROMPT_PACKAGE = "data_agent_baseline.prompts"
_FileInfoHook = Callable[[Path], str | None]


def _sqlite_table_info(path: Path) -> str | None:
    """只读获取 SQLite 用户表，失败时不影响任务提示词生成。"""

    database_uri = f"file:{path.resolve().as_posix()}?mode=ro"
    try:
        with closing(sqlite3.connect(database_uri, uri=True)) as connection:
            rows = connection.execute(
                """
                SELECT name
                FROM sqlite_schema
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name COLLATE NOCASE
                """
            ).fetchall()
    except sqlite3.Error:
        return None

    table_names = [str(row[0]) for row in rows]
    return f"SQLite tables: {', '.join(table_names) if table_names else '<none>'}"


# 文件信息钩子按后缀扩展，避免不同格式的探测逻辑堆积在清单循环中。
_FILE_INFO_HOOKS: dict[str, _FileInfoHook] = {
    ".db": _sqlite_table_info,
    ".sqlite": _sqlite_table_info,
    ".sqlite3": _sqlite_table_info,
}


def load_prompt(filename: str) -> str:
    """读取 prompts 包中的 Markdown 提示词。"""

    return files(_PROMPT_PACKAGE).joinpath(filename).read_text(encoding="utf-8").strip()


def load_tool_prompt(tool_name: str) -> str:
    """读取单个工具暴露给模型的提示词描述。"""

    if not tool_name or any(separator in tool_name for separator in ("/", "\\")):
        raise ValueError(f"Invalid tool prompt name: {tool_name!r}")
    filename = tool_name if tool_name.endswith(".md") else f"{tool_name}.md"
    return (
        files(_PROMPT_PACKAGE)
        .joinpath("tool_prompts", filename)
        .read_text(encoding="utf-8")
        .strip()
    )


def load_main_agent_prompt() -> str:
    return load_prompt("main_agent.md")


def load_subagent_prompt() -> str:
    return load_prompt("subagent.md")


def _context_inventory(context_dir: Path) -> str:
    entries: list[str] = []
    for path in sorted(context_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(context_dir).as_posix()
        entry = f"- /context/{relative_path} ({path.stat().st_size} bytes)"
        hook = _FILE_INFO_HOOKS.get(path.suffix.lower())
        if hook is not None:
            extra_info = hook(path)
            if extra_info:
                entry = f"{entry}\n  - {extra_info}"
        entries.append(entry)
    return "\n".join(entries) if entries else "- <no context files>"


def _knowledge_context(context_dir: Path) -> str:
    knowledge_path = context_dir / "knowledge.md"
    if not knowledge_path.is_file():
        return "<missing>"
    try:
        content = knowledge_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        return f"<unreadable: {exc}>"
    return content or "<empty>"


def build_task_prompt(
    task: PublicTask,
    *,
    question_structure: str = "<not_run>",
) -> str:
    """向任务模板注入用户问题、固定 knowledge 上下文和完整文件清单。"""

    template = load_prompt("task.md")
    return template.format(
        question=task.question,
        question_structure=question_structure,
        context_inventory=_context_inventory(task.context_dir),
        knowledge_context=_knowledge_context(task.context_dir),
    )
