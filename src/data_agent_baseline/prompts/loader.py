from __future__ import annotations

import json
import sqlite3
import hashlib
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from data_agent_baseline.agents.semantic_layer import build_semantic_context
from data_agent_baseline.benchmark.schema import PublicTask

_PROMPT_PACKAGE = "data_agent_baseline.prompts"
_FileInfoHook = Callable[[Path], str | None]


@dataclass(frozen=True, slots=True)
class KnowledgeBundle:
    """Single-source knowledge payload used by prompt and runtime state."""

    raw_content: str
    schema_json: str
    content_hash: str


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


def read_knowledge_content(context_dir: Path) -> str:
    knowledge_path = context_dir / "knowledge.md"
    if not knowledge_path.is_file():
        return ""
    try:
        return knowledge_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        return f"<unreadable: {exc}>"


def _knowledge_content_hash(raw_content: str) -> str:
    if not raw_content or raw_content.startswith("<unreadable:"):
        return ""
    return hashlib.sha256(raw_content.encode("utf-8")).hexdigest()


def _knowledge_schema_context(
    context_dir: Path,
    *,
    raw_content: str | None = None,
) -> str:
    content = read_knowledge_content(context_dir) if raw_content is None else raw_content
    content_hash = _knowledge_content_hash(content)
    if not content:
        return json.dumps(
            {
                "availability": "missing",
                "knowledge_status_for_plan": "unavailable",
                "content_hash": content_hash,
            },
            ensure_ascii=False,
            indent=2,
        )
    if content.startswith("<unreadable:"):
        return json.dumps(
            {
                "availability": "unreadable",
                "knowledge_status_for_plan": "unavailable",
                "content_hash": content_hash,
                "error": content,
            },
            ensure_ascii=False,
            indent=2,
        )

    semantic = build_semantic_context(context_dir, knowledge_content=content)
    bindings_by_source_hint: dict[str, list[dict[str, object]]] = {}
    for binding in semantic.bindings:
        bindings_by_source_hint.setdefault(binding.source_name_hint, []).append(
            {
                "source_path": binding.source_path,
                "source_type": binding.source_type,
                "table_or_path": binding.table_or_path,
                "confidence": binding.confidence,
                "fields": list(binding.fields[:80]),
                **({"field_count": len(binding.fields)} if len(binding.fields) > 80 else {}),
            }
        )

    sections: dict[str, dict[str, object]] = {}
    global_facts: list[dict[str, object]] = []
    for fact in semantic.facts:
        payload = {
            "fact_id": fact.fact_id,
            "kind": fact.kind,
            "field_key": fact.field_key,
            "operation": fact.operation,
            "binding_status": fact.status,
            "quote": fact.quote,
        }
        if fact.section_key:
            section_payload = sections.setdefault(
                fact.section_key,
                {
                    "section_key": fact.section_key,
                    "source_name_hint": fact.section_key,
                    "source_candidates": bindings_by_source_hint.get(fact.section_key, []),
                    "facts": [],
                },
            )
            facts = section_payload.setdefault("facts", [])
            if isinstance(facts, list):
                facts.append(payload)
        else:
            global_facts.append(payload)

    schema = {
        "availability": "available",
        "knowledge_status_for_plan": "authoritative",
        "source_path": "/context/knowledge.md",
        "content_hash": content_hash,
        "usage_contract": [
            "For analyze_plan.evidence.knowledge_status, use knowledge_status_for_plan exactly.",
            "Use fact_id plus quote when a knowledge fact authorizes a plan decision or output field.",
            "section_key/source_name_hint is a knowledge-document grouping and source-name hint, not proof of a physical data table or row grain.",
            "Treat binding_status values such as binding_unresolved/schema_conflict/narrative_only as local source-binding states, not as proof that all knowledge is invalid.",
            "When a target fact has binding_status narrative_only, bind execution_spec.source_bindings to the observed doc source and use the narrative extraction tool.",
        ],
        "knowledge_sections": list(sections.values()),
        "global_facts": global_facts,
        "issues": list(semantic.issues),
    }
    return json.dumps(schema, ensure_ascii=False, indent=2)


def build_knowledge_bundle(context_dir: Path) -> KnowledgeBundle:
    raw_content = read_knowledge_content(context_dir)
    return KnowledgeBundle(
        raw_content=raw_content,
        schema_json=_knowledge_schema_context(
            context_dir,
            raw_content=raw_content,
        ),
        content_hash=_knowledge_content_hash(raw_content),
    )


def build_task_prompt(
    task: PublicTask,
    *,
    question_structure: str = "<not_run>",
    knowledge_bundle: KnowledgeBundle | None = None,
) -> str:
    """向任务模板注入用户问题、结构化 knowledge schema 和完整文件清单。"""

    bundle = knowledge_bundle or build_knowledge_bundle(task.context_dir)
    template = load_prompt("task.md")
    return template.format(
        question=task.question,
        question_structure=question_structure,
        context_inventory=_context_inventory(task.context_dir),
        knowledge_schema=bundle.schema_json,
    )
