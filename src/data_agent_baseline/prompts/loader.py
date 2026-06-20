from __future__ import annotations

import json
import sqlite3
import hashlib
import re
import unicodedata
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

_KNOWLEDGE_SCHEMA_STOP_TERMS = {
    "column",
    "data",
    "database",
    "definition",
    "field",
    "fields",
    "record",
    "records",
    "semantic",
    "source",
    "table",
    "unit",
    "value",
    "values",
}


def _schema_terms(value: object) -> set[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    terms: set[str] = set()
    for token in re.findall(r"[0-9a-z_]{2,}", text):
        parts = [part for part in token.split("_") if part]
        for part in [token, *parts]:
            if len(part) > 3 and part.endswith("ies"):
                part = f"{part[:-3]}y"
            elif len(part) > 3 and part.endswith("s"):
                part = part[:-1]
            if part and part not in _KNOWLEDGE_SCHEMA_STOP_TERMS:
                terms.add(part)
    for sequence in re.findall(r"[\u3400-\u9fff]+", text):
        if len(sequence) < 2:
            continue
        terms.add(sequence)
        for size in (2, 3, 4):
            if len(sequence) < size:
                continue
            terms.update(
                sequence[index : index + size]
                for index in range(0, len(sequence) - size + 1)
            )
    return terms


def _source_hints_from_fact_text(text: object) -> set[str]:
    source_hints: set[str] = set()
    raw_text = str(text or "")
    for match in re.finditer(
        r"\b(?:FROM|JOIN)\s+[`\"\[]?([A-Za-z][A-Za-z0-9_]*)",
        raw_text,
        flags=re.IGNORECASE,
    ):
        source_hints.add(match.group(1))
    for match in re.finditer(r"`([A-Za-z][A-Za-z0-9_]*)\.[^`]+`", raw_text):
        source_hints.add(match.group(1))
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]*)\.[A-Za-z][A-Za-z0-9_]*", raw_text):
        source_hints.add(match.group(1))
    return source_hints


def _focused_section_keys(
    semantic_facts: object,
    *,
    focus_text: str | None,
    known_sections: set[str],
) -> set[str] | None:
    focus_terms = _schema_terms(focus_text)
    if not focus_terms:
        return None

    facts = list(semantic_facts or [])
    section_scores: dict[str, int] = {}
    relevant_fact_sections: set[str] = set()
    for fact in facts:
        section_key = str(getattr(fact, "section_key", "") or "").strip()
        if not section_key:
            continue
        fact_text = " ".join(
            str(item or "")
            for item in (
                getattr(fact, "section_key", ""),
                getattr(fact, "field_key", ""),
                getattr(fact, "operation", ""),
                getattr(fact, "quote", ""),
            )
        )
        overlap = _schema_terms(fact_text) & focus_terms
        score = len(overlap)
        if score < 2:
            continue
        section_scores[section_key] = max(section_scores.get(section_key, 0), score)
        relevant_fact_sections.add(section_key)
        for source_hint in _source_hints_from_fact_text(getattr(fact, "quote", "")):
            if source_hint in known_sections:
                relevant_fact_sections.add(source_hint)

    if not section_scores:
        return None
    return {
        section_key
        for section_key in relevant_fact_sections
        if section_key in known_sections
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
    focus_text: str | None = None,
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

    focused_section_keys = _focused_section_keys(
        semantic.facts,
        focus_text=focus_text,
        known_sections=set(sections),
    )
    omitted_section_count = 0
    if focused_section_keys:
        omitted_section_count = len(
            [key for key in sections if key not in focused_section_keys]
        )
        sections = {
            key: value
            for key, value in sections.items()
            if key in focused_section_keys
        }
        focused_text_terms = _schema_terms(focus_text)
        focused_section_text = " ".join(focused_section_keys)
        global_facts = [
            fact
            for fact in global_facts
            if (
                _schema_terms(" ".join(str(value or "") for value in fact.values()))
                & focused_text_terms
            )
            or any(
                section_key and section_key in str(fact.get("quote") or "")
                for section_key in focused_section_keys
            )
            or bool(
                _schema_terms(str(fact.get("quote") or ""))
                & _schema_terms(focused_section_text)
            )
        ]

    schema = {
        "availability": "available",
        "knowledge_status_for_plan": "authoritative",
        "source_path": "/context/knowledge.md",
        "content_hash": content_hash,
        "usage_contract": [
            "For analyze_plan.evidence.knowledge_status, use knowledge_status_for_plan exactly.",
            "Use knowledge facts as semantic definitions for terms, fields, units, metrics, and calculation logic.",
            "fact_id, section_key, field_key, source_name_hint, and markdown table headers are document references only; they do not prove physical tables, columns, row grain, or file formats.",
            "Treat binding_status values such as binding_unresolved/schema_conflict/narrative_only as discovery hints, not as proof that knowledge is invalid.",
            "Discover the actual data format with tools. For doc/PDF/Markdown sources, use grep/read_doc slices and extraction; field_key is an extraction target, not proof of an existing physical column.",
        ],
        "knowledge_sections": list(sections.values()),
        "global_facts": global_facts,
        "issues": list(semantic.issues),
    }
    if focused_section_keys:
        schema["focus"] = {
            "applied": True,
            "section_keys": sorted(focused_section_keys),
            "omitted_section_count": omitted_section_count,
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
        knowledge_schema=_knowledge_schema_context(
            task.context_dir,
            raw_content=bundle.raw_content,
            focus_text=f"{task.question}\n{question_structure}",
        ),
    )
