from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_agent_baseline.evidence_agent.semantic import semantic_terms
from data_agent_baseline.evidence_agent.codex_loop.protocol import KnowledgeSection
from data_agent_baseline.evidence_agent.text import code_mentions, normalize_key
from data_agent_baseline.prompts.loader import build_knowledge_bundle


def _section_from_payload(payload: dict[str, Any]) -> KnowledgeSection:
    mentions: list[str] = []
    for mention in payload.get("mentions") or []:
        if isinstance(mention, dict) and str(mention.get("token") or "").strip():
            mentions.append(str(mention["token"]).strip())
    return KnowledgeSection(
        id=str(payload.get("id") or ""),
        heading_path=str(payload.get("heading_path") or ""),
        line_start=payload.get("line_start") if isinstance(payload.get("line_start"), int) else None,
        line_end=payload.get("line_end") if isinstance(payload.get("line_end"), int) else None,
        text=str(payload.get("text") or ""),
        mentions=tuple(mentions),
    )


def build_knowledge_sections(context_dir: Path) -> tuple[list[KnowledgeSection], str, str]:
    bundle = build_knowledge_bundle(context_dir)
    try:
        schema = json.loads(bundle.schema_json)
    except json.JSONDecodeError:
        return [], bundle.schema_json, bundle.content_hash
    if schema.get("availability") != "available":
        return [], bundle.schema_json, bundle.content_hash
    sections = [
        _section_from_payload(section)
        for section in schema.get("sections") or []
        if isinstance(section, dict)
    ]
    return sections, bundle.schema_json, bundle.content_hash


def _query_terms(question: str) -> set[str]:
    terms = set(semantic_terms(question))
    terms.update(normalize_key(part) for part in code_mentions(question))
    return {term for term in terms if term}


def match_knowledge_sections(
    question: str,
    sections: list[KnowledgeSection],
    *,
    limit: int = 8,
) -> list[KnowledgeSection]:
    """Rank sections by lexical overlap only.

    The matcher deliberately has no domain dictionary.  Knowledge is an
    authority document, but matching it must not inject hidden physical or
    fixed business assumptions into the runtime.
    """

    if not sections:
        return []
    terms = _query_terms(question)
    matched: list[tuple[int, int, KnowledgeSection]] = []
    for section in sections:
        section_text = f"{section.heading_path}\n{section.text}"
        section_terms = set(semantic_terms(section_text))
        overlap = len(terms & section_terms)
        if overlap <= 0:
            continue
        matched.append(
            (
                -overlap,
                section.line_start or 10**9,
                KnowledgeSection(
                    id=section.id,
                    heading_path=section.heading_path,
                    line_start=section.line_start,
                    line_end=section.line_end,
                    text=section.text,
                    mentions=section.mentions,
                ),
            )
        )
    matched.sort(key=lambda item: (item[0], item[1]))
    return [section for _overlap, _line, section in matched[:limit]]
