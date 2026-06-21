from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class KnowledgeBundle:
    """Document-only knowledge payload used by the evidence runtime."""

    raw_content: str
    schema_json: str
    content_hash: str

_MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_MARKDOWN_LIST_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+?)\s*$")
_MARKDOWN_TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_MARKDOWN_RULE_PATTERN = re.compile(r"^\s*[-—_]{3,}\s*$")
_PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*(?:Page\s+\d+(?:\s+of\s+\d+)?|第\s*\d+\s*页(?:\s*共\s*\d+\s*页)?|\d+\s*/\s*\d+)\s*$",
    flags=re.IGNORECASE,
)
_MARKDOWN_CODE_SPAN_PATTERN = re.compile(r"`([^`\n]{1,160})`")
_MARKDOWN_IDENTIFIER_PATTERN = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b"
)
_MARKDOWN_IDENTIFIER_STOPWORDS = frozenset(
    {
        "and",
        "as",
        "asc",
        "avg",
        "by",
        "case",
        "column",
        "count",
        "database",
        "definition",
        "desc",
        "else",
        "end",
        "from",
        "group",
        "guide",
        "having",
        "in",
        "inner",
        "join",
        "json",
        "knowledge",
        "left",
        "limit",
        "max",
        "metric",
        "min",
        "on",
        "or",
        "order",
        "outer",
        "right",
        "role",
        "select",
        "semantic",
        "sql",
        "sum",
        "table",
        "then",
        "unit",
        "when",
        "yaml",
        "where",
    }
)
_KNOWLEDGE_SCHEMA_KIND = "knowledge_document_index"
_KNOWLEDGE_SCHEMA_VERSION = "1.0"
_EMBEDDED_PATH_PATTERN = re.compile(
    r"<path>\s*(.*?)\s*</path>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _strip_inline_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    return " ".join(text.split())


def _is_noise_markdown_line(line: str) -> bool:
    stripped = line.strip()
    return not stripped or bool(
        _MARKDOWN_RULE_PATTERN.match(stripped) or _PAGE_NUMBER_PATTERN.match(stripped)
    )


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _heading_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(title for _level, title in stack)


def _append_markdown_chunk(
    chunks: list[dict[str, object]],
    *,
    heading_stack: list[tuple[int, str]],
    chunk_type: str,
    line_start: int,
    line_end: int,
    raw_lines: list[str],
    text: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    raw_text = "\n".join(raw_lines).strip()
    rendered_text = text if text is not None else _strip_inline_markdown(raw_text)
    if not raw_text or not rendered_text:
        return
    payload: dict[str, object] = {
        "id": f"kchunk_{len(chunks) + 1:04d}",
        "heading_path": _heading_path(heading_stack),
        "heading_level": heading_stack[-1][0] if heading_stack else None,
        "section_title": heading_stack[-1][1] if heading_stack else "",
        "type": chunk_type,
        "line_start": line_start,
        "line_end": line_end,
        "text": rendered_text,
        "raw_text": raw_text,
    }
    if extra:
        payload.update(extra)
    chunks.append(payload)


def _parse_markdown_table(
    lines: list[str],
    start_index: int,
    heading_stack: list[tuple[int, str]],
    chunks: list[dict[str, object]],
) -> int:
    header = _split_table_row(lines[start_index])
    index = start_index + 1
    if index < len(lines) and _MARKDOWN_TABLE_SEPARATOR_PATTERN.match(lines[index]):
        index += 1
    table_start = start_index + 1
    row_number = 0
    while index < len(lines) and "|" in lines[index] and lines[index].strip():
        row = _split_table_row(lines[index])
        if not row or _MARKDOWN_TABLE_SEPARATOR_PATTERN.match(lines[index]):
            index += 1
            continue
        row_number += 1
        pairs = [
            f"{header[column_index] if column_index < len(header) else f'column_{column_index + 1}'}: {cell}"
            for column_index, cell in enumerate(row)
        ]
        _append_markdown_chunk(
            chunks,
            heading_stack=heading_stack,
            chunk_type="table_row",
            line_start=index + 1,
            line_end=index + 1,
            raw_lines=[lines[index]],
            text=f"In {_heading_path(heading_stack)}, " + "; ".join(pairs),
            extra={
                "table_start_line": table_start,
                "table_row_index": row_number,
                "headers": header,
                "cells": row,
            },
        )
        index += 1
    return index


def _knowledge_markdown_chunks(content: str) -> list[dict[str, object]]:
    lines = content.splitlines()
    chunks: list[dict[str, object]] = []
    heading_stack: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if _is_noise_markdown_line(line):
            index += 1
            continue

        heading_match = _MARKDOWN_HEADING_PATTERN.match(stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title = _strip_inline_markdown(heading_match.group(2).strip())
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            _append_markdown_chunk(
                chunks,
                heading_stack=heading_stack,
                chunk_type="heading",
                line_start=index + 1,
                line_end=index + 1,
                raw_lines=[line],
                text=title,
            )
            index += 1
            continue

        if stripped.startswith("```"):
            block_start = index
            block_lines = [line]
            index += 1
            while index < len(lines):
                block_lines.append(lines[index])
                if lines[index].strip().startswith("```"):
                    index += 1
                    break
                index += 1
            _append_markdown_chunk(
                chunks,
                heading_stack=heading_stack,
                chunk_type="code_block",
                line_start=block_start + 1,
                line_end=block_start + len(block_lines),
                raw_lines=block_lines,
            )
            continue

        if "|" in line and index + 1 < len(lines) and _MARKDOWN_TABLE_SEPARATOR_PATTERN.match(lines[index + 1]):
            index = _parse_markdown_table(lines, index, heading_stack, chunks)
            continue

        list_match = _MARKDOWN_LIST_PATTERN.match(line)
        if list_match:
            _append_markdown_chunk(
                chunks,
                heading_stack=heading_stack,
                chunk_type="list_item",
                line_start=index + 1,
                line_end=index + 1,
                raw_lines=[line],
                text=_strip_inline_markdown(list_match.group(1)),
            )
            index += 1
            continue

        paragraph_start = index
        paragraph_lines: list[str] = []
        while index < len(lines):
            candidate = lines[index]
            candidate_stripped = candidate.strip()
            if (
                _is_noise_markdown_line(candidate)
                or _MARKDOWN_HEADING_PATTERN.match(candidate_stripped)
                or candidate_stripped.startswith("```")
                or (
                    "|" in candidate
                    and index + 1 < len(lines)
                    and _MARKDOWN_TABLE_SEPARATOR_PATTERN.match(lines[index + 1])
                )
                or _MARKDOWN_LIST_PATTERN.match(candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        _append_markdown_chunk(
            chunks,
            heading_stack=heading_stack,
            chunk_type="content",
            line_start=paragraph_start + 1,
            line_end=paragraph_start + len(paragraph_lines),
            raw_lines=paragraph_lines,
        )
    return chunks


def _knowledge_identifier_value(value: str, *, code_span: bool) -> str | None:
    candidate = value.strip().strip("`").strip()
    candidate = candidate.strip(".,;:()[]{}<>")
    if len(candidate) < 2 or len(candidate) > 120:
        return None
    normalized = candidate.casefold()
    if normalized in _MARKDOWN_IDENTIFIER_STOPWORDS:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", candidate):
        return None
    if code_span:
        if re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
            candidate,
        ):
            return candidate
        return None
    if "." in candidate or "_" in candidate:
        return candidate
    if any(char.islower() for char in candidate) and any(
        char.isupper() for char in candidate[1:]
    ):
        return candidate
    return None


def _knowledge_route_target_candidates(
    value: str,
    *,
    code_span: bool,
) -> list[str]:
    identifier = _knowledge_identifier_value(value, code_span=code_span)
    if identifier is None:
        return []
    if "." not in identifier:
        candidates = [identifier]
    else:
        parts = [part for part in identifier.split(".") if part]
        candidates = parts if parts and len(parts[0]) == 1 else [identifier, *parts]
    return [
        candidate
        for candidate in candidates
        if len(candidate) >= 2
        and candidate.casefold() not in _MARKDOWN_IDENTIFIER_STOPWORDS
    ]


def _route_target_occurrences_from_part(
    part: dict[str, object],
) -> list[tuple[str, dict[str, object]]]:
    text = str(part.get("text") or "")
    raw_text = str(part.get("raw_text") or "")
    search_text = "\n".join(value for value in (raw_text, text) if value)
    part_type = str(part.get("part_type") or "")
    markdown_type = str(part.get("source_markdown_type") or part_type)
    source_text = text or raw_text
    if len(source_text) > 240:
        source_text = f"{source_text[:237]}..."
    occurrences: dict[tuple[str, str, object, object], tuple[str, dict[str, object]]] = {}

    def add(raw_value: str, *, code_span: bool) -> None:
        for value in _knowledge_route_target_candidates(raw_value, code_span=code_span):
            key = (
                value.casefold(),
                str(part.get("id") or ""),
                part.get("line_start"),
                part.get("line_end"),
            )
            occurrence = {
                "part_id": part.get("id"),
                "part_type": part_type,
                "source_markdown_type": markdown_type,
                "line_start": part.get("line_start"),
                "line_end": part.get("line_end"),
                "code_span": code_span,
                "source_text": source_text,
            }
            existing = occurrences.get(key)
            if existing is not None and existing[1].get("code_span"):
                continue
            occurrences[key] = (value, occurrence)

    for match in _MARKDOWN_CODE_SPAN_PATTERN.finditer(search_text):
        code_text = match.group(1)
        add(code_text, code_span=True)
        for identifier in _MARKDOWN_IDENTIFIER_PATTERN.finditer(code_text):
            add(identifier.group(0), code_span=True)
    identifier_code_span = markdown_type == "code_block"
    for match in _MARKDOWN_IDENTIFIER_PATTERN.finditer(search_text):
        add(match.group(0), code_span=identifier_code_span)
    return list(occurrences.values())


def _embedded_content_view(raw_content: str) -> tuple[str, dict[str, object]]:
    """Return Markdown content used for parsing while preserving useful source metadata."""

    source: dict[str, object] = {
        "path": "/context/knowledge.md",
    }
    path_match = _EMBEDDED_PATH_PATTERN.search(raw_content)
    if path_match is not None:
        embedded_path = " ".join(path_match.group(1).split())
        if embedded_path:
            source["embedded_path"] = embedded_path

    lines = raw_content.splitlines()
    content_start: int | None = None
    content_end: int | None = None
    for index, line in enumerate(lines):
        if re.search(r"<content>", line, flags=re.IGNORECASE):
            content_start = index
            continue
        if content_start is not None and re.search(
            r"</content>",
            line,
            flags=re.IGNORECASE,
        ):
            content_end = index
            break
    if content_start is None or content_end is None or content_end <= content_start:
        return raw_content, source

    prefix_blanks = [""] * (content_start + 1)
    return "\n".join([*prefix_blanks, *lines[content_start + 1 : content_end]]), source


def _is_skipped_knowledge(content: str) -> bool:
    normalized = "\n".join(line.strip().casefold() for line in content.splitlines()[:5])
    return "step3 skipped" in normalized


def _schema_usage_contract() -> list[str]:
    return [
        "This schema indexes document content only.",
        "Markdown tables, headings, lists, and code blocks are document forms, not physical data formats.",
        "Every mention is an untyped lookup token and must be verified against observed files/data before use.",
        "Sections are the only semantic body. Consume a matched section as complete document context.",
        "Do not treat a mention as a field, table, file, value, or executable plan until a tool observation proves that role.",
    ]


def _document_markdown_type(chunk_type: str) -> str:
    return {
        "heading": "heading",
        "content": "paragraph",
        "list_item": "list",
        "code_block": "code_block",
        "table_row": "markdown_table",
    }.get(chunk_type, "paragraph")


def _block_mentions(block: dict[str, object]) -> list[str]:
    part = {
        "id": block.get("id"),
        "part_type": block.get("markdown_type"),
        "source_markdown_type": block.get("markdown_type"),
        "line_start": block.get("line_start"),
        "line_end": block.get("line_end"),
        "text": block.get("text"),
        "raw_text": block.get("text"),
    }
    values: list[str] = []
    seen: set[str] = set()
    for value, _occurrence in _route_target_occurrences_from_part(part):
        normalized = value.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        values.append(value)
    return values


def _document_block_from_chunk(
    chunk: dict[str, object],
    *,
    block_id: str,
) -> dict[str, object]:
    markdown_type = _document_markdown_type(str(chunk.get("type") or ""))
    text = str(chunk.get("raw_text") or chunk.get("text") or "").strip()
    if markdown_type == "paragraph":
        text = str(chunk.get("text") or text).strip()
    return {
        key: value
        for key, value in {
            "id": block_id,
            "markdown_type": markdown_type,
            "line_start": chunk.get("line_start"),
            "line_end": chunk.get("line_end"),
            "text": text,
            "document_form_only": True,
        }.items()
        if value not in ("", [], None)
    }


def _document_table_block(
    table_chunks: list[dict[str, object]],
    *,
    block_id: str,
) -> dict[str, object]:
    first = table_chunks[0]
    headers = [str(item) for item in first.get("headers") or []]
    rows = [
        [str(cell) for cell in chunk.get("cells") or []]
        for chunk in table_chunks
    ]
    raw_rows = [
        str(chunk.get("raw_text") or "").strip()
        for chunk in table_chunks
        if str(chunk.get("raw_text") or "").strip()
    ]
    table_text_lines = []
    if headers:
        table_text_lines.append("| " + " | ".join(headers) + " |")
    table_text_lines.extend(raw_rows)
    return {
        key: value
        for key, value in {
            "id": block_id,
            "markdown_type": "markdown_table",
            "line_start": first.get("table_start_line") or first.get("line_start"),
            "line_end": table_chunks[-1].get("line_end"),
            "text": "\n".join(table_text_lines).strip(),
            "markdown_table": {
                "headers": headers,
                "rows": rows,
            },
            "document_form_only": True,
        }.items()
        if value not in ("", [], None)
    }


def _sections_from_document_chunks(
    chunks: list[dict[str, object]],
    *,
    content: str | None = None,
) -> list[dict[str, object]]:
    section_map: dict[str, dict[str, object]] = {}
    section_order: list[str] = []
    block_counter = 0
    content_lines = content.splitlines() if isinstance(content, str) else []
    for chunk in chunks:
        heading_path = str(chunk.get("heading_path") or "").strip()
        section_key = heading_path or "<root>"
        if section_key not in section_map:
            section_map[section_key] = {
                "id": f"sec_{len(section_order) + 1:04d}",
                "heading_path": heading_path,
                "line_start": chunk.get("line_start"),
                "line_end": chunk.get("line_end"),
                "_chunks": [],
            }
            section_order.append(section_key)
        section = section_map[section_key]
        section["line_start"] = min(
            int(section.get("line_start") or chunk.get("line_start") or 0),
            int(chunk.get("line_start") or section.get("line_start") or 0),
        )
        section["line_end"] = max(
            int(section.get("line_end") or chunk.get("line_end") or 0),
            int(chunk.get("line_end") or section.get("line_end") or 0),
        )
        section_chunks = section["_chunks"]
        if isinstance(section_chunks, list):
            section_chunks.append(chunk)

    sections: list[dict[str, object]] = []
    for section_key in section_order:
        section = section_map[section_key]
        chunks_for_section = [
            chunk
            for chunk in section.get("_chunks") or []
            if isinstance(chunk, dict)
        ]
        blocks: list[dict[str, object]] = []
        index = 0
        while index < len(chunks_for_section):
            chunk = chunks_for_section[index]
            block_counter += 1
            block_id = f"blk_{block_counter:04d}"
            if str(chunk.get("type") or "") == "table_row":
                table_start = chunk.get("table_start_line")
                table_chunks = [chunk]
                index += 1
                while index < len(chunks_for_section):
                    next_chunk = chunks_for_section[index]
                    if (
                        str(next_chunk.get("type") or "") != "table_row"
                        or next_chunk.get("table_start_line") != table_start
                    ):
                        break
                    table_chunks.append(next_chunk)
                    index += 1
                blocks.append(_document_table_block(table_chunks, block_id=block_id))
                continue
            blocks.append(_document_block_from_chunk(chunk, block_id=block_id))
            index += 1

        mention_map: dict[str, dict[str, object]] = {}
        mention_order: list[str] = []
        for block in blocks:
            block_id = str(block.get("id") or "")
            for token in _block_mentions(block):
                normalized = token.casefold()
                if normalized not in mention_map:
                    mention_order.append(normalized)
                    mention_map[normalized] = {
                        "token": token,
                        "evidence_refs": [],
                        "status": "document_mention_only",
                        "must_verify": True,
                    }
                refs = mention_map[normalized]["evidence_refs"]
                if isinstance(refs, list) and block_id and block_id not in refs:
                    refs.append(block_id)

        fallback_text = "\n\n".join(
            str(block.get("text") or "")
            for block in blocks
            if str(block.get("text") or "").strip()
        ).strip()
        line_start = section.get("line_start")
        line_end = section.get("line_end")
        text = fallback_text
        if (
            content_lines
            and isinstance(line_start, int)
            and isinstance(line_end, int)
            and line_start > 0
            and line_end >= line_start
        ):
            text = "\n".join(content_lines[line_start - 1 : line_end]).strip()
        sections.append(
            {
                key: value
                for key, value in {
                    "id": section.get("id"),
                    "heading_path": section.get("heading_path"),
                    "line_start": section.get("line_start"),
                    "line_end": section.get("line_end"),
                    "text": text,
                    "blocks": blocks,
                    "mentions": [mention_map[key] for key in mention_order],
                }.items()
                if value not in ("", [], None)
            }
        )
    return sections


def _document_lookup(sections: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for section in sections:
        section_id = str(section.get("id") or "")
        for mention in section.get("mentions") or []:
            if not isinstance(mention, dict):
                continue
            token = str(mention.get("token") or "").strip()
            if not token:
                continue
            normalized = token.casefold()
            if normalized not in lookup:
                order.append(normalized)
                lookup[normalized] = {
                    "token": token,
                    "section_refs": [],
                    "evidence_refs": [],
                    "status": "document_mention_only",
                    "must_verify": True,
                }
            section_refs = lookup[normalized]["section_refs"]
            if isinstance(section_refs, list) and section_id and section_id not in section_refs:
                section_refs.append(section_id)
            evidence_refs = lookup[normalized]["evidence_refs"]
            if isinstance(evidence_refs, list):
                for ref in mention.get("evidence_refs") or []:
                    if isinstance(ref, str) and ref not in evidence_refs:
                        evidence_refs.append(ref)
    return [lookup[key] for key in order]


def _base_document_schema(
    *,
    availability: str,
    source: dict[str, object],
    error: str | None = None,
) -> dict[str, object]:
    schema: dict[str, object] = {
        "schema_kind": _KNOWLEDGE_SCHEMA_KIND,
        "schema_version": _KNOWLEDGE_SCHEMA_VERSION,
        "availability": availability,
        "source": source,
        "contract": _schema_usage_contract(),
        "sections": [],
        "lookup": [],
    }
    if error:
        schema["error"] = error
    return schema


def _build_document_knowledge_schema(
    *,
    content: str,
    content_hash: str,
    source: dict[str, object],
) -> dict[str, object]:
    source = {
        key: value
        for key, value in {
            **source,
            "content_hash": content_hash,
        }.items()
        if value not in ("", [], None)
    }
    if _is_skipped_knowledge(content):
        return _base_document_schema(availability="skipped", source=source)

    sections = _sections_from_document_chunks(
        _knowledge_markdown_chunks(content),
        content=content,
    )
    return {
        "schema_kind": _KNOWLEDGE_SCHEMA_KIND,
        "schema_version": _KNOWLEDGE_SCHEMA_VERSION,
        "availability": "available",
        "source": source,
        "contract": _schema_usage_contract(),
        "sections": sections,
        "lookup": _document_lookup(sections),
    }


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
        schema = _base_document_schema(
            availability="missing",
            source={
                "path": "/context/knowledge.md",
                "content_hash": content_hash,
            },
        )
        return json.dumps(schema, ensure_ascii=False, indent=2)
    if content.startswith("<unreadable:"):
        schema = _base_document_schema(
            availability="unreadable",
            source={
                "path": "/context/knowledge.md",
                "content_hash": content_hash,
            },
            error=content,
        )
        return json.dumps(schema, ensure_ascii=False, indent=2)

    parse_content, source = _embedded_content_view(content)
    schema = _build_document_knowledge_schema(
        content=parse_content,
        content_hash=content_hash,
        source=source,
    )
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
