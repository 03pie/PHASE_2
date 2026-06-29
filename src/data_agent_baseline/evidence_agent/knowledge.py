from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from data_agent_baseline.evidence_agent.semantic import semantic_terms
from data_agent_baseline.evidence_agent.codex_loop.protocol import (
    KnowledgeLookupEntry,
    KnowledgeSection,
    KnowledgeSemanticCard,
    KnowledgeSourceMapping,
    SourceRef,
)
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


def _lookup_from_payload(payload: dict[str, Any]) -> KnowledgeLookupEntry | None:
    token = str(payload.get("token") or "").strip()
    if not token:
        return None
    return KnowledgeLookupEntry(
        token=token,
        section_refs=tuple(str(ref) for ref in payload.get("section_refs") or [] if str(ref).strip()),
        evidence_refs=tuple(str(ref) for ref in payload.get("evidence_refs") or [] if str(ref).strip()),
        status=str(payload.get("status") or "document_mention_only"),
        must_verify=bool(payload.get("must_verify", True)),
    )


def _lookup_key(token: str) -> str:
    return normalize_key(token) or token.casefold()


_CODE_SPAN_RE = re.compile(r"`([^`\n]{1,160})`")
_TABLE_ROW_RE = re.compile(r"^\s*\|\s*`?([^`|\n]+?)`?\s*\|\s*(.+?)\s*\|\s*$")
_FORMULA_RE = re.compile(r"`?([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)`?")


def _clean_table_cell(value: str) -> str:
    return value.strip().strip("` ").strip()


def _markdown_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [_clean_table_cell(cell) for cell in stripped.strip("|").split("|")]


def _is_markdown_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def _table_header_map(cells: list[str]) -> dict[str, int]:
    table_names = {"table", "tablename", "relation", "source", "表名", "数据表"}
    field_names = {"field", "fields", "column", "columns", "columnname", "字段", "列名", "字段名"}
    definition_names = {
        "definition",
        "meaning",
        "description",
        "semanticdefinition",
        "语义定义",
        "含义",
        "说明",
        "描述",
    }
    mapping: dict[str, int] = {}
    for index, cell in enumerate(cells):
        key = normalize_key(cell) or cell.casefold()
        if key in table_names and "table" not in mapping:
            mapping["table"] = index
        elif key in field_names and "field" not in mapping:
            mapping["field"] = index
        elif key in definition_names and "definition" not in mapping:
            mapping["definition"] = index
    return mapping


def _identifier_cell(value: str) -> str | None:
    text = _clean_table_cell(value)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return text
    return None


def _unique(items: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return tuple(output)


def _first_code_identifier(text: str, *, allow_dotted: bool = False) -> str | None:
    for match in _CODE_SPAN_RE.finditer(text):
        value = match.group(1).strip()
        pattern = (
            r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
            if allow_dotted
            else r"[A-Za-z_][A-Za-z0-9_]*"
        )
        if re.fullmatch(pattern, value):
            return value
    return None


def _unit_from_text(text: str) -> str | None:
    unit_match = re.search(r"unit\s*:\s*([^\n;。]+)", text, flags=re.IGNORECASE)
    if unit_match:
        return unit_match.group(1).strip()
    cn_match = re.search(r"单位\s*[:：]\s*([^\n;。]+)", text)
    if cn_match:
        return cn_match.group(1).strip()
    if "亿元" in text:
        return "亿元"
    if re.search(r"percentage|百分比|%", text, flags=re.IGNORECASE):
        return "percentage"
    if re.search(r"\bdays?\b|天数|天", text, flags=re.IGNORECASE):
        return "days"
    if re.search(r"\byears?\b|年限|年", text, flags=re.IGNORECASE):
        return "years"
    return None


def _record_grain(table: str, field: str | None, definition: str) -> str:
    haystack = f"{table} {field or ''} {definition}".casefold()
    if "personalcode" in haystack or "fund manager" in haystack or "基金经理" in haystack:
        return "fund_manager"
    if "innercode" in haystack or "fund " in haystack or "基金" in haystack:
        return "fund"
    if "company" in haystack or "fundcompany" in haystack or "公司" in haystack:
        return "fund_company"
    return table or "record"


def _join_keys(definition: str) -> tuple[str, ...]:
    keys: list[str] = []
    for match in _FORMULA_RE.finditer(definition):
        keys.append(f"{match.group(1)}.{match.group(2)}")
    if "primary key" in definition.casefold():
        keys.append("primary_key")
    if "foreign key" in definition.casefold():
        keys.append("foreign_key")
    return _unique(keys)


def _aliases(name: str, definition: str, extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    aliases = [name]
    aliases.extend(extra)
    aliases.extend(_CODE_SPAN_RE.findall(definition))
    for term in semantic_terms(definition):
        if len(term) >= 3:
            aliases.append(term)
    return _unique(aliases[:24])


def _section_kind(section: KnowledgeSection) -> str:
    text = f"{section.heading_path}\n{section.text}".casefold()
    if "business context" in text or "scope" in text or "业务" in text or "范围" in text:
        return "business_context"
    if "calculation" in text or "formula" in text or "metric" in text or "阈值" in text:
        return "metric_rule"
    if "definition" in text or "field" in text or "column" in text:
        return "definition_section"
    if "join" in text or "foreign key" in text or "primary key" in text:
        return "relationship_rule"
    return "section_rule"


def _canonical_scope(section: KnowledgeSection) -> str:
    identifier = (
        _first_code_identifier(section.heading_path, allow_dotted=True)
        or _first_code_identifier(section.text, allow_dotted=True)
    )
    if identifier and "." not in identifier:
        return identifier
    heading = section.heading_path.split(">")[-1].strip() or section.id
    normalized = normalize_key(heading)
    return normalized or "knowledge_global"


def build_semantic_cards(sections: list[KnowledgeSection]) -> list[KnowledgeSemanticCard]:
    cards: list[KnowledgeSemanticCard] = []

    def add_card(
        *,
        kind: str,
        table: str,
        field: str | None,
        name: str,
        definition: str,
        section: KnowledgeSection,
        formula: str | None = None,
        aliases: tuple[str, ...] = (),
    ) -> None:
        card_id = f"sem_{len(cards) + 1:04d}"
        cards.append(
            KnowledgeSemanticCard(
                id=card_id,
                kind=kind,
                canonical_table=table,
                canonical_field=field,
                name=name,
                definition=definition.strip(),
                aliases=_aliases(name, definition, aliases),
                unit=_unit_from_text(definition),
                record_grain=_record_grain(table, field, definition),
                join_keys=_join_keys(definition),
                formula=formula,
                section_id=section.id,
                heading_path=section.heading_path,
                line_start=section.line_start,
                line_end=section.line_end,
            )
        )

    for section in sections:
        section_definition = section.text.strip()
        if section_definition:
            add_card(
                kind=_section_kind(section),
                table=_canonical_scope(section),
                field=None,
                name=section.heading_path or section.id,
                definition=section_definition,
                section=section,
                aliases=section.mentions,
            )

        table = _first_code_identifier(section.heading_path) or _first_code_identifier(section.text)
        if not table:
            formula_match = _FORMULA_RE.search(section.text)
            if formula_match:
                table = formula_match.group(1)
        if not table:
            continue

        table_header: list[str] = []
        for raw_line in section.text.splitlines():
            line = raw_line.strip()
            cells = _markdown_table_cells(line)
            if not cells:
                table_header = []
                continue
            if _is_markdown_separator(cells):
                continue

            header_map = _table_header_map(table_header)
            if not table_header or not header_map:
                table_header = cells
                continue

            field_index = header_map.get("field")
            definition_index = header_map.get("definition")
            if field_index is None:
                continue
            if field_index >= len(cells):
                continue

            row_table = table
            table_index = header_map.get("table")
            if table_index is not None and table_index < len(cells):
                row_table = _identifier_cell(cells[table_index]) or row_table
            field = _identifier_cell(cells[field_index])
            if not field:
                continue
            if definition_index is not None and definition_index < len(cells):
                definition = cells[definition_index]
            else:
                definition = " | ".join(cell for index, cell in enumerate(cells) if index != field_index)
            add_card(
                kind="field",
                table=row_table,
                field=field,
                name=f"{row_table}.{field}",
                definition=definition,
                section=section,
            )

        formula_match = _FORMULA_RE.search(section.text)
        if "Formula:" in section.text and formula_match:
            metric_table = formula_match.group(1)
            metric_field = formula_match.group(2)
            title = section.heading_path.split(">")[-1].strip() or f"{metric_table}.{metric_field}"
            add_card(
                kind="metric",
                table=metric_table,
                field=metric_field,
                name=title,
                definition=section.text,
                section=section,
                formula=f"{metric_table}.{metric_field}",
            )
    return cards


def _canonical_field_name(card: KnowledgeSemanticCard) -> str | None:
    if not card.canonical_field:
        return None
    return f"{card.canonical_table}.{card.canonical_field}".casefold()


def _definition_field_refs(definition: str) -> set[str]:
    refs = {f"{match.group(1)}.{match.group(2)}".casefold() for match in _FORMULA_RE.finditer(definition)}
    alias_to_table: dict[str, str] = {}
    relation_re = re.compile(
        r"\b(?:from|join)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?(?:\s+(?:as\s+)?`?([A-Za-z_][A-Za-z0-9_]*)`?)?",
        re.IGNORECASE,
    )
    for match in relation_re.finditer(definition):
        table = match.group(1)
        alias = match.group(2)
        if alias:
            alias_to_table[alias.casefold()] = table.casefold()
    alias_field_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")
    for match in alias_field_re.finditer(definition):
        left = match.group(1).casefold()
        field = match.group(2).casefold()
        table = alias_to_table.get(left)
        if table:
            refs.add(f"{table}.{field}")
    return refs


def expand_semantic_card_dependencies(
    selected_cards: list[KnowledgeSemanticCard],
    all_cards: list[KnowledgeSemanticCard],
    *,
    max_cards: int = 24,
) -> list[KnowledgeSemanticCard]:
    """Add field/metric cards referenced by selected rule cards.

    This keeps relationship/use-case sections as the user-facing semantic
    entry point, while still surfacing the canonical physical fields they
    require. It is generic: dependencies are derived from dotted identifiers
    and SQL aliases in knowledge text, not from task ids.
    """

    output: list[KnowledgeSemanticCard] = []
    seen: set[str] = set()

    def add(card: KnowledgeSemanticCard) -> None:
        if card.id in seen or len(output) >= max_cards:
            return
        seen.add(card.id)
        output.append(card)

    canonical_to_cards: dict[str, list[KnowledgeSemanticCard]] = {}
    for card in all_cards:
        canonical = _canonical_field_name(card)
        if canonical:
            canonical_to_cards.setdefault(canonical, []).append(card)

    for card in selected_cards:
        add(card)
        if card.kind not in {"relationship_rule", "metric_rule", "section_rule", "definition_section"}:
            continue
        for ref in sorted(_definition_field_refs(card.definition)):
            for dependency in canonical_to_cards.get(ref, []):
                add(dependency)
    return output


def match_semantic_cards(
    question: str,
    cards: list[KnowledgeSemanticCard],
    *,
    limit: int = 12,
) -> list[KnowledgeSemanticCard]:
    query_terms = set(semantic_terms(question))
    matched: list[tuple[int, str, KnowledgeSemanticCard]] = []
    for card in cards:
        card_terms = set(
            semantic_terms(
                " ".join(
                    [
                        card.name,
                        card.canonical_table,
                        card.canonical_field or "",
                        card.definition,
                        *card.aliases,
                    ]
                )
            )
        )
        overlap = len(query_terms & card_terms)
        if overlap <= 0:
            continue
        matched.append((-overlap, card.id, card))
    matched.sort(key=lambda item: (item[0], item[1]))
    return [card for _score, _id, card in matched[:limit]]


def build_source_mappings(
    cards: list[KnowledgeSemanticCard],
    sources: list[SourceRef] | tuple[SourceRef, ...],
) -> list[KnowledgeSourceMapping]:
    mappings: list[KnowledgeSourceMapping] = []
    for card in cards:
        if card.kind in {
            "business_context",
            "definition_section",
            "relationship_rule",
            "metric_rule",
            "section_rule",
        } and not card.canonical_field:
            mappings.append(
                KnowledgeSourceMapping(
                    card_id=card.id,
                    source_id=None,
                    source_path=None,
                    data_form=None,
                    status="semantic_only",
                    matched_table=card.canonical_table,
                    matched_field=None,
                    match_reason="section-level knowledge rule; no physical source required",
                )
            )
            continue
        table_norm = normalize_key(card.canonical_table)
        field_norm = normalize_key(card.canonical_field or "")
        found = False
        for source in sources:
            source_names = {
                normalize_key(source.stem),
                normalize_key(source.basename),
                normalize_key(source.virtual_path),
                *(normalize_key(table) for table in source.tables),
            }
            column_names = {normalize_key(column) for column in source.columns}
            table_match = bool(
                table_norm
                and any(table_norm == name or table_norm in name for name in source_names if name)
            )
            field_match = bool(field_norm and field_norm in column_names)
            if source.data_form in {"sqlite_database", "csv_records", "json_records"}:
                if table_match and (field_match or not source.columns):
                    mappings.append(
                        KnowledgeSourceMapping(
                            card_id=card.id,
                            source_id=source.id,
                            source_path=source.virtual_path,
                            data_form=source.data_form,
                            status="exact_structured_source",
                            matched_table=card.canonical_table,
                            matched_field=card.canonical_field if field_match else None,
                            match_reason="table/source name matched knowledge table and field matched observed columns when available",
                        )
                    )
                    found = True
                elif table_match or field_match:
                    mappings.append(
                        KnowledgeSourceMapping(
                            card_id=card.id,
                            source_id=source.id,
                            source_path=source.virtual_path,
                            data_form=source.data_form,
                            status="fallback_candidate",
                            matched_table=card.canonical_table if table_match else None,
                            matched_field=card.canonical_field if field_match else None,
                            match_reason="partial table or field match; verify grain before use",
                            warnings=("partial_source_mapping",),
                        )
                    )
                    found = True
            elif source.data_form in {"pdf_document", "markdown_document"} and table_match:
                mappings.append(
                    KnowledgeSourceMapping(
                        card_id=card.id,
                        source_id=source.id,
                        source_path=source.virtual_path,
                        data_form=source.data_form,
                        status="document_source",
                        matched_table=card.canonical_table,
                        matched_field=None,
                        match_reason="document filename/path matched knowledge table; use DocumentAgent for evidence",
                    )
                )
                found = True
        if not found:
            mappings.append(
                KnowledgeSourceMapping(
                    card_id=card.id,
                    source_id=None,
                    source_path=None,
                    data_form=None,
                    status="unsupported_or_missing",
                    matched_table=card.canonical_table,
                    matched_field=card.canonical_field,
                    match_reason="no observed source matched this semantic card",
                    warnings=("must_locate_or_block",),
                )
            )
    return mappings


def build_knowledge_catalog(
    context_dir: Path,
) -> tuple[list[KnowledgeSection], dict[str, KnowledgeLookupEntry], str, str]:
    bundle = build_knowledge_bundle(context_dir)
    try:
        schema = json.loads(bundle.schema_json)
    except json.JSONDecodeError:
        return [], {}, bundle.schema_json, bundle.content_hash
    if schema.get("availability") != "available":
        return [], {}, bundle.schema_json, bundle.content_hash
    sections = [
        _section_from_payload(section)
        for section in schema.get("sections") or []
        if isinstance(section, dict)
    ]
    lookup_entries = [
        entry
        for payload in schema.get("lookup") or []
        if isinstance(payload, dict)
        for entry in [_lookup_from_payload(payload)]
        if entry is not None
    ]
    lookup = {_lookup_key(entry.token): entry for entry in lookup_entries}
    return sections, lookup, bundle.schema_json, bundle.content_hash


def build_knowledge_sections(context_dir: Path) -> tuple[list[KnowledgeSection], str, str]:
    sections, _lookup, schema_json, content_hash = build_knowledge_catalog(context_dir)
    return sections, schema_json, content_hash


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
