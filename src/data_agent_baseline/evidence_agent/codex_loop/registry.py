from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

import fitz
import pandas as pd

from data_agent_baseline.evidence_agent.codex_loop.compute import (
    load_binding_frame,
    run_sql_over_bindings,
)
from data_agent_baseline.evidence_agent.codex_loop.document_agent import (
    DocTask,
    DocumentAgent,
)
from data_agent_baseline.evidence_agent.codex_loop.protocol import (
    Evidence,
    LoopState,
    ModelAction,
    SourceRef,
    ToolSpec,
)
from data_agent_baseline.evidence_agent.text import normalize_key


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _value_keys(value: Any) -> set[str]:
    text = str(value).strip()
    keys = {
        _normalize(text),
        re.sub(r"\s+", "", text.casefold()),
    }
    try:
        number = float(text.replace(",", ""))
    except (TypeError, ValueError):
        return {key for key in keys if key}
    keys.add(f"num:{number:.12g}")
    return {key for key in keys if key}


def _terms(value: Any) -> list[str]:
    text = str(value).casefold()
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        term = term.strip()
        if not term or term in seen:
            return
        seen.add(term)
        terms.append(term)

    for part in re.split(r"[^0-9A-Za-z_]+", text):
        if part:
            add(part)
    for chunk in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", text):
        if len(chunk) >= 2:
            add(chunk)
        upper = min(8, len(chunk))
        for size in range(upper, 1, -1):
            for start in range(0, len(chunk) - size + 1):
                add(chunk[start : start + size])
    terms.sort(key=lambda item: (-len(item), item))
    return terms


def _source_from_args(state: LoopState, arguments: dict[str, Any]) -> SourceRef | None:
    source_ref = str(arguments.get("source_ref") or arguments.get("source_id") or "").strip()
    if source_ref and source_ref in state.sources:
        return state.sources[source_ref]
    path = str(arguments.get("path") or "").strip()
    if path:
        source_id = state.source_by_path.get(path) or state.source_by_path.get(Path(path).as_posix())
        if source_id:
            return state.sources[source_id]
    return None


_SQL_RELATION_PATTERN = re.compile(
    r'\b(?:from|join)\s+("([^"]+)"|[A-Za-z_][\w]*)',
    re.IGNORECASE,
)


def _binding_refs_from_sql_or_args(
    state: LoopState,
    *,
    sql: str,
    arguments: dict[str, Any],
) -> tuple[str, ...]:
    relation_names: set[str] = set()
    relation_arg = str(arguments.get("relation_name") or "").strip()
    if relation_arg:
        relation_names.add(relation_arg)
    for match in _SQL_RELATION_PATTERN.finditer(sql):
        relation_names.add((match.group(2) or match.group(1)).strip('"'))
    if not relation_names:
        return ()
    refs = [
        binding.id
        for binding in state.bindings.values()
        if binding.relation_name in relation_names
    ]
    return tuple(dict.fromkeys(refs))


def _sample_dataframe(frame: pd.DataFrame, *, limit: int = 5) -> list[dict[str, Any]]:
    return frame.head(limit).where(pd.notnull(frame), None).to_dict(orient="records")


def _json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if all(isinstance(item, dict) for item in payload):
            return list(payload)
        return [{"value": item} for item in payload]
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return list(value)
        return [payload]
    return [{"value": payload}]


def _document_lines(source: SourceRef) -> list[dict[str, Any]]:
    if source.data_form == "markdown_document":
        text = source.path.read_text(encoding="utf-8", errors="replace")
        return [
            {"line": index + 1, "page": None, "text": line}
            for index, line in enumerate(text.splitlines())
        ]
    if source.data_form == "pdf_document":
        lines: list[dict[str, Any]] = []
        with fitz.open(source.path) as document:
            global_line = 1
            for page_index, page in enumerate(document, start=1):
                for line in page.get_text("text").splitlines():
                    lines.append({"line": global_line, "page": page_index, "text": line})
                    global_line += 1
        return lines
    return []


def _recommend(tool_name: str, arguments: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"tool_name": tool_name, "arguments": arguments, "reason": reason}


def _answer_table_from_payload(answer: Any) -> tuple[list[str], list[list[Any]]] | None:
    if not isinstance(answer, dict) or not answer:
        return None
    columns = answer.get("columns")
    rows = answer.get("rows")
    if isinstance(columns, list) and isinstance(rows, list):
        normalized_columns = [str(column) for column in columns]
        normalized_rows: list[list[Any]] = []
        for row in rows:
            if isinstance(row, list):
                normalized_rows.append(list(row))
            elif isinstance(row, tuple):
                normalized_rows.append(list(row))
            elif isinstance(row, dict):
                normalized_rows.append([row.get(column) for column in normalized_columns])
            else:
                normalized_rows.append([row])
        return normalized_columns, normalized_rows
    if "value" in answer:
        column = str(answer.get("column") or answer.get("name") or "answer")
        return [column], [[answer.get("value")]]
    scalar_items = [
        (str(key), value)
        for key, value in answer.items()
        if key not in {"binding_refs", "evidence_refs", "alignment"}
        and not isinstance(value, (dict, list))
    ]
    if scalar_items:
        return [key for key, _value in scalar_items], [[value for _key, value in scalar_items]]
    return None


def _projection_values_supported(
    answer_rows: list[list[Any]],
    compute_rows: tuple[tuple[Any, ...], ...],
) -> tuple[bool, list[Any]]:
    observed = set()
    for row in compute_rows:
        for cell in row:
            if cell is not None and str(cell).strip():
                observed.update(_value_keys(cell))
    unsupported: list[Any] = []
    for row in answer_rows:
        for cell in row:
            if cell is None or not str(cell).strip():
                continue
            if not (_value_keys(cell) & observed):
                unsupported.append(cell)
                if len(unsupported) >= 20:
                    return False, unsupported
    return not unsupported, unsupported


def _project_compute_answer(
    answer: Any,
    *,
    compute_columns: tuple[str, ...],
    compute_rows: tuple[tuple[Any, ...], ...],
) -> tuple[list[str], list[list[Any]]] | None:
    if not isinstance(answer, dict):
        return None
    columns = answer.get("columns")
    if not isinstance(columns, list) or not columns or "rows" in answer:
        return None
    requested = [str(column) for column in columns]
    index_by_name = {str(column): index for index, column in enumerate(compute_columns)}
    if not all(column in index_by_name for column in requested):
        return None
    indexes = [index_by_name[column] for column in requested]
    return requested, [[row[index] for index in indexes] for row in compute_rows]


def _source_negative_scope(
    *,
    kind: str,
    source: SourceRef | None = None,
    **extra: Any,
) -> dict[str, Any]:
    scope = {"kind": kind, **extra}
    if source is not None:
        scope.update({"source_id": source.id, "path": source.virtual_path, "data_form": source.data_form})
    return scope


def _string_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        cleaned = value.split("</parameter", 1)[0].strip()
        if not cleaned:
            return ()
        if cleaned.startswith("["):
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return tuple(str(item) for item in parsed if str(item).strip())
            except json.JSONDecodeError:
                pass
        refs = re.findall(r"\b(?:src|ev|bind|comp|req|sec|cand|rel)_\d{4}\b", cleaned)
        if refs:
            return tuple(refs)
        if "," in cleaned:
            return tuple(part.strip() for part in cleaned.split(",") if part.strip())
        return (cleaned,)
    if isinstance(value, list | tuple):
        items: list[str] = []
        for item in value:
            items.extend(_string_list(item))
        return tuple(dict.fromkeys(items))
    return ()


def _source_ids_from_refs(state: LoopState, refs: tuple[str, ...]) -> tuple[str, ...]:
    source_ids: list[str] = []
    for ref in refs:
        source_id = ref if ref in state.sources else state.source_by_path.get(ref)
        if source_id and source_id in state.sources:
            source_ids.append(source_id)
    return tuple(dict.fromkeys(source_ids))


def _card_field_id(card: Any) -> str | None:
    if isinstance(card, dict):
        table = str(card.get("canonical_table") or "").strip()
        field = str(card.get("canonical_field") or "").strip()
    else:
        table = str(getattr(card, "canonical_table", "") or "").strip()
        field = str(getattr(card, "canonical_field", "") or "").strip()
    if not table or not field:
        return None
    return f"{table}.{field}".casefold()


def _enrich_document_agent_arguments(
    state: LoopState,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(arguments)
    card_by_id = {card.id: card for card in state.semantic_cards}
    card_by_field: dict[str, Any] = {}
    for card in state.semantic_cards:
        field_id = _card_field_id(card)
        if field_id and field_id not in card_by_field:
            card_by_field[field_id] = card

    raw_cards = [card for card in arguments.get("semantic_cards") or [] if isinstance(card, dict)]
    enriched_cards: list[Any] = []
    seen_card_ids: set[str] = set()
    for raw_card in raw_cards:
        full_card = card_by_id.get(str(raw_card.get("id") or ""))
        if full_card is None:
            field_id = _card_field_id(raw_card)
            full_card = card_by_field.get(field_id or "")
        if full_card is None:
            enriched_cards.append(raw_card)
            continue
        if full_card.id in seen_card_ids:
            continue
        seen_card_ids.add(full_card.id)
        enriched_cards.append(full_card)

    source_ids = _source_ids_from_refs(state, _string_list(arguments.get("source_candidates")))
    document_field_ids: set[str] = set()
    document_card_ids: set[str] = set()
    if source_ids:
        for mapping in state.source_mappings:
            if mapping.source_id not in source_ids or mapping.status != "document_source":
                continue
            card = card_by_id.get(mapping.card_id)
            field_id = _card_field_id(card) if card is not None else None
            if not field_id:
                continue
            document_field_ids.add(field_id)
            document_card_ids.add(mapping.card_id)

    if document_field_ids:
        raw_target_fields = _string_list(arguments.get("target_fields"))
        allowed_field_names = {
            str(card_by_id[card_id].canonical_field)
            for card_id in document_card_ids
            if card_id in card_by_id and card_by_id[card_id].canonical_field
        }
        target_fields = [
            field
            for field in raw_target_fields
            if field.casefold() in {item.casefold() for item in allowed_field_names}
        ]
        if not target_fields:
            target_fields = sorted(allowed_field_names)
        normalized["target_fields"] = target_fields

        filtered_cards: list[Any] = []
        seen_filtered_ids: set[str] = set()
        for card in enriched_cards or [card_by_id[card_id] for card_id in document_card_ids if card_id in card_by_id]:
            field_id = _card_field_id(card)
            card_id = str(card.get("id", "") if isinstance(card, dict) else getattr(card, "id", ""))
            if field_id not in document_field_ids:
                continue
            if card_id and card_id in seen_filtered_ids:
                continue
            if card_id:
                seen_filtered_ids.add(card_id)
            filtered_cards.append(card)
        enriched_cards = filtered_cards

    if enriched_cards:
        normalized["semantic_cards"] = [
            card.to_dict() if hasattr(card, "to_dict") else dict(card)
            for card in enriched_cards
        ]
    return normalized


def _source_evidence_refs(state: LoopState, source_ids: tuple[str, ...]) -> tuple[str, ...]:
    refs: list[str] = []
    for evidence in state.evidence.values():
        if evidence.source_id in source_ids and evidence.ok and evidence.tool_name not in {
            "verify_alignment",
            "track_requirements",
            "bind",
            "submit_final",
        }:
            refs.append(evidence.id)
    return tuple(dict.fromkeys(refs))


def _compute_has_candidate_answer_verification(state: LoopState, compute_ref: str) -> bool:
    for evidence in state.evidence.values():
        if evidence.tool_name != "verify_alignment" or not evidence.ok:
            continue
        payload = evidence.payload or {}
        if payload.get("decision") != "candidate_answer":
            continue
        if payload.get("target_kind") not in {"compute_result", "final_answer"}:
            continue
        refs = {
            str(ref)
            for ref in [
                *(payload.get("compute_refs") or []),
                *(payload.get("target_refs") or []),
            ]
        }
        if compute_ref in refs:
            return True
    return False


class EvidenceActionRegistry:
    def __init__(self, *, document_agent: DocumentAgent | None = None) -> None:
        self.document_agent = document_agent or DocumentAgent()
        self._dispatch: dict[str, Callable[[LoopState, dict[str, Any]], Evidence]] = {
            "list_inventory": self._list_inventory,
            "retrieve_knowledge": self._retrieve_knowledge,
            "locate_sources": self._locate_sources,
            "inspect_source": self._inspect_source,
            "sample_records": self._sample_records,
            "search_values": self._search_values,
            "run_document_agent": self._run_document_agent,
            "inspect_relation": self._inspect_relation,
            "discover_join_paths": self._discover_join_paths,
            "track_requirements": self._track_requirements,
            "verify_alignment": self._verify_alignment,
            "run_verified_compute": self._run_verified_compute,
            "submit_final": self._submit_final,
            "inspect_video": self._inspect_video,
            "extract_video_observations": self._extract_video_observations,
        }
        self._specs = {
            "list_inventory": ToolSpec("list_inventory", "Return observed context inventory."),
            "retrieve_knowledge": ToolSpec(
                "retrieve_knowledge", "Navigate knowledge catalog and return complete slices."
            ),
            "locate_sources": ToolSpec(
                "locate_sources", "Find source/table/field candidates by lexical evidence."
            ),
            "inspect_source": ToolSpec(
                "inspect_source", "Inspect schema/header/key shape/sample for an observed source."
            ),
            "sample_records": ToolSpec(
                "sample_records", "Read a small bounded sample from an observed structured source."
            ),
            "search_values": ToolSpec(
                "search_values", "Search literal values across observed sources."
            ),
            "run_document_agent": ToolSpec(
                "run_document_agent", "Delegate PDF/MD evidence work to DocumentAgent."
            ),
            "inspect_relation": ToolSpec(
                "inspect_relation", "Inspect verified relation schema/sample before compute or SQL repair."
            ),
            "discover_join_paths": ToolSpec(
                "discover_join_paths", "Inspect verified relations for generic join candidates."
            ),
            "track_requirements": ToolSpec(
                "track_requirements", "Maintain a generic requirement coverage ledger."
            ),
            "verify_alignment": ToolSpec(
                "verify_alignment", "Record structured verifier decisions over cited evidence."
            ),
            "run_verified_compute": ToolSpec(
                "run_verified_compute", "Run SQL over verified relation bindings."
            ),
            "submit_final": ToolSpec("submit_final", "Materialize a final answer from compute output."),
            "inspect_video": ToolSpec("inspect_video", "Return v1 unsupported video metadata."),
            "extract_video_observations": ToolSpec(
                "extract_video_observations", "Video extraction placeholder; unsupported in v1."
            ),
        }

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._dispatch)

    def spec(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def dispatch(self, state: LoopState, action: ModelAction) -> Evidence:
        if action.kind == "compute":
            return self._run_verified_compute(
                state,
                {"sql": action.sql, "binding_refs": list(action.binding_refs)},
            )
        if action.kind == "final":
            arguments = dict(action.arguments)
            if action.compute_ref:
                arguments["compute_ref"] = action.compute_ref
            if action.answer is not None:
                arguments["answer"] = action.answer
            if action.binding_refs:
                arguments["binding_refs"] = list(action.binding_refs)
            if action.evidence_refs:
                arguments["evidence_refs"] = list(action.evidence_refs)
            return self._submit_final(state, arguments)
        if action.tool_name not in self._dispatch:
            return state.add_evidence(
                tool_name=str(action.tool_name),
                ok=False,
                summary=f"Unknown tool: {action.tool_name}",
                payload={"error": "unknown_tool"},
                negative_scope={"kind": "unknown_tool", "tool_name": action.tool_name},
                allowed_next_tools=self.tool_names,
            )
        return self._dispatch[action.tool_name](state, action.arguments)

    def _list_inventory(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        del arguments
        sources = [source.public_dict() for source in state.sources.values()]
        return state.add_evidence(
            tool_name="list_inventory",
            ok=True,
            summary=f"Observed {len(sources)} context source(s).",
            payload={"sources": sources},
            allowed_next_tools=("retrieve_knowledge", "locate_sources", "inspect_source"),
        )

    def _retrieve_knowledge(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        mode = str(arguments.get("mode") or "").strip().casefold()
        raw_query = arguments.get("query")
        query = str(raw_query or "").strip()
        section_ids = {
            str(item)
            for item in arguments.get("section_ids", [])
            if str(item).strip()
        }
        card_ids = {
            str(item)
            for item in arguments.get("card_ids", [])
            if str(item).strip()
        }
        tokens = [str(item).strip() for item in arguments.get("tokens", []) if str(item).strip()]
        include_neighbors = bool(arguments.get("include_neighbors", False))
        try:
            limit = int(arguments.get("limit") or 8)
        except (TypeError, ValueError):
            limit = 8
        limit = min(max(limit, 1), 80)
        sections = state.knowledge_sections
        section_by_id = {section.id: section for section in sections}
        card_by_id = {card.id: card for card in state.semantic_cards}

        def catalog_entry(section: Any) -> dict[str, Any]:
            preview = " ".join(str(section.text or "").split())
            if len(preview) > 240:
                preview = preview[:237].rstrip() + "..."
            return {
                "id": section.id,
                "heading_path": section.heading_path,
                "line_start": section.line_start,
                "line_end": section.line_end,
                "mention_count": len(section.mentions),
                "mentions": list(section.mentions[:12]),
                "preview": preview,
            }

        def full_section_payload(section: Any) -> dict[str, Any]:
            return {
                "id": section.id,
                "heading_path": section.heading_path,
                "line_start": section.line_start,
                "line_end": section.line_end,
                "text": section.text,
                "mentions": list(section.mentions),
            }

        def add_neighbors(ids: set[str]) -> set[str]:
            if not include_neighbors:
                return ids
            ordered_ids = [section.id for section in sections]
            expanded = set(ids)
            for section_id in list(ids):
                try:
                    index = ordered_ids.index(section_id)
                except ValueError:
                    continue
                for neighbor_index in (index - 1, index + 1):
                    if 0 <= neighbor_index < len(ordered_ids):
                        expanded.add(ordered_ids[neighbor_index])
            return expanded

        def ordered_sections(ids: set[str]) -> list[Any]:
            return [section for section in sections if section.id in ids]

        def lookup_key(token: str) -> str:
            return normalize_key(token) or token.casefold()

        def lookup_entry(token: str) -> Any | None:
            key = lookup_key(token)
            if key in state.knowledge_lookup:
                return state.knowledge_lookup[key]
            token_casefold = token.casefold()
            for entry in state.knowledge_lookup.values():
                if entry.token.casefold() == token_casefold:
                    return entry
            return None

        catalog = {
            "section_count": len(sections),
            "lookup_count": len(state.knowledge_lookup),
            "semantic_card_count": len(state.semantic_cards),
            "sections": [catalog_entry(section) for section in sections],
            "semantic_cards": [
                {
                    "id": card.id,
                    "kind": card.kind,
                    "name": card.name,
                    "canonical_table": card.canonical_table,
                    "canonical_field": card.canonical_field,
                    "unit": card.unit,
                    "record_grain": card.record_grain,
                }
                for card in state.semantic_cards[:160]
            ],
            "lookup_tokens": [
                {
                    "token": entry.token,
                    "section_refs": list(entry.section_refs),
                    "status": entry.status,
                    "must_verify": entry.must_verify,
                }
                for entry in list(state.knowledge_lookup.values())[:160]
            ],
        }
        if mode == "semantic" or card_ids:
            selected_cards = [card_by_id[card_id] for card_id in card_ids if card_id in card_by_id]
            missing_card_ids = sorted(card_id for card_id in card_ids if card_id not in card_by_id)
            if not selected_cards:
                semantic_query = " ".join([query or state.question, *tokens]).strip()
                if semantic_query:
                    query_terms = set(_terms(semantic_query))
                    scored: list[tuple[int, str, Any]] = []
                    for card in state.semantic_cards:
                        haystack = " ".join(
                            [
                                card.name,
                                card.kind,
                                card.canonical_table,
                                card.canonical_field or "",
                                card.definition,
                                *card.aliases,
                            ]
                        )
                        overlap = len(query_terms & set(_terms(haystack)))
                        if overlap > 0:
                            scored.append((-overlap, card.id, card))
                    scored.sort(key=lambda item: (item[0], item[1]))
                    selected_cards = [card for _score, _id, card in scored[:limit]]
                else:
                    selected_cards = state.matched_semantic_cards[:limit] or state.semantic_cards[:limit]
            return state.add_evidence(
                tool_name="retrieve_knowledge",
                ok=bool(state.semantic_cards),
                summary=f"Knowledge semantic: returned {len(selected_cards)} semantic card(s).",
                payload={
                    "mode": "semantic",
                    "cards": [card.to_dict() for card in selected_cards],
                    "missing_card_ids": missing_card_ids,
                    "card_count": len(state.semantic_cards),
                    "usage_note": (
                        "Semantic cards define canonical meaning, aliases, units, grain, formulas, and ambiguity rules. "
                        "They do not define physical source format; use locate_sources for source candidates."
                    ),
                },
                negative_scope={"kind": "missing_semantic_knowledge"} if not state.semantic_cards else None,
                allowed_next_tools=("retrieve_knowledge", "locate_sources", "inspect_source", "run_document_agent"),
            )
        if not sections:
            return state.add_evidence(
                tool_name="retrieve_knowledge",
                ok=False,
                summary="No knowledge.md sections are available.",
                payload={
                    "mode": mode or "catalog",
                    "catalog": catalog,
                    "usage_note": "knowledge.md is missing, skipped, unreadable, or empty.",
                },
                negative_scope={"kind": "missing_knowledge"},
                allowed_next_tools=("locate_sources", "inspect_source", "run_document_agent"),
            )

        selected_ids: set[str] = set()
        missing_section_ids: list[str] = []
        missing_tokens: list[str] = []
        resolved_tokens: list[dict[str, Any]] = []

        if section_ids:
            mode = mode or "section"
            missing_section_ids = sorted(section_id for section_id in section_ids if section_id not in section_by_id)
            selected_ids.update(section_id for section_id in section_ids if section_id in section_by_id)
        if tokens:
            mode = mode or "token"
            for token in tokens:
                entry = lookup_entry(token)
                if entry is None:
                    missing_tokens.append(token)
                    continue
                refs = [ref for ref in entry.section_refs if ref in section_by_id]
                selected_ids.update(refs)
                resolved_tokens.append(
                    {
                        "query_token": token,
                        "token": entry.token,
                        "section_refs": refs,
                        "evidence_refs": list(entry.evidence_refs),
                        "status": entry.status,
                        "must_verify": entry.must_verify,
                    }
                )
        if mode == "catalog" or (not mode and not query and not selected_ids):
            selected: list[Any] = []
            actual_mode = "catalog"
        elif selected_ids:
            selected = ordered_sections(add_neighbors(selected_ids))
            actual_mode = mode or "section"
        else:
            actual_mode = "search"
            query = query or state.question
            query_terms = set(_terms(query))
            lookup_hits: set[str] = set()
            for term in query_terms:
                entry = lookup_entry(term)
                if entry is not None:
                    lookup_hits.update(ref for ref in entry.section_refs if ref in section_by_id)
            matched = [
                (
                    len(query_terms & set(_terms(f"{section.heading_path}\n{section.text}")))
                    + (3 if section.id in lookup_hits else 0),
                    section.line_start or 10**9,
                    section,
                )
                for section in sections
            ]
            selected = [
                section
                for overlap, _line, section in sorted(matched, key=lambda item: (-item[0], item[1]))
                if overlap > 0
            ][:8]
            if not selected:
                actual_mode = "catalog_fallback"
        payload = {
            "mode": actual_mode,
            "catalog": catalog if actual_mode in {"catalog", "catalog_fallback"} else {
                "section_count": catalog["section_count"],
                "lookup_count": catalog["lookup_count"],
            },
            "resolved_tokens": resolved_tokens,
            "missing_tokens": missing_tokens,
            "missing_section_ids": missing_section_ids,
            "sections": [full_section_payload(section) for section in selected],
            "usage_note": (
                "Returned section text is document-only semantic evidence. Mentions are lookup "
                "tokens and still require observed physical source verification."
            ),
        }
        if actual_mode == "catalog_fallback":
            payload["search_notice"] = (
                "No lexical section matched the query. Use the catalog/lookup tokens to choose "
                "a candidate section, then call retrieve_knowledge with section_ids or tokens."
            )
        return state.add_evidence(
            tool_name="retrieve_knowledge",
            ok=True,
            summary=f"Knowledge {actual_mode}: returned {len(selected)} full section(s).",
            payload=payload,
            allowed_next_tools=("retrieve_knowledge", "locate_sources", "inspect_source", "run_document_agent"),
        )

    def _locate_sources(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        query = str(arguments.get("query") or state.question)
        tokens = [str(token) for token in arguments.get("tokens", []) if str(token).strip()]
        if not tokens:
            tokens = _terms(query)
        query_norms = {_normalize(token) for token in tokens + [query] if _normalize(token)}
        candidates = []
        semantic_plan: list[dict[str, Any]] = []
        matched_card_ids = {card.id for card in state.matched_semantic_cards}
        seen_semantic_candidates: set[tuple[str, str, str]] = set()

        for card in state.semantic_cards:
            canonical_field = str(card.canonical_field or "").strip()
            if not canonical_field:
                continue
            canonical_id = f"{card.canonical_table}.{canonical_field}".casefold()
            card_haystack = " ".join(
                [
                    card.name,
                    card.kind,
                    card.canonical_table,
                    canonical_field,
                    card.definition,
                    *(card.aliases or ()),
                ]
            )
            card_norm = _normalize(card_haystack)
            relevant = card.id in matched_card_ids or any(
                token and token in card_norm for token in query_norms
            )
            if not relevant:
                continue
            mappings = [mapping for mapping in state.source_mappings if mapping.card_id == card.id]
            if not mappings:
                continue
            mapping_payloads: list[dict[str, Any]] = []
            for mapping in mappings:
                preferred = mapping.status in {"exact_structured_source", "document_source"}
                mapping_payload = {
                    **mapping.to_dict(),
                    "canonical_field": canonical_id,
                    "binding_priority": "preferred" if preferred else "fallback_only",
                    "recommended_tool": (
                        "run_document_agent"
                        if mapping.status == "document_source"
                        else "inspect_source"
                        if mapping.status == "exact_structured_source"
                        else "retrieve_knowledge"
                    ),
                    "usage_note": (
                        "Preferred mapping for this canonical field."
                        if preferred
                        else "Fallback candidates are lexical hints only; do not bind as this canonical field without explicit semantic_contract proof."
                    ),
                }
                mapping_payloads.append(mapping_payload)
                if not mapping.source_id:
                    continue
                candidate_key = (card.id, mapping.source_id, mapping.status)
                if candidate_key in seen_semantic_candidates:
                    continue
                seen_semantic_candidates.add(candidate_key)
                source = state.sources.get(mapping.source_id)
                candidate = state.add_candidate(
                    kind="semantic_source_mapping",
                    source_id=mapping.source_id,
                    data_form=mapping.data_form or (source.data_form if source else "unknown_file"),
                    match_reason=f"semantic_{mapping.status}: {mapping.match_reason}",
                    path=mapping.source_path,
                    table=mapping.matched_table,
                    field=mapping.matched_field,
                ).to_dict()
                candidate.update(
                    {
                        "semantic_card_id": card.id,
                        "canonical_field": canonical_id,
                        "mapping_status": mapping.status,
                        "binding_priority": "preferred" if preferred else "fallback_only",
                        "recommended_tool": mapping_payload["recommended_tool"],
                        "usage_note": mapping_payload["usage_note"],
                    }
                )
                candidates.append(candidate)
            semantic_plan.append(
                {
                    "card_id": card.id,
                    "canonical_field": canonical_id,
                    "canonical_table": card.canonical_table,
                    "field": canonical_field,
                    "unit": card.unit,
                    "record_grain": card.record_grain,
                    "source_mappings": mapping_payloads,
                }
            )

        for source in state.sources.values():
            haystack = " ".join(
                [source.virtual_path, source.basename, source.stem, *source.tables, *source.columns]
            )
            haystack_norm = _normalize(haystack)
            source_norms = {_normalize(source.basename), _normalize(source.stem), _normalize(source.virtual_path)}
            matched_source = bool(query_norms & source_norms) or any(
                token and token in haystack_norm for token in query_norms
            )
            if matched_source or not tokens:
                candidate = state.add_candidate(
                    kind="source_candidate",
                    source_id=source.id,
                    data_form=source.data_form,
                    match_reason="lexical_inventory_match" if matched_source else "inventory_listing",
                    path=source.virtual_path,
                )
                candidates.append(candidate.to_dict())

            for table in source.tables:
                table_norm = _normalize(table)
                if table_norm in query_norms or any(token and token in table_norm for token in query_norms):
                    candidate = state.add_candidate(
                        kind="table_candidate",
                        source_id=source.id,
                        data_form="sqlite_database",
                        match_reason="observed_table_name_match",
                        path=source.virtual_path,
                        table=table,
                    )
                    candidates.append(candidate.to_dict())

            for column in source.columns:
                column_norm = _normalize(column)
                if column_norm in query_norms or any(token and token in column_norm for token in query_norms):
                    candidate = state.add_candidate(
                        kind="field_candidate",
                        source_id=source.id,
                        data_form=source.data_form,
                        match_reason="observed_field_name_match",
                        path=source.virtual_path,
                        field=column,
                    )
                    candidates.append(candidate.to_dict())

        return state.add_evidence(
            tool_name="locate_sources",
            ok=True,
            summary=(
                f"Located {len(candidates)} candidate(s). Semantic preferred mappings are listed "
                "before lexical candidates; candidates are not bindings."
            ),
            payload={
                "query": query,
                "tokens": tokens,
                "semantic_source_plan": semantic_plan[:24],
                "candidates": candidates[:80],
                "usage_note": (
                    "Use semantic_source_plan preferred mappings for knowledge-defined fields before "
                    "substituting similarly named tables/columns. Fallback candidates require explicit "
                    "semantic_contract proof."
                ),
            },
            negative_scope=(
                {"kind": "candidate_search_empty", "query": query, "tokens": tokens}
                if not candidates
                else None
            ),
            allowed_next_tools=("inspect_source", "sample_records", "run_document_agent"),
        )

    def _inspect_source(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        source = _source_from_args(state, arguments)
        if source is None:
            return state.add_evidence(
                tool_name="inspect_source",
                ok=False,
                summary="No observed source matched inspect_source arguments.",
                payload={"arguments": arguments},
                negative_scope={"kind": "unknown_source", "tool": "inspect_source", "arguments": arguments},
                allowed_next_tools=("list_inventory", "locate_sources"),
                recommended_next_actions=(
                    _recommend("list_inventory", {}, "Refresh observed source ids and paths."),
                ),
            )
        table = str(arguments.get("table") or "").strip() or None
        limit = int(arguments.get("limit") or 5)

        if source.data_form == "csv_records":
            frame = pd.read_csv(source.path, dtype=object, nrows=max(limit, 1))
            payload = {
                "source_id": source.id,
                "path": source.virtual_path,
                "data_form": source.data_form,
                "columns": [str(column) for column in frame.columns],
                "sample": _sample_dataframe(frame, limit=limit),
            }
            summary = f"CSV source {source.virtual_path} has {len(frame.columns)} observed column(s)."
            return state.add_evidence(
                tool_name="inspect_source",
                ok=True,
                summary=summary,
                payload=payload,
                source_id=source.id,
                data_form=source.data_form,
                allowed_next_tools=("bind", "sample_records", "search_values"),
            )

        if source.data_form == "json_records":
            payload_raw = json.loads(source.path.read_text(encoding="utf-8", errors="replace"))
            frame = pd.json_normalize(_json_records(payload_raw))
            payload = {
                "source_id": source.id,
                "path": source.virtual_path,
                "data_form": source.data_form,
                "columns": [str(column) for column in frame.columns],
                "sample": _sample_dataframe(frame, limit=limit),
            }
            return state.add_evidence(
                tool_name="inspect_source",
                ok=True,
                summary=f"JSON source {source.virtual_path} has {len(frame.columns)} observed key path(s).",
                payload=payload,
                source_id=source.id,
                data_form=source.data_form,
                allowed_next_tools=("bind", "sample_records", "search_values"),
            )

        if source.data_form == "sqlite_database":
            uri = f"file:{source.path.resolve().as_posix()}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                if table is None and len(source.tables) == 1:
                    table = source.tables[0]
                if table is None:
                    payload = {
                        "source_id": source.id,
                        "path": source.virtual_path,
                        "data_form": source.data_form,
                        "tables": list(source.tables),
                    }
                    return state.add_evidence(
                        tool_name="inspect_source",
                        ok=True,
                        summary=f"SQLite source {source.virtual_path} has {len(source.tables)} table(s).",
                        payload=payload,
                        source_id=source.id,
                        data_form=source.data_form,
                        allowed_next_tools=("inspect_source", "locate_sources"),
                    )
                if table not in source.tables:
                    return state.add_evidence(
                        tool_name="inspect_source",
                        ok=False,
                        summary=f"SQLite table {table} was not observed in {source.virtual_path}.",
                        payload={"source_id": source.id, "table": table, "tables": list(source.tables)},
                        source_id=source.id,
                        data_form=source.data_form,
                        negative_scope=_source_negative_scope(
                            kind="missing_table",
                            source=source,
                            table=table,
                            observed_tables=list(source.tables),
                        ),
                        allowed_next_tools=("inspect_source", "locate_sources", "sample_records"),
                        recommended_next_actions=(
                            _recommend(
                                "inspect_source",
                                {"source_ref": source.id},
                                "Inspect available tables before choosing a different table.",
                            ),
                            _recommend(
                                "locate_sources",
                                {"query": table or ""},
                                "Search other observed sources for the requested logical name.",
                            ),
                        ),
                    )
                schema_rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
                sample_rows = connection.execute(f'SELECT * FROM "{table}" LIMIT ?', (limit,)).fetchall()
                columns = [str(row[1]) for row in schema_rows]
                sample = [dict(zip(columns, row, strict=False)) for row in sample_rows]
            return state.add_evidence(
                tool_name="inspect_source",
                ok=True,
                summary=f"SQLite table {table} has {len(columns)} observed column(s).",
                payload={
                    "source_id": source.id,
                    "path": source.virtual_path,
                    "data_form": source.data_form,
                    "table": table,
                    "columns": columns,
                    "schema": [
                        {"name": str(row[1]), "type": str(row[2]), "notnull": bool(row[3])}
                        for row in schema_rows
                    ],
                    "sample": sample,
                },
                source_id=source.id,
                data_form=source.data_form,
                allowed_next_tools=("bind", "sample_records", "search_values"),
            )

        if source.data_form in {"pdf_document", "markdown_document"}:
            return state.add_evidence(
                tool_name="inspect_source",
                ok=True,
                summary=f"{source.data_form} source observed. Use run_document_agent for record-slice evidence.",
                payload={
                    "source_id": source.id,
                    "path": source.virtual_path,
                    "data_form": source.data_form,
                    "size_bytes": source.size_bytes,
                },
                source_id=source.id,
                data_form=source.data_form,
                allowed_next_tools=("run_document_agent",),
            )

        if source.data_form == "video":
            return self._inspect_video(state, {"source_ref": source.id})

        return state.add_evidence(
            tool_name="inspect_source",
            ok=False,
            summary=f"Unsupported source data form: {source.data_form}.",
            payload={"source_id": source.id, "path": source.virtual_path, "data_form": source.data_form},
            source_id=source.id,
            data_form=source.data_form,
            negative_scope=_source_negative_scope(kind="unsupported_data_form", source=source),
            allowed_next_tools=("locate_sources", "blocked"),
        )

    def _sample_records(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        arguments = dict(arguments)
        arguments.setdefault("limit", 10)
        evidence = self._inspect_source(state, arguments)
        if evidence.ok and evidence.payload.get("sample") is not None:
            return state.add_evidence(
                tool_name="sample_records",
                ok=True,
                summary=f"Sampled records from {evidence.payload.get('path')}.",
                payload=evidence.payload,
                source_id=evidence.source_id,
                data_form=evidence.data_form,
            )
        return evidence

    def _search_values(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        value = str(
            arguments.get("value")
            or arguments.get("query")
            or arguments.get("search_pattern")
            or arguments.get("value_pattern")
            or ""
        ).strip()
        if not value:
            return state.add_evidence(
                tool_name="search_values",
                ok=False,
                summary="search_values requires a literal value/query.",
                payload={"arguments": arguments},
                negative_scope={"kind": "invalid_tool_arguments", "tool": "search_values", "missing": "value"},
                allowed_next_tools=("sample_records", "inspect_source", "blocked"),
            )
        scoped = _source_from_args(state, arguments)
        sources = [scoped] if scoped is not None else list(state.sources.values())
        limit = int(arguments.get("limit") or 20)
        hits: list[dict[str, Any]] = []
        needle = value.casefold()

        for source in sources:
            if len(hits) >= limit:
                break
            try:
                if source.data_form == "csv_records":
                    frame = pd.read_csv(source.path, dtype=str).fillna("")
                    mask = frame.apply(lambda column: column.str.casefold().str.contains(needle, regex=False))
                    for row_index in mask.any(axis=1)[mask.any(axis=1)].index[: limit - len(hits)]:
                        hits.append(
                            {
                                "source_id": source.id,
                                "path": source.virtual_path,
                                "row_index": int(row_index),
                                "record": frame.iloc[row_index].to_dict(),
                            }
                        )
                elif source.data_form == "json_records":
                    payload = json.loads(source.path.read_text(encoding="utf-8", errors="replace"))
                    frame = pd.json_normalize(_json_records(payload)).astype(str)
                    mask = frame.apply(lambda column: column.str.casefold().str.contains(needle, regex=False))
                    for row_index in mask.any(axis=1)[mask.any(axis=1)].index[: limit - len(hits)]:
                        hits.append(
                            {
                                "source_id": source.id,
                                "path": source.virtual_path,
                                "row_index": int(row_index),
                                "record": frame.iloc[row_index].to_dict(),
                            }
                        )
                elif source.data_form == "sqlite_database":
                    uri = f"file:{source.path.resolve().as_posix()}?mode=ro"
                    with closing(sqlite3.connect(uri, uri=True)) as connection:
                        for table in source.tables:
                            schema_rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
                            columns = [str(row[1]) for row in schema_rows]
                            for column in columns:
                                rows = connection.execute(
                                    f'SELECT * FROM "{table}" WHERE CAST("{column}" AS TEXT) LIKE ? LIMIT ?',
                                    (f"%{value}%", max(1, limit - len(hits))),
                                ).fetchall()
                                for row in rows:
                                    hits.append(
                                        {
                                            "source_id": source.id,
                                            "path": source.virtual_path,
                                            "table": table,
                                            "matched_column": column,
                                            "record": dict(zip(columns, row, strict=False)),
                                        }
                                    )
                                    if len(hits) >= limit:
                                        break
                                if len(hits) >= limit:
                                    break
                            if len(hits) >= limit:
                                break
                elif source.data_form in {"pdf_document", "markdown_document"}:
                    for line in _document_lines(source):
                        if needle in str(line["text"]).casefold():
                            hits.append(
                                {
                                    "source_id": source.id,
                                    "path": source.virtual_path,
                                    "page": line["page"],
                                    "line": line["line"],
                                    "text": line["text"],
                                }
                            )
                            if len(hits) >= limit:
                                break
            except Exception as exc:  # noqa: BLE001 - value search should keep scanning other sources
                hits.append(
                    {
                        "source_id": source.id,
                        "path": source.virtual_path,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        return state.add_evidence(
            tool_name="search_values",
            ok=True,
            summary=f"Found {len(hits)} value hit(s) for query.",
            payload={
                "value": value,
                "hits": hits,
            },
            negative_scope=(
                {"kind": "value_not_found", "value": value, "source_id": scoped.id if scoped else None}
                if not hits
                else None
            ),
            allowed_next_tools=(
                "inspect_source",
                "sample_records",
                "search_values",
                "verify_alignment",
                "bind",
                "locate_sources",
            ),
            recommended_next_actions=(),
        )

    def _run_document_agent(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        arguments = _enrich_document_agent_arguments(state, arguments)
        task = DocTask.from_arguments(arguments, fallback_question=state.question)
        package = self.document_agent.run(state, task)
        payload = package.to_dict()
        records = payload.get("records") if isinstance(payload.get("records"), list) else []
        source_refs = payload.get("source_refs") if isinstance(payload.get("source_refs"), list) else []
        source_id = str(source_refs[0]) if len(source_refs) == 1 else None
        source = state.sources.get(source_id or "")
        partial_coverage = bool(payload.get("remaining_risks")) or bool(payload.get("ambiguous_slice_ids"))
        payload["partial_coverage"] = partial_coverage
        payload["doc_task"] = task.to_dict()
        evidence_ref = f"ev_{state._evidence_seq + 1:04d}"
        summary = (
            f"DocumentAgent returned {len(records)} validated record(s) "
            f"from {len(source_refs)} document source(s)."
        )
        return state.add_evidence(
            tool_name="run_document_agent",
            ok=True,
            summary=summary,
            payload=payload,
            source_id=source_id,
            data_form=source.data_form if source is not None else None,
            negative_scope=(
                {
                    "kind": "document_agent_remaining_risk",
                    "remaining_risks": payload.get("remaining_risks") or [],
                }
                if payload.get("remaining_risks")
                else None
            ),
            allowed_next_tools=("verify_alignment", "bind", "run_document_agent", "locate_sources", "blocked"),
            recommended_next_actions=(
                {
                    "tool_name": "bind",
                    "arguments": {
                        "binding_type": "document_record_set",
                        "evidence_refs": [evidence_ref],
                        "allowed_columns": list((records[0] or {}).keys()) if records else [],
                        "alignment": "Bind validated DocumentAgent records if they satisfy the target semantic fields.",
                    },
                    "reason": "Bind the document record set only when records and coverage are sufficient.",
                },
            )
            if records
            else (),
        )

    def _track_requirements(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        raw_requirements = arguments.get("requirements")
        if not isinstance(raw_requirements, list) or not raw_requirements:
            return state.add_evidence(
                tool_name="track_requirements",
                ok=False,
                summary="track_requirements requires at least one requirement item.",
                payload={"arguments": arguments},
                negative_scope={"kind": "invalid_tool_arguments", "tool": "track_requirements"},
                allowed_next_tools=("track_requirements", "retrieve_knowledge", "locate_sources"),
            )

        upserted = []
        invalid_refs: list[dict[str, Any]] = []
        valid_statuses = {"pending", "satisfied", "not_applicable", "conflict", "blocked"}
        for raw in raw_requirements:
            if not isinstance(raw, dict):
                invalid_refs.append({"item": raw, "error": "requirement_item_not_object"})
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                invalid_refs.append({"item": raw, "error": "missing_text"})
                continue
            requirement_id = str(raw.get("id") or "").strip() or None
            status = str(raw.get("status") or "pending").strip()
            if status not in valid_statuses:
                invalid_refs.append({"item": raw, "error": f"invalid_status:{status}"})
                continue
            source_refs = _string_list(raw.get("source_refs"))
            evidence_refs = _string_list(raw.get("evidence_refs"))
            binding_refs = _string_list(raw.get("binding_refs"))
            compute_refs = _string_list(raw.get("compute_refs"))

            known_section_ids = {section.id for section in state.knowledge_sections}
            unknown_sources = [
                ref for ref in source_refs
                if (
                    (ref.startswith("src_") and ref not in state.sources)
                    or (ref.startswith("sec_") and ref not in known_section_ids)
                )
            ]
            unknown_evidence = [ref for ref in evidence_refs if ref not in state.evidence]
            unknown_bindings = [ref for ref in binding_refs if ref not in state.bindings]
            unknown_computes = [ref for ref in compute_refs if ref not in state.compute_results]
            if unknown_sources or unknown_evidence or unknown_bindings or unknown_computes:
                invalid_refs.append(
                    {
                        "requirement_id": requirement_id,
                        "text": text,
                        "unknown_sources": unknown_sources,
                        "unknown_evidence": unknown_evidence,
                        "unknown_bindings": unknown_bindings,
                        "unknown_computes": unknown_computes,
                    }
                )
                continue
            if status == "satisfied" and not (evidence_refs or binding_refs or compute_refs):
                invalid_refs.append(
                    {
                        "requirement_id": requirement_id,
                        "text": text,
                        "error": "satisfied_requirement_needs_lineage",
                    }
                )
                continue

            requirement = state.upsert_requirement(
                requirement_id=requirement_id,
                text=text,
                status=status,
                source_refs=source_refs,
                evidence_refs=evidence_refs,
                binding_refs=binding_refs,
                compute_refs=compute_refs,
                note=str(raw.get("note") or ""),
            )
            upserted.append(requirement.to_dict())

        if invalid_refs:
            return state.add_evidence(
                tool_name="track_requirements",
                ok=False,
                summary="track_requirements rejected invalid requirement items or references.",
                payload={
                    "updated_requirements": upserted,
                    "invalid_items": invalid_refs,
                    "requirements": [item.to_dict() for item in state.requirements.values()],
                },
                negative_scope={
                    "kind": "invalid_requirement_update",
                    "invalid_count": len(invalid_refs),
                },
                allowed_next_tools=("track_requirements", "retrieve_knowledge", "locate_sources"),
            )

        return state.add_evidence(
            tool_name="track_requirements",
            ok=True,
            summary=f"Updated {len(upserted)} requirement(s).",
            payload={
                "updated_requirements": upserted,
                "requirements": [item.to_dict() for item in state.requirements.values()],
            },
            allowed_next_tools=(
                "retrieve_knowledge",
                "locate_sources",
                "inspect_source",
                "run_document_agent",
                "verify_alignment",
                "bind",
                "run_verified_compute",
                "submit_final",
                "blocked",
            ),
        )

    def _verify_alignment(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        decision = str(arguments.get("decision") or "").strip()
        target_kind = str(arguments.get("target_kind") or "").strip()
        alignment = str(arguments.get("alignment") or "").strip()
        evidence_refs = _string_list(arguments.get("evidence_refs"))
        binding_refs = _string_list(arguments.get("binding_refs"))
        compute_refs = _string_list(arguments.get("compute_refs"))
        requirement_refs = _string_list(arguments.get("requirement_refs"))
        knowledge_section_ids = _string_list(arguments.get("knowledge_section_ids"))
        target_refs = _string_list(arguments.get("target_refs"))
        limitations = str(arguments.get("limitations") or "").strip()
        next_actions = arguments.get("next_actions")
        if not isinstance(next_actions, list):
            next_actions = []

        evidence_refs = tuple(
            dict.fromkeys(
                [
                    *evidence_refs,
                    *(ref for ref in target_refs if ref in state.evidence),
                ]
            )
        )
        binding_refs = tuple(
            dict.fromkeys(
                [
                    *binding_refs,
                    *(ref for ref in target_refs if ref in state.bindings),
                ]
            )
        )
        compute_refs = tuple(
            dict.fromkeys(
                [
                    *compute_refs,
                    *(ref for ref in target_refs if ref in state.compute_results),
                ]
            )
        )
        requirement_refs = tuple(
            dict.fromkeys(
                [
                    *requirement_refs,
                    *(ref for ref in target_refs if ref in state.requirements),
                ]
            )
        )
        knowledge_section_ids = tuple(
            dict.fromkeys(
                [
                    *knowledge_section_ids,
                    *(
                        ref
                        for ref in target_refs
                        if any(section.id == ref for section in state.knowledge_sections)
                    ),
                ]
            )
        )
        if decision in {"bindable", "candidate_answer"} and not evidence_refs:
            source_refs = tuple(ref for ref in target_refs if ref in state.sources)
            if source_refs:
                evidence_refs = _source_evidence_refs(state, source_refs)

        valid_decisions = {
            "bindable",
            "candidate_answer",
            "intermediate",
            "not_applicable",
            "needs_more_evidence",
            "conflict",
            "blocked_ok",
        }
        valid_target_kinds = {
            "source",
            "field",
            "document_window",
            "document_record_set",
            "compute_result",
            "direct_evidence",
            "alternative_source",
            "requirement",
            "blocked",
            "final_answer",
        }
        invalid: list[str] = []
        if decision not in valid_decisions:
            invalid.append(f"invalid_decision:{decision}")
        if target_kind not in valid_target_kinds:
            invalid.append(f"invalid_target_kind:{target_kind}")
        if not alignment:
            invalid.append("missing_alignment")

        unknown_evidence = [ref for ref in evidence_refs if ref not in state.evidence]
        unknown_bindings = [ref for ref in binding_refs if ref not in state.bindings]
        unknown_computes = [ref for ref in compute_refs if ref not in state.compute_results]
        unknown_requirements = [ref for ref in requirement_refs if ref not in state.requirements]
        unknown_sections = [
            ref for ref in knowledge_section_ids
            if not any(section.id == ref for section in state.knowledge_sections)
        ]
        if unknown_evidence:
            invalid.append("unknown_evidence:" + ",".join(unknown_evidence))
        if unknown_bindings:
            invalid.append("unknown_bindings:" + ",".join(unknown_bindings))
        if unknown_computes:
            invalid.append("unknown_computes:" + ",".join(unknown_computes))
        if unknown_requirements:
            invalid.append("unknown_requirements:" + ",".join(unknown_requirements))
        if unknown_sections:
            invalid.append("unknown_knowledge_sections:" + ",".join(unknown_sections))

        failed_evidence = [
            ref for ref in evidence_refs
            if ref in state.evidence and not state.evidence[ref].ok
        ]
        failed_computes = [
            ref for ref in compute_refs
            if ref in state.compute_results and not state.compute_results[ref].ok
        ]
        if decision in {"bindable", "candidate_answer"}:
            if not (evidence_refs or binding_refs or compute_refs):
                invalid.append("positive_verification_needs_lineage")
            if failed_evidence:
                invalid.append("positive_verification_cites_failed_evidence:" + ",".join(failed_evidence))
            if failed_computes:
                invalid.append("positive_verification_cites_failed_compute:" + ",".join(failed_computes))
        if decision == "candidate_answer":
            empty_computes = [
                ref for ref in compute_refs
                if ref in state.compute_results and not state.compute_results[ref].rows
            ]
            if empty_computes:
                invalid.append("candidate_answer_cites_empty_compute:" + ",".join(empty_computes))
        if decision == "blocked_ok" and not (evidence_refs or limitations or state.negative_scopes):
            invalid.append("blocked_ok_needs_evidence_limitations_or_negative_scope")

        recommended_actions = [
            action for action in next_actions if isinstance(action, dict)
        ][:8]
        if decision == "candidate_answer" and compute_refs:
            recommended_actions.append(
                _recommend(
                    "submit_final",
                    {"compute_ref": compute_refs[0]},
                    "Submit this verified compute candidate if it is the final requested answer.",
                )
            )
        elif decision == "candidate_answer" and binding_refs:
            binding = state.bindings.get(binding_refs[0])
            recommended_actions.append(
                _recommend(
                    "submit_final",
                    {
                        "binding_refs": [binding_refs[0]],
                        "evidence_refs": list(binding.evidence_refs) if binding else list(evidence_refs),
                        "answer": {},
                    },
                    "Submit a direct answer table only if the verified binding fully supports the answer.",
                )
            )
        elif decision == "bindable" and evidence_refs:
            recommended_actions.append(
                _recommend(
                    "bind",
                    {"evidence_refs": list(evidence_refs), "alignment": alignment},
                    "Create a binding with the appropriate generic binding_type if this evidence is ready to execute or finalize.",
                )
            )
        elif decision in {"intermediate", "needs_more_evidence"}:
            if target_kind in {"document_window", "direct_evidence"} and evidence_refs:
                recommended_actions.append(
                    _recommend(
                        "run_document_agent",
                        {"question": state.question},
                        "Delegate remaining PDF/MD evidence work to DocumentAgent instead of extracting records in the main loop.",
                    )
                )
            if requirement_refs:
                recommended_actions.append(
                    _recommend(
                        "track_requirements",
                        {
                            "requirements": [
                                {
                                    "id": ref,
                                    "text": state.requirements[ref].text,
                                    "status": "pending",
                                    "note": limitations or alignment,
                                }
                                for ref in requirement_refs
                                if ref in state.requirements
                            ]
                        },
                        "Keep unresolved requirements explicit before gathering more evidence.",
                    )
                )
            recommended_actions.append(
                _recommend(
                    "locate_sources",
                    {"query": state.question},
                    "Search for alternative observed sources when current evidence is only intermediate.",
                )
            )
        elif decision == "blocked_ok":
            recommended_actions.append(
                _recommend(
                    "blocked",
                    {"reason": alignment, "evidence_refs": list(evidence_refs)},
                    "Stop with cited evidence if no valid evidence path remains.",
                )
            )
        elif decision in {"not_applicable", "conflict"}:
            recommended_actions.append(
                _recommend(
                    "locate_sources",
                    {"query": state.question},
                    "Switch to alternative sources or evidence after rejecting this target.",
                )
            )

        payload = {
            "decision": decision,
            "target_kind": target_kind,
            "target_refs": list(target_refs),
            "requirement_refs": list(requirement_refs),
            "knowledge_section_ids": list(knowledge_section_ids),
            "evidence_refs": list(evidence_refs),
            "binding_refs": list(binding_refs),
            "compute_refs": list(compute_refs),
            "alignment": alignment,
            "limitations": limitations,
            "next_actions": recommended_actions,
        }
        if invalid:
            return state.add_evidence(
                tool_name="verify_alignment",
                ok=False,
                summary="Verifier decision rejected: " + "; ".join(invalid),
                payload={**payload, "invalid": invalid},
                negative_scope={
                    "kind": "invalid_verifier_decision",
                    "decision": decision,
                    "target_kind": target_kind,
                    "invalid": invalid,
                },
                allowed_next_tools=(
                    "verify_alignment",
                    "track_requirements",
                    "inspect_relation",
                    "run_verified_compute",
                    "blocked",
                ),
            )

        negative_scope = None
        if decision in {"not_applicable", "conflict", "blocked_ok"}:
            negative_scope = {
                "kind": f"verifier_{decision}",
                "target_kind": target_kind,
                "target_refs": list(target_refs),
                "evidence_refs": list(evidence_refs),
                "binding_refs": list(binding_refs),
                "compute_refs": list(compute_refs),
            }
        return state.add_evidence(
            tool_name="verify_alignment",
            ok=True,
            summary=f"Verifier marked {target_kind} as {decision}.",
            payload=payload,
            negative_scope=negative_scope,
            allowed_next_tools=(
                "bind",
                "track_requirements",
                "inspect_relation",
                "run_verified_compute",
                "submit_final",
                "locate_sources",
                "run_document_agent",
                "blocked",
            ),
            recommended_next_actions=tuple(recommended_actions[:8]),
        )

    def _inspect_relation(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        binding_ref = str(arguments.get("binding_ref") or "").strip()
        relation_name = str(arguments.get("relation_name") or "").strip()
        limit = max(1, min(int(arguments.get("limit") or 10), 100))
        binding = None
        if binding_ref:
            binding = state.bindings.get(binding_ref)
        elif relation_name:
            binding = next(
                (
                    item
                    for item in state.bindings.values()
                    if item.relation_name == relation_name
                ),
                None,
            )
        else:
            relation_bindings = [
                item for item in state.bindings.values() if item.relation_name
            ]
            if len(relation_bindings) == 1:
                binding = relation_bindings[0]
        if binding is not None and not binding.relation_name:
            return state.add_evidence(
                tool_name="inspect_relation",
                ok=False,
                summary=f"Binding {binding.id} is not a compute relation.",
                payload={"binding": binding.to_dict()},
                source_id=binding.source_id,
                negative_scope={"kind": "binding_not_relation", "binding_ref": binding.id},
                allowed_next_tools=("submit_final", "bind", "run_verified_compute", "blocked"),
                recommended_next_actions=(
                    _recommend(
                        "submit_final",
                        {
                            "binding_refs": [binding.id],
                            "evidence_refs": list(binding.evidence_refs),
                            "answer": {},
                        },
                        "Use direct final only if this non-relation binding fully supports the answer.",
                    ),
                ),
            )
        if binding is None:
            return state.add_evidence(
                tool_name="inspect_relation",
                ok=False,
                summary="No verified relation matched inspect_relation arguments.",
                payload={
                    "arguments": arguments,
                    "available_relations": [
                        {
                            "binding_ref": item.id,
                            "relation_name": item.relation_name,
                            "columns": list(item.allowed_columns),
                        }
                        for item in state.bindings.values()
                        if item.relation_name
                    ],
                },
                negative_scope={"kind": "unknown_relation", "arguments": arguments},
                allowed_next_tools=("bind", "inspect_source", "run_document_agent"),
            )
        try:
            frame = load_binding_frame(state, binding)
        except Exception as exc:  # noqa: BLE001 - relation inspection failure is evidence
            return state.add_evidence(
                tool_name="inspect_relation",
                ok=False,
                summary=f"Failed to inspect relation {binding.relation_name}: {type(exc).__name__}: {exc}",
                payload={
                    "binding": binding.to_dict(),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
                source_id=binding.source_id,
                negative_scope={"kind": "relation_inspection_failed", "binding_ref": binding.id},
                allowed_next_tools=("bind", "inspect_source", "blocked"),
            )
        columns = [str(column) for column in frame.columns]
        sample = _sample_dataframe(frame, limit=limit)
        return state.add_evidence(
            tool_name="inspect_relation",
            ok=True,
            summary=f"Relation {binding.relation_name} has {len(columns)} column(s) and {len(frame)} row(s).",
            payload={
                "binding_ref": binding.id,
                "relation_name": binding.relation_name,
                "binding_type": binding.binding_type,
                "source_id": binding.source_id,
                "table": binding.table,
                "columns": columns,
                "types": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
                "row_count": int(len(frame)),
                "sample": sample,
            },
            source_id=binding.source_id,
            allowed_next_tools=("run_verified_compute", "bind", "blocked"),
        )

    def _discover_join_paths(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        binding_refs = tuple(
            str(item) for item in arguments.get("binding_refs", []) if str(item).strip()
        )
        if not binding_refs:
            binding_refs = tuple(
                binding.id for binding in state.bindings.values()
                if binding.relation_name
            )
        bindings = [state.bindings.get(ref) for ref in binding_refs]
        bindings = [binding for binding in bindings if binding is not None and binding.relation_name]
        if len(bindings) < 2:
            return state.add_evidence(
                tool_name="discover_join_paths",
                ok=False,
                summary="discover_join_paths requires at least two verified relation bindings.",
                payload={
                    "binding_refs": list(binding_refs),
                    "available_relations": [
                        {
                            "binding_ref": binding.id,
                            "relation_name": binding.relation_name,
                            "columns": list(binding.allowed_columns),
                        }
                        for binding in state.bindings.values()
                        if binding.relation_name
                    ],
                },
                negative_scope={"kind": "insufficient_verified_relations_for_join_discovery"},
                allowed_next_tools=("bind", "inspect_relation", "run_verified_compute", "blocked"),
            )

        sample_limit = max(1, min(int(arguments.get("sample_limit") or 2000), 5000))
        max_pairs = max(1, min(int(arguments.get("max_pairs") or 80), 200))
        relation_payloads: list[dict[str, Any]] = []
        frames: dict[str, pd.DataFrame] = {}
        load_errors: list[dict[str, Any]] = []
        for binding in bindings:
            try:
                frame = load_binding_frame(state, binding)
            except Exception as exc:  # noqa: BLE001 - failed relation load is evidence
                load_errors.append(
                    {
                        "binding_ref": binding.id,
                        "relation_name": binding.relation_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            sample = frame.head(sample_limit).copy()
            frames[binding.id] = sample
            relation_payloads.append(
                {
                    "binding_ref": binding.id,
                    "relation_name": binding.relation_name,
                    "binding_type": binding.binding_type,
                    "row_count_sampled": int(len(sample)),
                    "row_count_total": int(len(frame)),
                    "columns": [str(column) for column in sample.columns],
                }
            )

        def values_for(frame: pd.DataFrame, column: str) -> set[str]:
            values: set[str] = set()
            if column not in frame.columns:
                return values
            for value in frame[column].dropna().head(sample_limit):
                text = str(value).strip()
                if text:
                    values.add(text.casefold())
                if len(values) >= sample_limit:
                    break
            return values

        candidates: list[dict[str, Any]] = []
        binding_items = [binding for binding in bindings if binding.id in frames]
        for left_index, left in enumerate(binding_items):
            left_frame = frames[left.id]
            left_columns = [str(column) for column in left_frame.columns][:80]
            for right in binding_items[left_index + 1 :]:
                right_frame = frames[right.id]
                right_columns = [str(column) for column in right_frame.columns][:80]
                right_norm = {_normalize(column): column for column in right_columns}
                for left_column in left_columns:
                    normalized = _normalize(left_column)
                    if normalized and normalized in right_norm:
                        right_column = right_norm[normalized]
                        overlap = values_for(left_frame, left_column) & values_for(right_frame, right_column)
                        candidates.append(
                            {
                                "left_binding_ref": left.id,
                                "left_relation_name": left.relation_name,
                                "left_column": left_column,
                                "right_binding_ref": right.id,
                                "right_relation_name": right.relation_name,
                                "right_column": right_column,
                                "match_reason": "same_normalized_column_name",
                                "overlap_count": len(overlap),
                                "overlap_values_preview": sorted(overlap)[:10],
                            }
                        )

                value_sets_left = {
                    column: values_for(left_frame, column)
                    for column in left_columns
                }
                value_sets_right = {
                    column: values_for(right_frame, column)
                    for column in right_columns
                }
                for left_column, left_values in value_sets_left.items():
                    if not left_values:
                        continue
                    for right_column, right_values in value_sets_right.items():
                        if not right_values:
                            continue
                        overlap = left_values & right_values
                        if not overlap:
                            continue
                        if _normalize(left_column) == _normalize(right_column):
                            continue
                        candidates.append(
                            {
                                "left_binding_ref": left.id,
                                "left_relation_name": left.relation_name,
                                "left_column": left_column,
                                "right_binding_ref": right.id,
                                "right_relation_name": right.relation_name,
                                "right_column": right_column,
                                "match_reason": "sample_value_overlap",
                                "overlap_count": len(overlap),
                                "left_unique_sample_values": len(left_values),
                                "right_unique_sample_values": len(right_values),
                                "overlap_values_preview": sorted(overlap)[:10],
                            }
                        )

        candidates.sort(
            key=lambda item: (
                item.get("match_reason") != "same_normalized_column_name",
                -int(item.get("overlap_count") or 0),
                str(item.get("left_relation_name") or ""),
                str(item.get("left_column") or ""),
            )
        )
        candidates = candidates[:max_pairs]
        return state.add_evidence(
            tool_name="discover_join_paths",
            ok=True,
            summary=f"Discovered {len(candidates)} generic join candidate(s) across verified relations.",
            payload={
                "binding_refs": [binding.id for binding in binding_items],
                "sample_limit": sample_limit,
                "relations": relation_payloads,
                "load_errors": load_errors,
                "join_candidates": candidates,
            },
            negative_scope=(
                {"kind": "join_path_candidates_not_found", "binding_refs": list(binding_refs)}
                if not candidates
                else None
            ),
            allowed_next_tools=("run_verified_compute", "inspect_relation", "verify_alignment", "blocked"),
            recommended_next_actions=(
                _recommend(
                    "run_verified_compute",
                    {"binding_refs": [binding.id for binding in binding_items]},
                    "Use only observed relation names and candidate join columns if they satisfy the question.",
                ),
            ) if candidates else (),
        )

    def _run_verified_compute(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        sql = str(arguments.get("sql") or "").strip()
        binding_refs = tuple(
            str(item) for item in arguments.get("binding_refs", []) if str(item).strip()
        )
        if not binding_refs:
            binding_refs = _binding_refs_from_sql_or_args(state, sql=sql, arguments=arguments)
        if not binding_refs:
            binding_refs = tuple(state.bindings)
        if not sql:
            return state.add_evidence(
                tool_name="run_verified_compute",
                ok=False,
                summary="run_verified_compute requires SQL.",
                payload={"arguments": arguments},
                negative_scope={"kind": "invalid_tool_arguments", "tool": "run_verified_compute", "missing": "sql"},
                allowed_next_tools=("inspect_relation", "blocked"),
            )
        try:
            columns, rows, evidence_refs = run_sql_over_bindings(
                state,
                sql=sql,
                binding_refs=binding_refs,
            )
        except Exception as exc:  # noqa: BLE001 - compute failures are observations
            error_type = type(exc).__name__
            compute_result = state.add_compute_result(
                sql=sql,
                columns=(),
                rows=(),
                binding_refs=binding_refs,
                evidence_refs=(),
                ok=False,
                error=f"{error_type}: {exc}",
            )
            return state.add_evidence(
                tool_name="run_verified_compute",
                ok=False,
                summary=f"Verified compute failed: {compute_result.error}",
                payload={
                    "compute_ref": compute_result.id,
                    "sql_error": {
                        "type": error_type,
                        "message": str(exc),
                        "sql": sql,
                        "available_relations": [
                            {
                                "binding_ref": item.id,
                            "relation_name": item.relation_name,
                            "columns": list(item.allowed_columns),
                        }
                        for item in state.bindings.values()
                        if item.relation_name
                    ],
                    },
                    "sql": sql,
                },
                negative_scope={
                    "kind": "sql_error",
                    "sql": sql,
                    "error_type": error_type,
                    "binding_refs": list(binding_refs),
                },
                allowed_next_tools=("inspect_relation", "run_verified_compute", "blocked"),
                recommended_next_actions=(
                    _recommend(
                        "inspect_relation",
                        {"binding_ref": binding_refs[0]} if binding_refs else {},
                        "Inspect verified relation columns/types before retrying SQL.",
                    ),
                ),
            )
        compute_result = state.add_compute_result(
            sql=sql,
            columns=columns,
            rows=rows,
            binding_refs=binding_refs,
            evidence_refs=evidence_refs,
        )
        recommended_actions: list[dict[str, Any]] = []
        if len(binding_refs) >= 2 and re.search(r"\bunion(?:\s+all)?\b", sql, re.IGNORECASE):
            recommended_actions.append(
                _recommend(
                    "run_verified_compute",
                    {"binding_refs": list(binding_refs)},
                    (
                        "This multi-relation SQL uses UNION, which stacks rows and keeps column names by position. "
                        "If the question asks for different fields side by side, retry with a join/alignment on observed shared keys "
                        "so each requested field remains a distinct output column."
                    ),
                )
            )
        recommended_actions.append(
            _recommend(
                "verify_alignment",
                {
                    "decision": "candidate_answer",
                    "target_kind": "compute_result",
                    "compute_refs": [compute_result.id],
                    "evidence_refs": list(evidence_refs),
                    "binding_refs": list(binding_refs),
                    "alignment": "<explain why this exact compute result satisfies the question and knowledge>",
                },
                "Classify this compute result before submit_final; final submission also requires explicit answer.columns.",
            )
        )
        return state.add_evidence(
            tool_name="run_verified_compute",
            ok=True,
            summary=f"Verified compute produced {len(rows)} row(s) and {len(columns)} column(s).",
            payload={"compute_ref": compute_result.id, "columns": columns, "rows": rows[:50], "sql": sql},
            allowed_next_tools=("verify_alignment", "submit_final", "run_verified_compute", "inspect_relation"),
            recommended_next_actions=tuple(recommended_actions),
        )

    def _submit_final(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        compute_ref = str(arguments.get("compute_ref") or "").strip()
        if compute_ref:
            compute_result = state.compute_results.get(compute_ref)
            if compute_result is None or not compute_result.ok:
                return state.add_evidence(
                    tool_name="submit_final",
                    ok=False,
                    summary="submit_final requires an existing successful compute_ref.",
                    payload={"compute_ref": compute_ref},
                    negative_scope={"kind": "invalid_final_compute_ref", "compute_ref": compute_ref},
                    allowed_next_tools=("run_verified_compute", "inspect_relation", "blocked"),
                )
            if not compute_result.rows:
                return state.add_evidence(
                    tool_name="submit_final",
                    ok=False,
                    summary="submit_final cannot materialize an empty compute result.",
                    payload={"compute_ref": compute_ref, "columns": list(compute_result.columns)},
                    negative_scope={"kind": "empty_final_compute_ref", "compute_ref": compute_ref},
                    allowed_next_tools=("run_verified_compute", "inspect_relation", "blocked"),
                )
            if not _compute_has_candidate_answer_verification(state, compute_ref):
                return state.add_evidence(
                    tool_name="submit_final",
                    ok=False,
                    summary="Compute-backed submit_final requires a candidate_answer verify_alignment decision for this compute_ref.",
                    payload={
                        "compute_ref": compute_ref,
                        "columns": list(compute_result.columns),
                    },
                    negative_scope={
                        "kind": "unverified_compute_final",
                        "compute_ref": compute_ref,
                    },
                    allowed_next_tools=("verify_alignment", "run_verified_compute", "inspect_relation", "blocked"),
                    recommended_next_actions=(
                        _recommend(
                            "verify_alignment",
                            {
                                "decision": "candidate_answer",
                                "target_kind": "compute_result",
                                "compute_refs": [compute_ref],
                                "evidence_refs": list(compute_result.evidence_refs),
                                "binding_refs": list(compute_result.binding_refs),
                                "alignment": "<explain why this exact compute result satisfies the question and knowledge>",
                            },
                            "Classify this compute result before final submission.",
                        ),
                    ),
                )
            answer_payload = arguments.get("answer")
            normalized_answer = _project_compute_answer(
                answer_payload,
                compute_columns=compute_result.columns,
                compute_rows=compute_result.rows,
            )
            if normalized_answer is None and (
                isinstance(answer_payload, list)
                or (
                    isinstance(answer_payload, dict)
                    and "columns" in answer_payload
                    and "rows" in answer_payload
                )
            ):
                normalized_answer = _answer_table_from_payload(answer_payload)
            if normalized_answer is not None:
                columns, rows = normalized_answer
                supported, unsupported_values = _projection_values_supported(
                    rows,
                    compute_result.rows,
                )
                if not supported:
                    return state.add_evidence(
                        tool_name="submit_final",
                        ok=False,
                        summary="Compute-backed final answer contains values not present in the cited compute result.",
                        payload={
                            "compute_ref": compute_ref,
                            "unsupported_values": unsupported_values,
                            "answer": arguments.get("answer"),
                        },
                        negative_scope={
                            "kind": "unsupported_compute_projection_values",
                            "compute_ref": compute_ref,
                            "unsupported_values": unsupported_values,
                        },
                        allowed_next_tools=("run_verified_compute", "submit_final", "inspect_relation", "blocked"),
                    )
                state.final_answer = {
                    "columns": columns,
                    "rows": rows,
                    "compute_ref": compute_ref,
                    "binding_refs": list(compute_result.binding_refs),
                    "evidence_refs": list(compute_result.evidence_refs),
                    "alignment": str(arguments.get("alignment") or ""),
                }
                return state.add_evidence(
                    tool_name="submit_final",
                    ok=True,
                    summary=f"Final answer materialized as a compute-backed projection from {compute_ref}.",
                    payload=state.final_answer,
                )
            return state.add_evidence(
                tool_name="submit_final",
                ok=False,
                summary="Compute-backed submit_final requires an explicit answer.columns projection.",
                payload={
                    "compute_ref": compute_ref,
                    "available_columns": list(compute_result.columns),
                    "answer": answer_payload,
                },
                negative_scope={
                    "kind": "missing_final_projection",
                    "compute_ref": compute_ref,
                },
                allowed_next_tools=("submit_final", "run_verified_compute", "inspect_relation", "blocked"),
                recommended_next_actions=(
                    _recommend(
                        "submit_final",
                        {"compute_ref": compute_ref, "answer": {"columns": list(compute_result.columns)}},
                        "Choose the final output columns explicitly from the compute result.",
                    ),
                ),
            )

        normalized = _answer_table_from_payload(arguments.get("answer"))
        binding_refs = [
            str(item) for item in arguments.get("binding_refs", []) if str(item).strip()
        ]
        evidence_refs = [
            str(item) for item in arguments.get("evidence_refs", []) if str(item).strip()
        ]
        if normalized is None or not binding_refs or not evidence_refs:
            return state.add_evidence(
                tool_name="submit_final",
                ok=False,
                summary="Direct submit_final requires answer, binding_refs, and evidence_refs.",
                payload={
                    "arguments": arguments,
                    "required": ["answer", "binding_refs", "evidence_refs"],
                },
                negative_scope={"kind": "invalid_direct_final_arguments"},
                allowed_next_tools=("bind", "run_verified_compute", "blocked"),
            )
        unknown_bindings = [ref for ref in binding_refs if ref not in state.bindings]
        unknown_evidence = [ref for ref in evidence_refs if ref not in state.evidence]
        failed_evidence = [
            ref for ref in evidence_refs if ref in state.evidence and not state.evidence[ref].ok
        ]
        if unknown_bindings or unknown_evidence or failed_evidence:
            return state.add_evidence(
                tool_name="submit_final",
                ok=False,
                summary="Direct submit_final references unknown or failed lineage.",
                payload={
                    "unknown_bindings": unknown_bindings,
                    "unknown_evidence": unknown_evidence,
                    "failed_evidence": failed_evidence,
                },
                negative_scope={"kind": "invalid_direct_final_lineage"},
                allowed_next_tools=("bind", "run_document_agent", "inspect_relation", "blocked"),
            )
        columns, rows = normalized
        state.final_answer = {
            "columns": columns,
            "rows": rows,
            "binding_refs": binding_refs,
            "evidence_refs": evidence_refs,
            "alignment": str(arguments.get("alignment") or ""),
        }
        return state.add_evidence(
            tool_name="submit_final",
            ok=True,
            summary="Final answer materialized from direct verified evidence.",
            payload=state.final_answer,
        )

    def _inspect_video(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        source = _source_from_args(state, arguments)
        if source is None:
            return state.add_evidence(
                tool_name="inspect_video",
                ok=False,
                summary="No observed video source matched arguments.",
                payload={"arguments": arguments},
                negative_scope={"kind": "unknown_source", "tool": "inspect_video", "arguments": arguments},
                allowed_next_tools=("list_inventory", "locate_sources"),
            )
        return state.add_evidence(
            tool_name="inspect_video",
            ok=False,
            summary="Video source observed, but the v1 video adapter is unsupported.",
            payload={
                "source_id": source.id,
                "path": source.virtual_path,
                "data_form": source.data_form,
                "unsupported": True,
            },
            source_id=source.id,
            data_form=source.data_form,
            negative_scope=_source_negative_scope(kind="video_unsupported_v1", source=source),
            allowed_next_tools=("locate_sources", "blocked"),
        )

    def _extract_video_observations(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        source = _source_from_args(state, arguments)
        return state.add_evidence(
            tool_name="extract_video_observations",
            ok=False,
            summary="Video extraction is an interface placeholder and unsupported in v1.",
            payload={
                "source_id": source.id if source else None,
                "unsupported": True,
                "needs_video_adapter": True,
            },
            source_id=source.id if source else None,
            data_form=source.data_form if source else "video",
            negative_scope={
                "kind": "video_extraction_unsupported_v1",
                "source_id": source.id if source else None,
            },
            allowed_next_tools=("locate_sources", "blocked"),
        )

