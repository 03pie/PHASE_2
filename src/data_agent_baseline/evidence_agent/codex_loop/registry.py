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
    AnswerContract,
    Evidence,
    LoopState,
    ModelAction,
    SemanticSelection,
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
    row_indices: tuple[int, ...] | None = None,
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
    selected_rows = (
        tuple(compute_rows[index] for index in row_indices)
        if row_indices is not None
        else compute_rows
    )
    return requested, [[row[index] for index in indexes] for row in selected_rows]


def _row_indices_from_answer(answer: Any, *, row_count: int) -> tuple[tuple[int, ...] | None, list[str]]:
    if not isinstance(answer, dict) or "row_indices" not in answer:
        return None, []
    raw = answer.get("row_indices")
    if not isinstance(raw, list):
        return None, ["row_indices_not_list"]
    if not raw:
        return None, ["row_indices_empty"]
    indexes: list[int] = []
    errors: list[str] = []
    for position, item in enumerate(raw):
        try:
            index = int(item)
        except (TypeError, ValueError):
            errors.append(f"row_index_not_integer:{position}")
            continue
        if index < 0 or index >= row_count:
            errors.append(f"row_index_out_of_range:{index}")
            continue
        indexes.append(index)
    if errors:
        return None, errors
    return tuple(indexes), []


def _record_columns(records: Any) -> list[str]:
    if not isinstance(records, list):
        return []
    columns: list[str] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        for column in record:
            text = str(column).strip()
            if not text or text == "provenance" or text in seen:
                continue
            seen.add(text)
            columns.append(text)
    return columns


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


def _helper_fields_from_arguments(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        return {}
    helper_fields: dict[str, tuple[str, ...]] = {}
    for role, raw_fields in value.items():
        role_text = str(role).strip()
        fields = _string_list(raw_fields)
        if not role_text or not fields:
            continue
        helper_fields[role_text] = fields
    return helper_fields


def _field_roles_from_arguments(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    roles: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").strip()
        role = str(item.get("role") or "").strip()
        if not field or not role:
            continue
        roles.append(
            {
                **dict(item),
                "field": field,
                "role": role,
                "semantic_card_ids": list(_string_list(item.get("semantic_card_ids"))),
                "semantic_field": str(item.get("semantic_field") or "").strip(),
                "reason": str(item.get("reason") or "").strip(),
            }
        )
    return tuple(roles)


def _contract_document_fields(contract: AnswerContract) -> tuple[str, ...]:
    fields: list[str] = []
    fields.extend(contract.final_outputs or contract.requested_outputs)
    for values in contract.helper_fields.values():
        fields.extend(values)
    if isinstance(contract.document_policy, dict):
        fields.extend(_string_list(contract.document_policy.get("required_fields")))
    for item in contract.field_roles:
        fields.extend(_string_list(item.get("semantic_field")))
        fields.extend(_string_list(item.get("field")))
    return tuple(dict.fromkeys(field for field in fields if str(field).strip()))


def _compute_error_relation_samples(state: LoopState, binding_refs: tuple[str, ...], *, limit: int = 8) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for ref in binding_refs:
        binding = state.bindings.get(ref)
        if binding is None:
            continue
        metadata = binding.metadata if isinstance(binding.metadata, dict) else {}
        records = metadata.get("records") if isinstance(metadata.get("records"), list) else []
        samples.append(
            {
                "binding_ref": ref,
                "relation_name": binding.relation_name,
                "allowed_columns": list(binding.allowed_columns),
                "records": records[:limit],
            }
        )
    return samples


def _source_ids_from_refs(state: LoopState, refs: tuple[str, ...]) -> tuple[str, ...]:
    source_ids: list[str] = []
    for ref in refs:
        source_id = ref if ref in state.sources else state.source_by_path.get(ref)
        if source_id and source_id in state.sources:
            source_ids.append(source_id)
    return tuple(dict.fromkeys(source_ids))


def _card_field_id(card: Any) -> str | None:
    if isinstance(card, dict):
        table = str(card.get("semantic_scope") or card.get("canonical_table") or "").strip()
        field = str(card.get("semantic_slot") or card.get("canonical_field") or "").strip()
    else:
        table = str(getattr(card, "semantic_scope", getattr(card, "canonical_table", "")) or "").strip()
        field = str(getattr(card, "semantic_slot", getattr(card, "canonical_field", "")) or "").strip()
    if not table or not field:
        return None
    return f"{table}.{field}".casefold()


def _enrich_document_agent_arguments(
    state: LoopState,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(arguments)
    contract = state.answer_contract
    if contract is not None:
        contract_fields = _contract_document_fields(contract)
        if not _string_list(normalized.get("target_fields")) and contract_fields:
            normalized["target_fields"] = list(contract_fields)
        policy = dict(normalized.get("coverage_policy") or {})
        contract_policy = dict(contract.document_policy or {})
        if "required_fields" in contract_policy and not isinstance(contract_policy.get("required_fields"), list):
            contract_policy.pop("required_fields", None)
        if contract_policy:
            policy = {**contract_policy, **policy}
        if contract.null_policy == "preserve":
            policy.setdefault("include_missing_records", True)
            missing_policy = dict(policy.get("missing_value_policy") or {})
            missing_policy.setdefault("include_missing_records", True)
            policy["missing_value_policy"] = missing_policy
        if policy.get("include_missing_records"):
            policy["required_fields"] = list(_string_list(contract_policy.get("required_fields")))
        if policy:
            normalized["coverage_policy"] = policy
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
    selected_card_ids = (
        set(state.semantic_selection.card_ids)
        if state.semantic_selection is not None
        else set()
    )
    if source_ids:
        for mapping in state.source_mappings:
            if selected_card_ids and mapping.card_id not in selected_card_ids:
                continue
            if not selected_card_ids and not raw_cards:
                continue
            if mapping.source_id not in source_ids or mapping.status != "unverified_document_candidate":
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
            str(card_by_id[card_id].semantic_slot)
            for card_id in document_card_ids
            if card_id in card_by_id and card_by_id[card_id].semantic_slot
        }
        target_fields = [
            field
            for field in raw_target_fields
            if field.casefold() in {item.casefold() for item in allowed_field_names}
        ]
        requested_card_field_names = {
            str(
                card.get("semantic_slot", card.get("canonical_field", ""))
                if isinstance(card, dict)
                else getattr(card, "semantic_slot", getattr(card, "canonical_field", ""))
            ).strip()
            for card in enriched_cards
            if _card_field_id(card) in document_field_ids
        }
        requested_card_field_names = {field for field in requested_card_field_names if field}
        if not target_fields:
            target_fields = sorted(requested_card_field_names or allowed_field_names)
        normalized["target_fields"] = target_fields

        target_field_names = {field.casefold() for field in target_fields}
        for card_id in document_card_ids:
            card = card_by_id.get(card_id)
            semantic_slot = str(getattr(card, "semantic_slot", getattr(card, "canonical_field", "")) or "")
            if not card or semantic_slot.casefold() not in target_field_names:
                continue
            if card.id in seen_card_ids:
                continue
            seen_card_ids.add(card.id)
            enriched_cards.append(card)

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
        semantic_field_names = {
            str(
                card.get("semantic_slot", card.get("canonical_field", ""))
                if isinstance(card, dict)
                else getattr(card, "semantic_slot", getattr(card, "canonical_field", ""))
            ).strip()
            for card in enriched_cards
        }
        semantic_field_names = {field for field in semantic_field_names if field}
        raw_target_fields = _string_list(arguments.get("target_fields"))
        if semantic_field_names and raw_target_fields:
            semantic_norms = {field.casefold() for field in semantic_field_names}
            raw_norms = {field.casefold() for field in raw_target_fields}
            if not (semantic_norms & raw_norms):
                normalized["target_fields"] = sorted(semantic_field_names)
        normalized["semantic_cards"] = [
            card.to_dict() if hasattr(card, "to_dict") else dict(card)
            for card in enriched_cards
        ]
    return normalized


def _source_evidence_refs(state: LoopState, source_ids: tuple[str, ...]) -> tuple[str, ...]:
    refs: list[str] = []
    for evidence in state.evidence.values():
        if evidence.source_id in source_ids and evidence.ok and evidence.tool_name not in {
            "declare_answer_contract",
            "select_semantic_cards",
            "bind",
            "submit_final",
        }:
            refs.append(evidence.id)
    return tuple(dict.fromkeys(refs))


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
            "declare_answer_contract": self._declare_answer_contract,
            "select_semantic_cards": self._select_semantic_cards,
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
                "inspect_relation", "Inspect verified relation schema/sample before compute or SQL changes."
            ),
            "discover_join_paths": ToolSpec(
                "discover_join_paths", "Inspect verified relations for generic join candidates."
            ),
            "declare_answer_contract": ToolSpec(
                "declare_answer_contract", "Record the LLM-declared semantic question contract."
            ),
            "select_semantic_cards": ToolSpec(
                "select_semantic_cards", "Record the LLM-selected semantic knowledge cards."
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
            allowed_next_tools=("declare_answer_contract", "retrieve_knowledge", "locate_sources", "inspect_source"),
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
                    "semantic_scope": card.semantic_scope,
                    "semantic_slot": card.semantic_slot,
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
                                card.semantic_scope,
                                card.semantic_slot or "",
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
                        "Semantic cards define meaning, aliases, units, grain, formulas, and ambiguity rules. "
                        "They do not define physical source format; use locate_sources for source candidates."
                    ),
                },
                negative_scope={"kind": "missing_semantic_knowledge"} if not state.semantic_cards else None,
                allowed_next_tools=("select_semantic_cards", "retrieve_knowledge", "locate_sources", "inspect_source", "run_document_agent"),
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
        selected_card_ids = (
            set(state.semantic_selection.card_ids)
            if state.semantic_selection is not None
            else set()
        )
        seen_semantic_candidates: set[tuple[str, str, str]] = set()

        for card in state.semantic_cards:
            if card.id not in selected_card_ids:
                continue
            semantic_slot = str(card.semantic_slot or "").strip()
            if not semantic_slot:
                continue
            semantic_slot_id = f"{card.semantic_scope}.{semantic_slot}".casefold()
            mappings = [mapping for mapping in state.source_mappings if mapping.card_id == card.id]
            if not mappings:
                continue
            mapping_payloads: list[dict[str, Any]] = []
            for mapping in mappings:
                preferred = mapping.status in {"unverified_structured_candidate", "unverified_document_candidate"}
                mapping_payload = {
                    **mapping.to_dict(),
                    "semantic_slot_id": semantic_slot_id,
                    "binding_priority": "grounding_candidate" if preferred else "lexical_candidate",
                    "recommended_tool": (
                        "run_document_agent"
                        if mapping.status == "unverified_document_candidate"
                        else "inspect_source"
                        if mapping.status == "unverified_structured_candidate"
                        else "retrieve_knowledge"
                    ),
                    "usage_note": (
                        "Grounding candidate for this semantic field; inspect/sample or extract before binding."
                        if preferred
                        else "Lexical candidate only; bind it as this canonical field only after the LLM declares semantic/grain alignment from observed evidence."
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
                    table=mapping.physical_table,
                    field=mapping.physical_field,
                ).to_dict()
                candidate.update(
                    {
                        "semantic_card_id": card.id,
                        "semantic_slot_id": semantic_slot_id,
                        "mapping_status": mapping.status,
                        "binding_priority": "grounding_candidate" if preferred else "lexical_candidate",
                        "recommended_tool": mapping_payload["recommended_tool"],
                        "usage_note": mapping_payload["usage_note"],
                    }
                )
                candidates.append(candidate)
            semantic_plan.append(
                {
                    "card_id": card.id,
                    "semantic_slot_id": semantic_slot_id,
                    "semantic_scope": card.semantic_scope,
                    "semantic_slot": semantic_slot,
                    "unit": card.unit,
                    "record_grain": card.record_grain,
                    "grounding_candidates": mapping_payloads,
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
                f"Located {len(candidates)} candidate(s). Semantic grounding candidates are listed "
                "before lexical inventory candidates; candidates are not bindings."
            ),
            payload={
                "query": query,
                "tokens": tokens,
                "semantic_source_plan": semantic_plan[:24],
                "candidates": candidates[:80],
                "usage_note": (
                    "Use semantic_source_plan as grounding candidates for knowledge-defined fields. "
                    "Before bind/compute, inspect/sample structured candidates or run DocumentAgent for "
                    "document candidates. Lexical candidates can be bound only after observed evidence "
                    "and explicit LLM-declared semantic mappings."
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
        coverage = payload.get("coverage_summary") if isinstance(payload.get("coverage_summary"), dict) else {}
        partial_coverage = (
            int(coverage.get("processed_slice_count") or 0)
            < int(coverage.get("total_slice_count") or 0)
            if coverage.get("total_slice_count") is not None
            else False
        )
        remaining_risks = payload.get("remaining_risks") if isinstance(payload.get("remaining_risks"), list) else []
        unresolved_doc_risks = {
            "partial_document_coverage",
            "unresolved_candidate_document_slices",
            "ambiguous_document_slices",
            "document_scan_validation_failed",
        }
        partial_coverage = partial_coverage or any(str(risk) in unresolved_doc_risks for risk in remaining_risks)
        payload["partial_coverage"] = partial_coverage
        payload["doc_task"] = task.to_dict()
        payload["record_fields"] = list(_record_columns(records))
        payload["provenance"] = [
            record.get("provenance")
            for record in records
            if isinstance(record, dict) and isinstance(record.get("provenance"), dict)
        ]
        payload["processed_slice_count"] = coverage.get("processed_slice_count")
        payload["total_slice_count"] = coverage.get("total_slice_count")
        payload["unresolved_slices"] = payload.get("uncertain_slices") or []
        payload["coverage_notes"] = {
            "remaining_risks": payload.get("remaining_risks") or [],
            "validation_warnings": payload.get("validation_warnings") or [],
        }
        evidence_ref = f"ev_{state._evidence_seq + 1:04d}"
        uncertain_slices = payload.get("uncertain_slices") if isinstance(payload.get("uncertain_slices"), list) else []
        summary = (
            f"DocumentAgent returned {len(records)} validated record(s) "
            f"from {len(source_refs)} document source(s)"
            f" with {len(uncertain_slices)} uncertain slice(s)."
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
            allowed_next_tools=("bind", "run_document_agent", "locate_sources", "blocked"),
            recommended_next_actions=(
                {
                    "tool_name": "bind",
                    "arguments": {
                        "binding_type": "document_record_set",
                        "evidence_refs": [evidence_ref],
                        "allowed_columns": _record_columns(records),
                        "alignment": "Bind validated DocumentAgent records if they satisfy the target semantic fields.",
                    },
                    "reason": "Bind the document record set only when records and coverage are sufficient.",
                },
            )
            if records
            else (),
        )

    def _declare_answer_contract(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        intent_summary = str(arguments.get("intent_summary") or "").strip()
        answer_grain = str(arguments.get("answer_grain") or "").strip()
        final_outputs = tuple(
            str(item).strip()
            for item in (
                _string_list(arguments.get("final_outputs"))
                or _string_list(arguments.get("requested_outputs"))
            )
            if str(item).strip()
        )
        requested_outputs = final_outputs
        raw_constraints = arguments.get("constraints")
        constraints = tuple(
            dict(item)
            for item in raw_constraints
            if isinstance(item, dict)
        ) if isinstance(raw_constraints, list) else ()
        operations = dict(arguments.get("operations")) if isinstance(arguments.get("operations"), dict) else {}
        helper_fields = _helper_fields_from_arguments(arguments.get("helper_fields"))
        field_roles = _field_roles_from_arguments(arguments.get("field_roles"))
        row_shape = str(arguments.get("row_shape") or operations.get("row_shape") or "preserve_rows").strip()
        null_policy = str(arguments.get("null_policy") or "preserve").strip()
        unresolved_terms = _string_list(arguments.get("unresolved_terms"))
        valid_row_shapes = {"preserve_rows", "single_row", "top_n", "aggregate"}
        valid_null_policies = {
            "preserve",
            "filter_when_metric_requires",
            "filter_when_question_requests_non_empty",
        }
        invalid: list[str] = []
        if not intent_summary:
            invalid.append("intent_summary")
        if not answer_grain:
            invalid.append("answer_grain")
        if not final_outputs:
            invalid.append("final_outputs")
        if raw_constraints is not None and not isinstance(raw_constraints, list):
            invalid.append("constraints")
        if "operations" in arguments and not isinstance(arguments.get("operations"), dict):
            invalid.append("operations")
        if row_shape not in valid_row_shapes:
            invalid.append(f"row_shape:{row_shape}")
        if null_policy not in valid_null_policies:
            invalid.append(f"null_policy:{null_policy}")
        document_policy = (
            dict(arguments.get("document_policy"))
            if isinstance(arguments.get("document_policy"), dict)
            else {}
        )
        if "required_fields" in document_policy and not isinstance(document_policy.get("required_fields"), list):
            document_policy.pop("required_fields", None)
        if invalid:
            return state.add_evidence(
                tool_name="declare_answer_contract",
                ok=False,
                summary="declare_answer_contract rejected invalid or incomplete contract fields.",
                payload={"arguments": arguments, "invalid_fields": invalid},
                negative_scope={
                    "kind": "invalid_answer_contract",
                    "invalid_fields": invalid,
                },
                allowed_next_tools=("declare_answer_contract", "retrieve_knowledge", "locate_sources"),
            )
        raw_limit = arguments.get("n", arguments.get("row_limit"))
        if raw_limit in {None, ""}:
            raw_limit = operations.get("top_n")
        row_limit: int | None = None
        if raw_limit not in {None, ""}:
            try:
                row_limit = int(raw_limit)
            except (TypeError, ValueError):
                row_limit = None
            if row_limit is not None and row_limit < 1:
                row_limit = None
        contract = AnswerContract(
            intent_summary=intent_summary,
            answer_grain=answer_grain,
            final_outputs=final_outputs,
            requested_outputs=requested_outputs,
            constraints=constraints,
            operations=operations,
            helper_fields=helper_fields,
            field_roles=field_roles,
            row_shape=row_shape,
            row_limit=row_limit,
            null_policy=null_policy,
            transform_intent=str(arguments.get("transform_intent") or "").strip(),
            document_policy=document_policy,
            unresolved_terms=unresolved_terms,
            notes=str(arguments.get("notes") or "").strip(),
        )
        state.answer_contract = contract
        return state.add_evidence(
            tool_name="declare_answer_contract",
            ok=True,
            summary="Recorded LLM-declared semantic question contract.",
            payload={"answer_contract": contract.to_dict()},
            allowed_next_tools=(
                "retrieve_knowledge",
                "select_semantic_cards",
                "locate_sources",
                "inspect_source",
                "sample_records",
                "run_document_agent",
                "bind",
                "run_verified_compute",
                "submit_final",
                "blocked",
            ),
        )

    def _select_semantic_cards(self, state: LoopState, arguments: dict[str, Any]) -> Evidence:
        raw_card_ids = _string_list(arguments.get("card_ids"))
        rationale = str(arguments.get("rationale") or "").strip()
        unmapped_intents = tuple(
            str(item).strip()
            for item in _string_list(arguments.get("unmapped_intents"))
            if str(item).strip()
        )
        card_by_id = {card.id: card for card in state.semantic_cards}
        selected_cards = [card_by_id[card_id] for card_id in raw_card_ids if card_id in card_by_id]
        missing_card_ids = [card_id for card_id in raw_card_ids if card_id not in card_by_id]
        if not selected_cards and not unmapped_intents:
            state.semantic_selection_errors.append("select_semantic_cards requires card_ids or unmapped_intents")
            return state.add_evidence(
                tool_name="select_semantic_cards",
                ok=False,
                summary="select_semantic_cards requires at least one known card_id or unmapped_intents.",
                payload={
                    "arguments": arguments,
                    "missing_card_ids": missing_card_ids,
                },
                negative_scope={
                    "kind": "invalid_semantic_selection",
                    "missing_card_ids": missing_card_ids,
                },
                allowed_next_tools=("retrieve_knowledge", "select_semantic_cards", "locate_sources"),
            )
        selected_ids = tuple(card.id for card in selected_cards)
        selected_mappings = [
            mapping for mapping in state.source_mappings if mapping.card_id in selected_ids
        ]
        state.semantic_selection = SemanticSelection(
            card_ids=selected_ids,
            selected_cards=tuple(card.to_dict() for card in selected_cards),
            rationale=rationale,
            unmapped_intents=unmapped_intents,
        )
        state.selected_source_mappings = selected_mappings
        payload = {
            "semantic_selection": state.semantic_selection.to_dict(),
            "missing_card_ids": missing_card_ids,
            "selected_source_mappings": [mapping.to_dict() for mapping in selected_mappings],
            "usage_note": (
                "Only these selected cards are eligible for automatic source candidate expansion. "
                "For unmapped intents, use locate_sources/search_values over observed inventory."
            ),
        }
        return state.add_evidence(
            tool_name="select_semantic_cards",
            ok=True,
            summary=(
                f"Selected {len(selected_cards)} semantic card(s) and "
                f"{len(selected_mappings)} source candidate mapping(s)."
            ),
            payload=payload,
            allowed_next_tools=(
                "locate_sources",
                "inspect_source",
                "sample_records",
                "search_values",
                "run_document_agent",
                "bind",
                "run_verified_compute",
                "submit_final",
                "blocked",
            ),
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
                "compute_helpers": {
                    "parse_date_key(value)": (
                        "Returns a YYYYMMDD integer or NULL for common date strings, "
                        "including Chinese numeral dates."
                    ),
                },
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
            allowed_next_tools=("run_verified_compute", "inspect_relation", "blocked"),
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
                        "relation_samples": _compute_error_relation_samples(state, binding_refs),
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
                        "run_verified_compute",
                        {
                            "binding_refs": list(binding_refs),
                            "sql": "<rewrite SQL using explicit casts/string operations chosen from observed relation_samples>",
                        },
                        "Retry SQL using only observed relation columns and explicit transformations chosen by the LLM from the sample values.",
                    ),
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
        return state.add_evidence(
            tool_name="run_verified_compute",
            ok=True,
            summary=f"Verified compute produced {len(rows)} row(s) and {len(columns)} column(s).",
            payload={"compute_ref": compute_result.id, "columns": columns, "rows": rows[:50], "sql": sql},
            allowed_next_tools=("submit_final", "run_verified_compute", "inspect_relation"),
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
            answer_payload = arguments.get("answer")
            row_indices, row_index_errors = _row_indices_from_answer(
                answer_payload,
                row_count=len(compute_result.rows),
            )
            if row_index_errors:
                return state.add_evidence(
                    tool_name="submit_final",
                    ok=False,
                    summary="Compute-backed final answer has invalid row_indices.",
                    payload={
                        "compute_ref": compute_ref,
                        "row_count": len(compute_result.rows),
                        "row_index_errors": row_index_errors,
                        "answer": answer_payload,
                    },
                    negative_scope={
                        "kind": "invalid_final_row_indices",
                        "compute_ref": compute_ref,
                        "row_index_errors": row_index_errors,
                    },
                    allowed_next_tools=("submit_final", "run_verified_compute", "inspect_relation", "blocked"),
                    recommended_next_actions=(
                        _recommend(
                            "submit_final",
                            {
                                "compute_ref": compute_ref,
                                "answer": {
                                    "columns": [],
                                    "row_indices": [],
                                },
                                "available_columns": list(compute_result.columns),
                                "row_count": len(compute_result.rows),
                                "instruction": "Fill columns and zero-based row_indices from this compute result only.",
                            },
                            "Retry with row_indices that exist in the cited compute result, or recompute the desired row.",
                        ),
                    ),
                )
            normalized_answer = _project_compute_answer(
                answer_payload,
                compute_columns=compute_result.columns,
                compute_rows=compute_result.rows,
                row_indices=row_indices,
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
                if not rows:
                    return state.add_evidence(
                        tool_name="submit_final",
                        ok=False,
                        summary="Compute-backed final answer projection produced no rows.",
                        payload={
                            "compute_ref": compute_ref,
                            "answer": arguments.get("answer"),
                            "available_columns": list(compute_result.columns),
                            "row_count": len(compute_result.rows),
                        },
                        negative_scope={
                            "kind": "empty_final_projection",
                            "compute_ref": compute_ref,
                        },
                        allowed_next_tools=("submit_final", "run_verified_compute", "inspect_relation", "blocked"),
                    )
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
                    "row_indices": list(row_indices) if row_indices is not None else None,
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
                        {
                            "compute_ref": compute_ref,
                            "answer": {"columns": []},
                            "available_columns": list(compute_result.columns),
                            "instruction": "Fill answer.columns with requested output columns only.",
                        },
                        "Choose the final output columns explicitly from the compute result; do not submit all columns by default.",
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

