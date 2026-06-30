from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field as dataclass_field
from typing import Any

import fitz
from langchain_core.messages import HumanMessage, SystemMessage

from data_agent_baseline.evidence_agent.codex_loop.native_tools import extract_tool_calls
from data_agent_baseline.evidence_agent.codex_loop.protocol import LoopState, SourceRef
from data_agent_baseline.evidence_agent.semantic import semantic_terms

_DOCUMENT_FORMS = {"pdf_document", "markdown_document"}
_MAX_SLICE_CHARS = 3_500


@dataclass(frozen=True, slots=True)
class RecordSlice:
    slice_id: str
    source_id: str
    path: str
    data_form: str
    slice_index: int
    text: str
    page_start: int | None = None
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    block_start: int | None = None
    block_end: int | None = None
    anchor: str = ""
    previous_slice_id: str | None = None
    next_slice_id: str | None = None
    candidate_semantic_fields: tuple[str, ...] = ()

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.text) // 4)

    @property
    def preview(self) -> str:
        compact = " ".join(self.text.split())
        if len(compact) <= 220:
            return compact
        return compact[:217].rstrip() + "..."

    def public_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_estimate"] = self.token_estimate
        payload["preview"] = self.preview
        if not include_text:
            payload.pop("text", None)
        return payload


@dataclass(frozen=True, slots=True)
class DocumentRecordIndex:
    source_id: str
    path: str
    data_form: str
    slice_count: int
    page_count: int | None
    slices: tuple[RecordSlice, ...] = ()

    def public_summary(self, *, limit: int = 20) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "data_form": self.data_form,
            "slice_count": self.slice_count,
            "page_count": self.page_count,
            "slices": [item.public_dict() for item in self.slices[:limit]],
            "truncated": len(self.slices) > limit,
        }


@dataclass(frozen=True, slots=True)
class DocTask:
    question: str
    target_fields: tuple[str, ...] = ()
    semantic_cards: tuple[dict[str, Any], ...] = ()
    source_candidates: tuple[str, ...] = ()
    required_record_grain: str = ""
    coverage_policy: dict[str, Any] = dataclass_field(default_factory=dict)
    records: tuple[dict[str, Any], ...] = ()
    slice_decisions: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_arguments(cls, arguments: dict[str, Any], *, fallback_question: str) -> "DocTask":
        return cls(
            question=str(arguments.get("question") or fallback_question or "").strip(),
            target_fields=tuple(str(item) for item in arguments.get("target_fields") or [] if str(item).strip()),
            semantic_cards=tuple(
                dict(item) for item in arguments.get("semantic_cards") or [] if isinstance(item, dict)
            ),
            source_candidates=tuple(
                str(item) for item in arguments.get("source_candidates") or [] if str(item).strip()
            ),
            required_record_grain=str(arguments.get("required_record_grain") or "").strip(),
            coverage_policy=(
                dict(arguments.get("coverage_policy"))
                if isinstance(arguments.get("coverage_policy"), dict)
                else {}
            ),
            records=tuple(dict(item) for item in arguments.get("records") or [] if isinstance(item, dict)),
            slice_decisions=tuple(
                dict(item) for item in arguments.get("slice_decisions") or [] if isinstance(item, dict)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "target_fields": list(self.target_fields),
            "semantic_cards": list(self.semantic_cards),
            "source_candidates": list(self.source_candidates),
            "required_record_grain": self.required_record_grain,
            "coverage_policy": self.coverage_policy,
            "records": list(self.records),
            "slice_decisions": list(self.slice_decisions),
        }


@dataclass(frozen=True, slots=True)
class DocEvidencePackage:
    records: tuple[dict[str, Any], ...]
    record_schema: dict[str, Any]
    source_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    processed_slice_ids: tuple[str, ...]
    no_relevant_slice_ids: tuple[str, ...]
    ambiguous_slice_ids: tuple[str, ...]
    coverage_summary: dict[str, Any]
    remaining_risks: tuple[str, ...]
    validation_warnings: tuple[dict[str, Any], ...] = ()
    uncertain_slices: tuple[dict[str, Any], ...] = ()
    agent_trace: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "records": list(self.records),
            "record_schema": self.record_schema,
            "source_refs": list(self.source_refs),
            "evidence_refs": list(self.evidence_refs),
            "processed_slice_ids": list(self.processed_slice_ids),
            "no_relevant_slice_ids": list(self.no_relevant_slice_ids),
            "ambiguous_slice_ids": list(self.ambiguous_slice_ids),
            "coverage_summary": self.coverage_summary,
            "remaining_risks": list(self.remaining_risks),
            "validation_warnings": list(self.validation_warnings),
            "uncertain_slices": list(self.uncertain_slices),
        }
        if self.agent_trace:
            payload["agent_trace"] = list(self.agent_trace)
        return payload


def _compact_text(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _terms_from_items(*items: Any) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict):
            iterable: Any = item.values()
        elif isinstance(item, (list, tuple, set)):
            iterable = item
        else:
            iterable = (item,)
        for value in iterable:
            for term in semantic_terms(str(value)):
                if term not in seen:
                    seen.add(term)
                    terms.append(term)
    return tuple(terms)


def _summarize_doc_tool_args(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "record_document_decisions":
        record_fields = sorted(
            {
                str(key)
                for record in arguments.get("records") or []
                if isinstance(record, dict)
                for key in record
                if key not in {"provenance", "slice_id", "slice_ids", "evidence_slice_ids"}
            }
        )
        return {
            "record_count": len(arguments.get("records") or []),
            "decision_count": len(arguments.get("slice_decisions") or []),
            "record_fields": record_fields[:20],
            "scan_cursor": arguments.get("scan_cursor"),
        }
    if name == "inspect_document_index":
        return {
            "source_ref": arguments.get("source_ref"),
            "limit": arguments.get("limit"),
        }
    if name == "search_document_records":
        return {
            "query": _compact_text(arguments.get("query"), limit=160),
            "semantic_fields": list(arguments.get("semantic_fields") or []),
            "source_ref": arguments.get("source_ref"),
            "source_candidates": list(arguments.get("source_candidates") or []),
            "limit": arguments.get("limit"),
        }
    if name == "read_record_slice":
        return {
            "slice_ids": list(arguments.get("slice_ids") or ([arguments.get("slice_id")] if arguments.get("slice_id") else [])),
        }
    if name == "extract_semantic_records":
        target_schema = arguments.get("target_schema") if isinstance(arguments.get("target_schema"), dict) else {}
        return {
            "record_count": len(arguments.get("records") or []),
            "decision_count": len(arguments.get("slice_decisions") or []),
            "target_schema": target_schema,
        }
    if name == "check_document_coverage":
        return {"processed_slice_ids": list(arguments.get("processed_slice_ids") or [])}
    return {"keys": sorted(arguments.keys())}


def _summarize_doc_tool_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    summary: dict[str, Any] = {
        "ok": bool(result.get("ok")),
        "summary": _compact_text(result.get("summary"), limit=220),
    }
    if name == "record_document_decisions":
        summary.update(
            {
                "record_count": len(payload.get("records") or []),
                "decision_count": len(payload.get("slice_decisions") or []),
                "missing_decision_count": payload.get("missing_decision_count"),
            }
        )
    elif name == "inspect_document_index":
        summary.update(
            {
                "document_count": payload.get("document_count"),
                "total_slice_count": payload.get("total_slice_count"),
            }
        )
    elif name == "search_document_records":
        summary.update(
            {
                "total_matches": payload.get("total_matches"),
                "returned_slice_ids": [
                    item.get("slice_id")
                    for item in payload.get("matches") or []
                    if isinstance(item, dict)
                ],
                "more_matches_available": payload.get("more_matches_available"),
            }
        )
    elif name == "read_record_slice":
        summary["slice_ids"] = list(payload.get("slice_ids") or [])
    elif name == "extract_semantic_records":
        summary.update(
            {
                "record_count": len(payload.get("records") or []),
                "processed_slice_ids": list(payload.get("processed_slice_ids") or []),
                "invalid_count": len(payload.get("invalid") or []),
            }
        )
    elif name == "check_document_coverage":
        summary.update(
            {
                "processed_slice_count": payload.get("processed_slice_count"),
                "unprocessed_slice_count": payload.get("unprocessed_slice_count"),
            }
        )
    if result.get("negative_scope"):
        summary["negative_scope"] = result.get("negative_scope")
    return summary


def _target_fields_from_schema(target_schema: dict[str, Any], arguments: dict[str, Any]) -> set[str]:
    fields = target_schema.get("fields")
    if not fields and isinstance(target_schema.get("properties"), dict):
        fields = list(target_schema["properties"].keys())
    if not fields:
        fields = arguments.get("target_fields") or []
    return {str(item) for item in fields if str(item).strip()}


def _normalize_field_name(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value or "").casefold())


def _ordered_unique(values: Any) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _field_aliases_from_arguments(
    target_fields: set[str],
    arguments: dict[str, Any],
) -> dict[str, set[str]]:
    aliases_by_field: dict[str, set[str]] = {field: {field} for field in target_fields}
    raw_aliases = arguments.get("field_aliases") if isinstance(arguments.get("field_aliases"), dict) else {}
    target_by_norm = {_normalize_field_name(field): field for field in target_fields}
    for raw_field, raw_values in raw_aliases.items():
        field = target_by_norm.get(_normalize_field_name(raw_field))
        if not field:
            continue
        values = raw_values if isinstance(raw_values, (list, tuple, set)) else [raw_values]
        aliases_by_field.setdefault(field, {field}).update(str(item) for item in values if str(item).strip())
    return aliases_by_field


def _canonicalize_record_field(
    field_name: str,
    target_fields: set[str],
    aliases_by_field: dict[str, set[str]],
) -> str | None:
    if not target_fields:
        return field_name
    normalized = _normalize_field_name(field_name)
    target_by_norm = {_normalize_field_name(field): field for field in target_fields}
    if normalized in target_by_norm:
        return target_by_norm[normalized]
    for field, aliases in aliases_by_field.items():
        alias_norms = {_normalize_field_name(alias) for alias in aliases}
        if normalized in alias_norms:
            return field
    wrapper_prefixes = {"record", "field", "value", "extracted", "observed", "target"}
    for field in target_fields:
        field_norm = _normalize_field_name(field)
        if not field_norm or not normalized.endswith(field_norm):
            continue
        prefix = normalized[: -len(field_norm)]
        if prefix in wrapper_prefixes:
            return field
    return None


def _compact_evidence_value(value: Any) -> str:
    return re.sub(r"\s+", "", str(value).casefold())


def _evidence_supports_value(value: Any, source_text: str) -> bool:
    compact_value = _compact_evidence_value(value)
    if not compact_value:
        return True
    compact_source = _compact_evidence_value(source_text)
    return compact_value in compact_source


def _record_slice_ids(record: dict[str, Any], provenance: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("slice_ids", "evidence_slice_ids"):
        raw = record.get(key)
        if isinstance(raw, (list, tuple, set)):
            values.extend(raw)
    if record.get("slice_id"):
        values.append(record["slice_id"])
    for key in ("slice_ids", "evidence_slice_ids"):
        raw = provenance.get(key)
        if isinstance(raw, (list, tuple, set)):
            values.extend(raw)
    if provenance.get("slice_id"):
        values.append(provenance["slice_id"])
    return _ordered_unique(values)


def _candidate_values_for_record(
    record: dict[str, Any],
    target_fields: set[str],
    aliases_by_field: dict[str, set[str]],
) -> list[str]:
    values: list[str] = []
    for key, value in record.items():
        if key in {"provenance", "slice_id"}:
            continue
        if target_fields and _canonicalize_record_field(str(key), target_fields, aliases_by_field) is None:
            continue
        if value is None or not str(value).strip():
            continue
        compact = _compact_evidence_value(value)
        if compact:
            values.append(compact)
    return values


def _infer_record_slice(
    *,
    record: dict[str, Any],
    by_id: dict[str, RecordSlice],
    candidate_slice_ids: set[str],
    target_fields: set[str],
    aliases_by_field: dict[str, set[str]],
) -> RecordSlice | None:
    values = _candidate_values_for_record(record, target_fields, aliases_by_field)
    if not values:
        return None
    candidates = [by_id[slice_id] for slice_id in candidate_slice_ids if slice_id in by_id] or list(by_id.values())
    matches: list[RecordSlice] = []
    for candidate in candidates:
        source_text = _compact_evidence_value(candidate.text)
        if all(value in source_text for value in values):
            matches.append(candidate)
            if len(matches) > 1:
                return None
    return matches[0] if matches else None


def _slice_id(source: SourceRef, index: int) -> str:
    return f"{source.id}_record_{index:04d}"


def _page_count(source: SourceRef) -> int | None:
    if source.data_form != "pdf_document":
        return None
    try:
        with fitz.open(source.path) as document:
            return len(document)
    except Exception:  # noqa: BLE001 - index summary should tolerate corrupt PDFs
        return None


def _split_long_text(text: str, *, limit: int = _MAX_SLICE_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in re.split(r"(?<=[。！？.!?])\s+", text):
        if current and current_len + len(piece) > limit:
            parts.append(" ".join(current).strip())
            current = []
            current_len = 0
        if len(piece) > limit:
            for start in range(0, len(piece), limit):
                parts.append(piece[start : start + limit].strip())
            continue
        current.append(piece)
        current_len += len(piece)
    if current:
        parts.append(" ".join(current).strip())
    return [part for part in parts if part]


def _selected_document_indexes(
    state: LoopState,
    indexes: dict[str, DocumentRecordIndex],
    task: DocTask,
) -> list[DocumentRecordIndex]:
    refs = {str(ref) for ref in task.source_candidates if str(ref).strip()}
    if not refs:
        return list(indexes.values())
    selected: list[DocumentRecordIndex] = []
    for index in indexes.values():
        source = state.sources.get(index.source_id)
        source_refs = {
            index.source_id,
            index.path,
            source.virtual_path if source is not None else "",
            source.path.as_posix() if source is not None else "",
        }
        if refs & source_refs:
            selected.append(index)
    return selected or list(indexes.values())


def _focus_slice_ids(task: DocTask) -> tuple[str, ...]:
    policy = task.coverage_policy if isinstance(task.coverage_policy, dict) else {}
    raw = policy.get("focus_slice_ids") or policy.get("slice_ids") or []
    if isinstance(raw, str):
        raw = [raw]
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _coverage_summary_for_slices(
    slices: list[RecordSlice],
    *,
    processed_slice_ids: tuple[str, ...],
) -> dict[str, Any]:
    processed = set(processed_slice_ids)
    source_ids = tuple(dict.fromkeys(item.source_id for item in slices))
    return {
        "document_count": len(source_ids),
        "total_slice_count": len(slices),
        "processed_slice_count": sum(1 for item in slices if item.slice_id in processed),
        "unprocessed_slice_count": sum(1 for item in slices if item.slice_id not in processed),
        "source_slice_counts": {
            source_id: sum(1 for item in slices if item.source_id == source_id)
            for source_id in source_ids
        },
        "coverage_scope": "focus_slices",
    }


def _next_scan_batch(
    slices: list[RecordSlice],
    cursor: int,
    *,
    max_slices: int,
    max_chars: int,
) -> tuple[list[RecordSlice], int]:
    batch: list[RecordSlice] = []
    total_chars = 0
    index = cursor
    while index < len(slices) and len(batch) < max_slices:
        item = slices[index]
        if batch and total_chars + len(item.text) > max_chars:
            break
        batch.append(item)
        total_chars += len(item.text)
        index += 1
    if not batch and cursor < len(slices):
        batch.append(slices[cursor])
        index = cursor + 1
    return batch, index


def _task_candidate_terms(task: DocTask) -> dict[str, set[str]]:
    field_terms: dict[str, set[str]] = {
        field: set(semantic_terms(field)) | {_normalize_field_name(field)}
        for field in task.target_fields
        if str(field).strip()
    }
    field_by_norm = {_normalize_field_name(field): field for field in field_terms}
    for card in task.semantic_cards:
        slot = str(card.get("semantic_slot") or card.get("canonical_field") or "").strip()
        name = str(card.get("name") or "").strip()
        candidates = [slot, name.rsplit(".", 1)[-1] if "." in name else name]
        field = next(
            (
                field_by_norm[_normalize_field_name(candidate)]
                for candidate in candidates
                if _normalize_field_name(candidate) in field_by_norm
            ),
            candidates[0] if candidates and candidates[0] else "",
        )
        if not field:
            continue
        card_text = " ".join(
            str(value or "")
            for value in (
                card.get("name"),
                card.get("semantic_scope"),
                card.get("semantic_slot"),
                card.get("definition"),
                card.get("unit"),
                card.get("record_grain"),
                " ".join(str(item) for item in card.get("aliases") or []),
            )
        )
        terms = field_terms.setdefault(field, set())
        terms.update(semantic_terms(card_text))
        compact = _normalize_field_name(field)
        if compact:
            terms.add(compact)
    return {
        field: {term for term in terms if term and len(term) > 1}
        for field, terms in field_terms.items()
    }


def _candidate_fields_for_slice(text: str, field_terms: dict[str, set[str]]) -> tuple[str, ...]:
    text_terms = semantic_terms(text)
    text_norm = _normalize_field_name(text)
    candidates: list[str] = []
    for field, terms in field_terms.items():
        compact_field = _normalize_field_name(field)
        if compact_field and compact_field in text_norm:
            candidates.append(field)
            continue
        overlap = terms & text_terms
        if len(overlap) >= 2:
            candidates.append(field)
    return tuple(candidates)


def _candidate_review_snippets(
    slices: list[RecordSlice],
    decisions: list[dict[str, Any]],
    candidate_fields_by_slice: dict[str, tuple[str, ...]],
    *,
    limit: int = 40,
) -> tuple[dict[str, Any], ...]:
    by_id = {item.slice_id: item for item in slices}
    snippets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for decision in decisions:
        if str(decision.get("status") or "") != "no_relevant_record":
            continue
        slice_id = str(decision.get("slice_id") or "").strip()
        candidates = candidate_fields_by_slice.get(slice_id, ())
        if not slice_id or not candidates or slice_id in seen or slice_id not in by_id:
            continue
        seen.add(slice_id)
        item = by_id[slice_id]
        snippets.append(
            {
                "slice_id": slice_id,
                "source_id": item.source_id,
                "path": item.path,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "line_start": item.line_start,
                "line_end": item.line_end,
                "reason": (
                    "LLM marked no_relevant_record although the slice has lexical candidate "
                    "fields from the declared document task."
                ),
                "llm_reason": str(decision.get("reason") or ""),
                "candidate_fields": list(candidates),
                "evidence_text": _compact_text(
                    decision.get("evidence_text") or item.text,
                    limit=900,
                ),
            }
        )
        if len(snippets) >= limit:
            break
    return tuple(snippets)


def _candidate_record_gap_snippets(
    slices: list[RecordSlice],
    decisions: list[dict[str, Any]],
    records: list[dict[str, Any]],
    candidate_fields_by_slice: dict[str, tuple[str, ...]],
    *,
    limit: int = 40,
) -> tuple[dict[str, Any], ...]:
    by_id = {item.slice_id: item for item in slices}
    fields_by_slice: dict[str, set[str]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
        record_fields = {
            str(field)
            for field in record
            if field not in {"provenance", "slice_id", "slice_ids", "evidence_slice_ids"}
        }
        for slice_id in _record_slice_ids(record, provenance):
            fields_by_slice.setdefault(slice_id, set()).update(record_fields)

    snippets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for decision in decisions:
        if str(decision.get("status") or "") != "record_extracted":
            continue
        slice_id = str(decision.get("slice_id") or "").strip()
        candidates = candidate_fields_by_slice.get(slice_id, ())
        if not slice_id or not candidates or slice_id in seen or slice_id not in by_id:
            continue
        present_fields = fields_by_slice.get(slice_id, set())
        missing_candidates = [field for field in candidates if field not in present_fields]
        if not missing_candidates:
            continue
        seen.add(slice_id)
        item = by_id[slice_id]
        snippets.append(
            {
                "slice_id": slice_id,
                "source_id": item.source_id,
                "path": item.path,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "line_start": item.line_start,
                "line_end": item.line_end,
                "reason": (
                    "LLM marked record_extracted for a lexical candidate slice, but validated "
                    "records did not retain the candidate field(s)."
                ),
                "candidate_fields": missing_candidates,
                "evidence_text": _compact_text(
                    decision.get("evidence_text") or item.text,
                    limit=900,
                ),
            }
        )
        if len(snippets) >= limit:
            break
    return tuple(snippets)


def _decision_slice_snippets(
    slices: list[RecordSlice],
    decisions: list[dict[str, Any]],
    *,
    limit: int = 40,
) -> tuple[dict[str, Any], ...]:
    by_id = {item.slice_id: item for item in slices}
    snippets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for decision in decisions:
        if str(decision.get("status") or "") != "ambiguous":
            continue
        slice_id = str(decision.get("slice_id") or "").strip()
        if not slice_id or slice_id in seen or slice_id not in by_id:
            continue
        seen.add(slice_id)
        item = by_id[slice_id]
        snippets.append(
            {
                "slice_id": slice_id,
                "source_id": item.source_id,
                "path": item.path,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "line_start": item.line_start,
                "line_end": item.line_end,
                "reason": str(decision.get("reason") or ""),
                "candidate_fields": list(decision.get("candidate_fields") or []),
                "evidence_text": _compact_text(
                    decision.get("evidence_text") or item.text,
                    limit=900,
                ),
            }
        )
        if len(snippets) >= limit:
            break
    return tuple(snippets)


def _record_document_decisions_result(
    arguments: dict[str, Any],
    *,
    allowed_slice_ids: set[str],
) -> dict[str, Any]:
    records = [dict(item) for item in arguments.get("records") or [] if isinstance(item, dict)]
    raw_decisions = [dict(item) for item in arguments.get("slice_decisions") or [] if isinstance(item, dict)]
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    invalid: list[dict[str, Any]] = []
    for decision in raw_decisions:
        slice_id = str(decision.get("slice_id") or "").strip()
        if slice_id not in allowed_slice_ids:
            invalid.append({"slice_id": slice_id, "error": "slice_not_in_current_batch"})
            continue
        status = str(decision.get("status") or "").strip()
        if status not in {"record_extracted", "no_relevant_record", "ambiguous"}:
            status = "ambiguous"
        seen.add(slice_id)
        decisions.append({**decision, "slice_id": slice_id, "status": status})
    for slice_id in sorted(allowed_slice_ids - seen):
        decisions.append(
            {
                "slice_id": slice_id,
                "status": "ambiguous",
                "reason": "LLM did not record a decision for this slice in the current scan batch.",
            }
        )
    return {
        "ok": True,
        "tool_name": "record_document_decisions",
        "summary": f"Recorded {len(records)} document record(s) and {len(decisions)} slice decision(s).",
        "payload": {
            "records": records,
            "slice_decisions": decisions,
            "missing_decision_count": len(allowed_slice_ids - seen),
            "invalid": invalid,
        },
    }


def _record_merge_key(record: dict[str, Any]) -> str:
    provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    for key in ("record_anchor", "merge_id", "entity_id", "anchor"):
        value = record.get(key, provenance.get(key))
        if value is not None and str(value).strip():
            return str(value).strip()
    slice_ids = _record_slice_ids(record, provenance)
    return slice_ids[0] if slice_ids else f"record:{id(record)}"


def _merge_scan_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_anchor: dict[str, dict[str, Any]] = {}
    slice_ids_by_anchor: dict[str, list[str]] = {}
    conflicts_by_anchor: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for record in records:
        anchor = _record_merge_key(record)
        if anchor not in by_anchor:
            by_anchor[anchor] = {"record_anchor": anchor}
            slice_ids_by_anchor[anchor] = []
            order.append(anchor)
        merged = by_anchor[anchor]
        provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
        for slice_id in _record_slice_ids(record, provenance):
            if slice_id not in slice_ids_by_anchor[anchor]:
                slice_ids_by_anchor[anchor].append(slice_id)
        for field, value in record.items():
            if field in {"provenance", "slice_id", "slice_ids", "evidence_slice_ids"}:
                continue
            if field in {"record_anchor", "merge_id", "entity_id", "anchor"}:
                continue
            if value is None or not str(value).strip():
                continue
            if field not in merged or merged.get(field) in {None, ""}:
                merged[field] = value
            elif str(merged[field]) != str(value):
                conflicts_by_anchor.setdefault(anchor, []).append(
                    {"field": field, "kept": merged[field], "ignored": value}
                )
    output: list[dict[str, Any]] = []
    for anchor in order:
        record = by_anchor[anchor]
        slice_ids = slice_ids_by_anchor.get(anchor) or []
        record["slice_ids"] = slice_ids
        record["provenance"] = {
            "slice_ids": slice_ids,
            "merge_id": anchor,
        }
        if conflicts_by_anchor.get(anchor):
            record["provenance"]["field_conflicts"] = conflicts_by_anchor[anchor]
        output.append(record)
    return output


class DocumentAgent:
    """Stateful document sub-loop for PDF/MD evidence packages.

    The public controller talks to this class through one coarse task call.
    Fine-grained inspect/search/read/extract/coverage tools stay inside the
    document loop so the main loop does not carry large document windows.
    """

    def __init__(self, *, model: Any | None = None, max_steps: int = 10) -> None:
        self.model = model
        self.max_steps = max_steps

    def ensure_indexes(self, state: LoopState) -> dict[str, DocumentRecordIndex]:
        missing = [
            source
            for source in state.sources.values()
            if source.data_form in _DOCUMENT_FORMS and source.id not in state.document_record_indexes
        ]
        for source in missing:
            state.document_record_indexes[source.id] = self._build_index(source)
        return {
            source_id: index
            for source_id, index in state.document_record_indexes.items()
            if isinstance(index, DocumentRecordIndex)
        }

    def run(self, state: LoopState, task: DocTask) -> DocEvidencePackage:
        self.ensure_indexes(state)
        if self.model is not None and hasattr(self.model, "bind_tools"):
            package = self._run_scan_loop(state, task)
        else:
            package = self._run_deterministic(state, task)
        state.document_agent_packages.append(package.to_dict())
        return package

    def inspect_document_index(
        self,
        state: LoopState,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        indexes = self.ensure_indexes(state)
        source_ref = str(arguments.get("source_ref") or "").strip()
        limit = max(1, min(int(arguments.get("limit") or 20), 100))
        selected = [indexes[source_ref]] if source_ref in indexes else list(indexes.values())
        return {
            "ok": True,
            "tool_name": "inspect_document_index",
            "summary": f"Indexed {len(selected)} document source(s).",
            "payload": {
                "document_count": len(selected),
                "total_slice_count": sum(index.slice_count for index in selected),
                "indexes": [index.public_summary(limit=limit) for index in selected],
            },
        }

    def search_document_records(
        self,
        state: LoopState,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        indexes = self.ensure_indexes(state)
        query = str(arguments.get("query") or "").strip()
        semantic_fields = tuple(str(item) for item in arguments.get("semantic_fields") or [] if str(item).strip())
        source_ref = str(arguments.get("source_ref") or "").strip()
        source_candidates = tuple(
            str(item) for item in arguments.get("source_candidates") or [] if str(item).strip()
        )
        limit = max(1, min(int(arguments.get("limit") or 20), 100))
        terms = _terms_from_items(query, semantic_fields)
        if not terms:
            return {
                "ok": False,
                "tool_name": "search_document_records",
                "summary": "search_document_records requires query or semantic_fields.",
                "payload": {"arguments": arguments},
                "negative_scope": {"kind": "invalid_doc_agent_search", "missing": "query_or_semantic_fields"},
            }
        allowed_sources = set(source_candidates)
        if source_ref:
            allowed_sources.add(source_ref)
        selected = [
            index
            for index in indexes.values()
            if not allowed_sources or index.source_id in allowed_sources or index.path in allowed_sources
        ]
        matches: list[dict[str, Any]] = []
        for index in selected:
            for item in index.slices:
                text = item.text.casefold()
                matched = [term for term in terms if term and term.casefold() in text]
                if not matched:
                    continue
                matches.append(
                    {
                        **item.public_dict(),
                        "score": len(matched),
                        "matched_terms": matched[:20],
                        "recommended_read": {"slice_ids": [item.slice_id]},
                    }
                )
        matches.sort(key=lambda item: (-int(item["score"]), str(item["source_id"]), int(item["slice_index"])))
        returned = matches[:limit]
        return {
            "ok": True,
            "tool_name": "search_document_records",
            "summary": f"Found {len(matches)} matching record slice(s).",
            "payload": {
                "query": query,
                "semantic_fields": list(semantic_fields),
                "total_matches": len(matches),
                "returned_matches": len(returned),
                "more_matches_available": len(matches) > len(returned),
                "matches": returned,
            },
            "negative_scope": (
                {"kind": "document_record_query_not_found", "query": query, "semantic_fields": list(semantic_fields)}
                if not matches
                else None
            ),
        }

    def read_record_slice(
        self,
        state: LoopState,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        indexes = self.ensure_indexes(state)
        slice_ids = tuple(str(item) for item in arguments.get("slice_ids") or [] if str(item).strip())
        if not slice_ids and arguments.get("slice_id"):
            slice_ids = (str(arguments["slice_id"]),)
        by_id = {item.slice_id: item for index in indexes.values() for item in index.slices}
        records = [by_id[slice_id] for slice_id in slice_ids if slice_id in by_id]
        unknown = [slice_id for slice_id in slice_ids if slice_id not in by_id]
        if unknown or not records:
            return {
                "ok": False,
                "tool_name": "read_record_slice",
                "summary": "No indexed record slice matched read_record_slice arguments.",
                "payload": {"unknown_slice_ids": unknown, "arguments": arguments},
                "negative_scope": {"kind": "unknown_record_slice", "slice_ids": list(slice_ids)},
            }
        payload_records = [item.public_dict(include_text=True) for item in records]
        return {
            "ok": True,
            "tool_name": "read_record_slice",
            "summary": f"Read {len(records)} complete record slice(s).",
            "payload": {
                "slice_ids": [item.slice_id for item in records],
                "records": payload_records,
                "text": "\n\n".join(
                    f"[slice_id={item.slice_id} source={item.source_id} page={item.page_start}-{item.page_end}]\n{item.text}"
                    for item in records
                ),
            },
        }

    def extract_semantic_records(
        self,
        state: LoopState,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        indexes = self.ensure_indexes(state)
        by_id = {item.slice_id: item for index in indexes.values() for item in index.slices}
        records = [dict(item) for item in arguments.get("records") or [] if isinstance(item, dict)]
        decisions = [dict(item) for item in arguments.get("slice_decisions") or [] if isinstance(item, dict)]
        target_schema = dict(arguments.get("target_schema") or {})
        target_fields = _target_fields_from_schema(target_schema, arguments)
        aliases_by_field = _field_aliases_from_arguments(target_fields, arguments)
        required_fields = {
            field
            for field in (
                _canonicalize_record_field(str(item), target_fields, aliases_by_field)
                for item in target_schema.get("required") or arguments.get("required_fields") or []
            )
            if field
        }
        allowed_slice_ids = {
            str(item)
            for item in arguments.get("allowed_slice_ids") or []
            if str(item).strip()
        }
        invalid: list[dict[str, Any]] = []
        validation_warnings: list[dict[str, Any]] = []
        processed: set[str] = set()
        no_relevant: set[str] = set()
        ambiguous: set[str] = set()
        decision_slice_ids: set[str] = set()
        missing_status_decisions: list[str] = []
        valid_statuses = {"record_extracted", "no_relevant_record", "ambiguous"}
        for decision in decisions:
            slice_id = str(decision.get("slice_id") or "").strip()
            status = str(decision.get("status") or "").strip()
            if slice_id not in by_id:
                invalid.append({"slice_id": slice_id, "error": "unknown_slice_id"})
                continue
            if allowed_slice_ids and slice_id not in allowed_slice_ids:
                invalid.append({"slice_id": slice_id, "error": "slice_not_read"})
                continue
            decision_slice_ids.add(slice_id)
            if status not in valid_statuses:
                if not status:
                    missing_status_decisions.append(slice_id)
                else:
                    invalid.append({"slice_id": slice_id, "error": f"invalid_status:{status}"})
                continue
            processed.add(slice_id)
            if status == "no_relevant_record":
                no_relevant.add(slice_id)
            elif status == "ambiguous":
                ambiguous.add(slice_id)
        normalized_records: list[dict[str, Any]] = []
        record_slice_ids: set[str] = set()
        for record in records:
            provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
            cited_slice_ids = _record_slice_ids(record, provenance)
            if allowed_slice_ids:
                unread = [slice_id for slice_id in cited_slice_ids if slice_id not in allowed_slice_ids]
                if unread:
                    invalid.append({"slice_ids": unread, "error": "slice_not_read", "record": record})
                    continue
            source_slices = [by_id[slice_id] for slice_id in cited_slice_ids if slice_id in by_id]
            if not source_slices:
                inferred_slice = _infer_record_slice(
                    record=record,
                    by_id=by_id,
                    candidate_slice_ids=decision_slice_ids & allowed_slice_ids if allowed_slice_ids else decision_slice_ids,
                    target_fields=target_fields,
                    aliases_by_field=aliases_by_field,
                )
                if inferred_slice is None:
                    invalid.append({"record": record, "error": "record_missing_known_slice_id"})
                    continue
                source_slices = [inferred_slice]
                cited_slice_ids = [inferred_slice.slice_id]
            unknown_slice_ids = [slice_id for slice_id in cited_slice_ids if slice_id not in by_id]
            if unknown_slice_ids:
                invalid.append({"slice_ids": unknown_slice_ids, "error": "unknown_slice_id", "record": record})
                continue
            for slice_id in cited_slice_ids:
                processed.add(slice_id)
                record_slice_ids.add(slice_id)
            source_text = "\n\n".join(item.text for item in source_slices)
            unsupported_values: list[str] = []
            normalized: dict[str, Any] = {}
            for key, value in record.items():
                if key in {"provenance", "slice_id", "slice_ids", "evidence_slice_ids"}:
                    continue
                field_name = _canonicalize_record_field(str(key), target_fields, aliases_by_field)
                if field_name is None:
                    continue
                normalized[field_name] = value
                if value is None or not str(value).strip():
                    continue
                if not _evidence_supports_value(value, source_text):
                    unsupported_values.append(str(value))
                    continue
            missing_required = [
                field
                for field in sorted(required_fields)
                if field not in normalized or normalized[field] is None or not str(normalized[field]).strip()
            ]
            if missing_required:
                invalid.append(
                    {
                        "slice_ids": cited_slice_ids,
                        "error": "missing_required_fields",
                        "fields": missing_required,
                    }
                )
                continue
            if unsupported_values:
                invalid.append(
                    {
                        "slice_ids": cited_slice_ids,
                        "error": "unsupported_extracted_values",
                        "values": unsupported_values[:20],
                    }
                )
                continue
            primary_slice = source_slices[0]
            normalized["provenance"] = {
                **provenance,
                "slice_id": cited_slice_ids[0],
                "slice_ids": cited_slice_ids,
                "source_id": primary_slice.source_id,
                "path": primary_slice.path,
                "page_start": min((item.page_start for item in source_slices if item.page_start is not None), default=None),
                "page_end": max((item.page_end for item in source_slices if item.page_end is not None), default=None),
                "evidence_text": _compact_text(
                    provenance.get("evidence_text") or " ".join(item.preview for item in source_slices),
                    limit=500,
                ),
            }
            normalized_records.append(normalized)
        for slice_id in missing_status_decisions:
            processed.add(slice_id)
            if slice_id in record_slice_ids:
                continue
            ambiguous.add(slice_id)
        coverage = self._coverage_summary(state, processed_slice_ids=tuple(sorted(processed)))
        if invalid:
            return {
                "ok": False,
                "tool_name": "extract_semantic_records",
                "summary": "Semantic document extraction failed validation.",
                "payload": {
                    "invalid": invalid,
                    "validation_warnings": validation_warnings,
                    "records": normalized_records,
                    "target_schema": target_schema,
                    "processed_slice_ids": sorted(processed),
                    "no_relevant_slice_ids": sorted(no_relevant),
                    "ambiguous_slice_ids": sorted(ambiguous),
                    "coverage_summary": coverage,
                    "partial_coverage": coverage.get("processed_slice_count", 0) < coverage.get("total_slice_count", 0),
                },
                "negative_scope": {"kind": "invalid_semantic_document_extraction", "invalid": invalid[:20]},
            }
        return {
            "ok": True,
            "tool_name": "extract_semantic_records",
            "summary": f"Validated {len(normalized_records)} semantic record(s).",
            "payload": {
                "records": normalized_records,
                "record_schema": target_schema,
                "slice_decisions": decisions,
                "validation_warnings": validation_warnings,
                "processed_slice_ids": sorted(processed),
                "no_relevant_slice_ids": sorted(no_relevant),
                "ambiguous_slice_ids": sorted(ambiguous),
                "coverage_summary": coverage,
                "partial_coverage": coverage.get("processed_slice_count", 0) < coverage.get("total_slice_count", 0),
            },
        }

    def check_document_coverage(
        self,
        state: LoopState,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        processed = tuple(str(item) for item in arguments.get("processed_slice_ids") or [] if str(item).strip())
        coverage = self._coverage_summary(state, processed_slice_ids=processed)
        return {
            "ok": True,
            "tool_name": "check_document_coverage",
            "summary": "Checked document-agent slice coverage.",
            "payload": coverage,
        }

    def _build_index(self, source: SourceRef) -> DocumentRecordIndex:
        if source.data_form == "markdown_document":
            raw_slices = self._markdown_raw_slices(source)
        elif source.data_form == "pdf_document":
            raw_slices = self._pdf_raw_slices(source)
        else:
            raw_slices = []
        slices: list[RecordSlice] = []
        for index, raw in enumerate(raw_slices, start=1):
            slices.append(
                RecordSlice(
                    slice_id=_slice_id(source, index),
                    source_id=source.id,
                    path=source.virtual_path,
                    data_form=source.data_form,
                    slice_index=index,
                    text=str(raw["text"]),
                    page_start=raw.get("page_start"),
                    page_end=raw.get("page_end"),
                    line_start=raw.get("line_start"),
                    line_end=raw.get("line_end"),
                    block_start=raw.get("block_start"),
                    block_end=raw.get("block_end"),
                    anchor=str(raw.get("anchor") or ""),
                )
            )
        linked: list[RecordSlice] = []
        for index, item in enumerate(slices):
            linked.append(
                RecordSlice(
                    **{
                        **asdict(item),
                        "previous_slice_id": slices[index - 1].slice_id if index > 0 else None,
                        "next_slice_id": slices[index + 1].slice_id if index + 1 < len(slices) else None,
                    }
                )
            )
        return DocumentRecordIndex(
            source_id=source.id,
            path=source.virtual_path,
            data_form=source.data_form,
            slice_count=len(linked),
            page_count=_page_count(source),
            slices=tuple(linked),
        )

    def _markdown_raw_slices(self, source: SourceRef) -> list[dict[str, Any]]:
        text = source.path.read_text(encoding="utf-8", errors="replace")
        raw_lines = text.splitlines()
        slices: list[dict[str, Any]] = []
        current: list[str] = []
        line_start: int | None = None
        for line_number, line in enumerate(raw_lines, start=1):
            if line.strip():
                if line_start is None:
                    line_start = line_number
                current.append(line)
                continue
            if current:
                paragraph = "\n".join(current).strip()
                for part in _split_long_text(paragraph):
                    slices.append(
                        {
                            "text": part,
                            "line_start": line_start,
                            "line_end": line_number - 1,
                            "anchor": _compact_text(current[0], limit=80),
                        }
                    )
                current = []
                line_start = None
        if current:
            paragraph = "\n".join(current).strip()
            for part in _split_long_text(paragraph):
                slices.append(
                    {
                        "text": part,
                        "line_start": line_start,
                        "line_end": len(raw_lines),
                        "anchor": _compact_text(current[0], limit=80),
                    }
                )
        return slices

    def _pdf_raw_slices(self, source: SourceRef) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        with fitz.open(source.path) as document:
            global_block = 0
            for page_index, page in enumerate(document, start=1):
                page_blocks = page.get_text("blocks")
                page_blocks = sorted(page_blocks, key=lambda item: (float(item[1]), float(item[0])))
                for block in page_blocks:
                    text = str(block[4] if len(block) > 4 else "").strip()
                    if not text:
                        continue
                    global_block += 1
                    blocks.append(
                        {
                            "text": text,
                            "page": page_index,
                            "block": global_block,
                            "anchor": _compact_text(text.splitlines()[0], limit=80),
                        }
                    )
        slices: list[dict[str, Any]] = []
        current: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal current
            if not current:
                return
            text = "\n".join(str(item["text"]).strip() for item in current if str(item["text"]).strip())
            for part in _split_long_text(text):
                slices.append(
                    {
                        "text": part,
                        "page_start": current[0]["page"],
                        "page_end": current[-1]["page"],
                        "block_start": current[0]["block"],
                        "block_end": current[-1]["block"],
                        "anchor": current[0]["anchor"],
                    }
                )
            current = []

        for block in blocks:
            text = str(block["text"])
            current_len = sum(len(str(item["text"])) for item in current)
            if current and current_len + len(text) > _MAX_SLICE_CHARS:
                flush()
            current.append(block)
        flush()
        return slices

    def _coverage_summary(
        self,
        state: LoopState,
        *,
        processed_slice_ids: tuple[str, ...],
        source_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        indexes = self.ensure_indexes(state)
        selected_indexes = [
            index
            for source_id, index in indexes.items()
            if source_ids is None or source_id in set(source_ids)
        ]
        all_slices = [item for index in selected_indexes for item in index.slices]
        processed = set(processed_slice_ids)
        summary = {
            "document_count": len(selected_indexes),
            "total_slice_count": len(all_slices),
            "processed_slice_count": sum(1 for item in all_slices if item.slice_id in processed),
            "unprocessed_slice_count": sum(1 for item in all_slices if item.slice_id not in processed),
            "source_slice_counts": {
                index.source_id: index.slice_count for index in selected_indexes
            },
        }
        state.document_coverage = summary
        return summary

    def _run_deterministic(self, state: LoopState, task: DocTask) -> DocEvidencePackage:
        search = self.search_document_records(
            state,
            {
                "query": task.question,
                "semantic_fields": list(task.target_fields),
                "source_candidates": list(task.source_candidates),
                "limit": task.coverage_policy.get("initial_slice_limit") or 8,
            },
        )
        matches = search.get("payload", {}).get("matches") or []
        selected_slice_ids = tuple(str(item["slice_id"]) for item in matches if isinstance(item, dict))
        extract = self.extract_semantic_records(
            state,
            {
                "records": list(task.records),
                "slice_decisions": list(task.slice_decisions)
                or [
                    {"slice_id": slice_id, "status": "ambiguous", "reason": "no worker model supplied"}
                    for slice_id in selected_slice_ids
                ],
                "target_schema": {
                    "fields": list(task.target_fields),
                    "record_grain": task.required_record_grain,
                },
            },
        )
        payload = extract.get("payload", {})
        remaining_risks = []
        if not task.records:
            remaining_risks.append("semantic_extraction_requires_document_agent_model_or_provided_records")
        if extract.get("negative_scope"):
            remaining_risks.append(str(extract["negative_scope"].get("kind") or "document_agent_validation_failed"))
        uncertain_slices = tuple(
            {
                "slice_id": str(item.get("slice_id") or ""),
                "source_id": str(item.get("source_id") or ""),
                "path": str(item.get("path") or ""),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "line_start": item.get("line_start"),
                "line_end": item.get("line_end"),
                "reason": "deterministic path found a candidate slice but no LLM extraction records were supplied",
                "candidate_fields": list(task.target_fields),
                "evidence_text": _compact_text(item.get("preview") or item.get("text") or "", limit=900),
            }
            for item in matches[:40]
            if isinstance(item, dict)
        )
        return DocEvidencePackage(
            records=tuple(payload.get("records") or []),
            record_schema=payload.get("record_schema") or {"fields": list(task.target_fields)},
            source_refs=tuple(sorted({str(item.get("source_id")) for item in matches if isinstance(item, dict)})),
            evidence_refs=tuple(selected_slice_ids),
            processed_slice_ids=tuple(payload.get("processed_slice_ids") or selected_slice_ids),
            no_relevant_slice_ids=tuple(payload.get("no_relevant_slice_ids") or []),
            ambiguous_slice_ids=tuple(payload.get("ambiguous_slice_ids") or selected_slice_ids),
            coverage_summary=payload.get("coverage_summary") or self._coverage_summary(state, processed_slice_ids=selected_slice_ids),
            remaining_risks=tuple(remaining_risks),
            uncertain_slices=uncertain_slices,
        )

    def _run_scan_loop(self, state: LoopState, task: DocTask) -> DocEvidencePackage:
        indexes = self.ensure_indexes(state)
        selected_indexes = _selected_document_indexes(state, indexes, task)
        selected_slices = [item for index in selected_indexes for item in index.slices]
        focus_slice_ids = _focus_slice_ids(task)
        if focus_slice_ids:
            focus_set = set(focus_slice_ids)
            selected_slices = [item for item in selected_slices if item.slice_id in focus_set]
        selected_source_ids = tuple(dict.fromkeys(index.source_id for index in selected_indexes))
        if not selected_slices:
            return DocEvidencePackage(
                records=(),
                record_schema={"fields": list(task.target_fields), "record_grain": task.required_record_grain},
                source_refs=selected_source_ids,
                evidence_refs=(),
                processed_slice_ids=(),
                no_relevant_slice_ids=(),
                ambiguous_slice_ids=(),
                coverage_summary={"document_count": 0, "total_slice_count": 0, "processed_slice_count": 0},
                remaining_risks=("document_agent_no_document_slices",),
            )

        tool_model = self.model.bind_tools(_DOC_SCAN_TOOL_SPECS, parallel_tool_calls=False)
        batch_size = max(1, min(int(task.coverage_policy.get("scan_batch_size") or 20), 80))
        max_chars = max(2_000, min(int(task.coverage_policy.get("scan_batch_chars") or 16_000), 80_000))
        field_terms = _task_candidate_terms(task)
        candidate_fields_by_slice = {
            item.slice_id: _candidate_fields_for_slice(item.text, field_terms)
            for item in selected_slices
        }
        cursor = 0
        batch_index = 0
        accumulated_records: list[dict[str, Any]] = []
        accumulated_decisions: list[dict[str, Any]] = []
        trace_events: list[dict[str, Any]] = []

        while cursor < len(selected_slices):
            batch, next_cursor = _next_scan_batch(
                selected_slices,
                cursor,
                max_slices=batch_size,
                max_chars=max_chars,
            )
            batch_index += 1
            allowed_slice_ids = {item.slice_id for item in batch}
            package_context = {
                "doc_task": task.to_dict(),
                "scan_progress": {
                    "batch_index": batch_index,
                    "cursor": cursor,
                    "next_cursor": next_cursor,
                    "processed_before_batch": cursor,
                    "total_slices": len(selected_slices),
                    "remaining_after_batch": len(selected_slices) - next_cursor,
                },
                "instructions": (
                    "Read every slice in this batch. Call record_document_decisions exactly once. "
                    "For each slice, decide whether it contains evidence for any target field in the "
                    "document task by semantic meaning, not by whether canonical field labels appear. "
                    "Record partial records: a slice does not need to contain all target fields to be "
                    "useful or final-answer-ready. If it narratively states a value for any target field, "
                    "record a record_extracted partial record under the target field key with exact "
                    "evidence-supported text. If it contains a date only, record the date; if it contains "
                    "a metric only, record the metric; if it contains an identifier/record code that can "
                    "anchor a later merge, put that visible code in record_anchor even when no other "
                    "target field is present. Use "
                    "canonical target field names as output keys when possible and cite the slice_id. "
                    "If a document spreads one logical record across sections, use "
                    "the same stable record_anchor explicitly visible in the text so the ledger can merge "
                    "fields mechanically across batches. Do not copy headings, row labels, or record anchors "
                    "into target fields unless the text explicitly identifies that label as the target field; "
                    "put those values in record_anchor instead. Do not invent missing fields. Do not compute, "
                    "sum, average, compare, or derive a target value from components during document extraction; "
                    "record only values explicitly stated for the target field in the slice text. Do not normalize "
                    "dates or numbers unless you also keep the evidence-supported value in the record or provenance; "
                    "downstream compute can make explicit transformations later. Do not mark a slice ambiguous "
                    "merely because it has only one target field or lacks enough fields to answer the whole task; "
                    "partial target evidence should be a record_extracted partial record. Mark ambiguous only when "
                    "you cannot decide whether the text supports a target field at all, and include candidate_fields "
                    "plus a short evidence_text excerpt so the main loop can continue from that recorded evidence. "
                    "candidate_semantic_fields are lexical hints from the declared task, not extracted facts; use "
                    "your semantic judgment to decide whether the slice actually supports those fields."
                ),
                "slices": [
                    {
                        **item.public_dict(include_text=True),
                        "candidate_semantic_fields": list(candidate_fields_by_slice.get(item.slice_id, ())),
                    }
                    for item in batch
                ],
            }
            messages: list[Any] = [
                SystemMessage(content="You are a bounded DocumentAgent scan worker."),
                HumanMessage(content=json.dumps(package_context, ensure_ascii=False, default=str)),
            ]
            response = tool_model.invoke(messages)
            calls = extract_tool_calls(response)
            trace_events.append(
                {
                    "step": batch_index,
                    "event": "model_response",
                    "tool_calls": [str(call.get("name") or "") for call in calls],
                    "content": _compact_text(getattr(response, "content", ""), limit=240),
                    "scan_cursor": cursor,
                    "next_cursor": next_cursor,
                }
            )
            if calls and str(calls[0].get("name") or "") == "record_document_decisions":
                args = calls[0].get("args") if isinstance(calls[0].get("args"), dict) else {}
            else:
                args = {
                    "records": [],
                    "slice_decisions": [
                        {
                            "slice_id": item.slice_id,
                            "status": "ambiguous",
                            "reason": "LLM did not call record_document_decisions for this scan batch.",
                        }
                        for item in batch
                    ],
                }
            result = _record_document_decisions_result(dict(args), allowed_slice_ids=allowed_slice_ids)
            payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
            accumulated_records.extend(payload.get("records") or [])
            accumulated_decisions.extend(payload.get("slice_decisions") or [])
            trace_events.append(
                {
                    "step": batch_index,
                    "event": "tool_result",
                    "tool": "record_document_decisions",
                    "arguments": _summarize_doc_tool_args("record_document_decisions", dict(args)),
                    "result": _summarize_doc_tool_result("record_document_decisions", result),
                }
            )
            cursor = next_cursor

        candidate_review_risks = _candidate_review_snippets(
            selected_slices,
            accumulated_decisions,
            candidate_fields_by_slice,
            limit=int(task.coverage_policy.get("uncertain_slice_limit") or 40),
        )
        merged_records = _merge_scan_records(accumulated_records)
        validation = self.extract_semantic_records(
            state,
            {
                "records": merged_records,
                "slice_decisions": accumulated_decisions,
                "target_schema": {
                    "fields": list(task.target_fields),
                    "record_grain": task.required_record_grain,
                },
            },
        )
        payload = validation.get("payload", {}) if isinstance(validation.get("payload"), dict) else {}
        processed = tuple(str(item) for item in payload.get("processed_slice_ids") or [])
        candidate_record_gaps = _candidate_record_gap_snippets(
            selected_slices,
            accumulated_decisions,
            list(payload.get("records") or []),
            candidate_fields_by_slice,
            limit=int(task.coverage_policy.get("uncertain_slice_limit") or 40),
        )
        coverage = (
            _coverage_summary_for_slices(selected_slices, processed_slice_ids=processed)
            if focus_slice_ids
            else self._coverage_summary(
                state,
                processed_slice_ids=processed,
                source_ids=selected_source_ids,
            )
        )
        if isinstance(payload, dict):
            payload["coverage_summary"] = coverage
            payload["partial_coverage"] = coverage.get("processed_slice_count", 0) < coverage.get("total_slice_count", 0)
            payload["scan_batches"] = batch_index
            payload["scan_record_count_before_merge"] = len(accumulated_records)
            payload["scan_record_count_after_merge"] = len(merged_records)
            ambiguous_snippets = _decision_slice_snippets(
                selected_slices,
                accumulated_decisions,
                limit=int(task.coverage_policy.get("uncertain_slice_limit") or 40),
            )
            payload["uncertain_slices"] = list(
                (*candidate_record_gaps, *candidate_review_risks, *ambiguous_snippets)[
                    : int(task.coverage_policy.get("uncertain_slice_limit") or 40)
                ]
            )
            if candidate_record_gaps or candidate_review_risks:
                payload["candidate_review_risks"] = list((*candidate_record_gaps, *candidate_review_risks))
                payload["partial_coverage"] = True
        uncertain_slices = tuple(payload.get("uncertain_slices") or [])
        validation_warnings = tuple(payload.get("validation_warnings") or [])

        if not validation.get("ok"):
            invalid = payload.get("invalid") if isinstance(payload.get("invalid"), list) else []
            remaining_risks = ["document_scan_validation_failed", f"invalid_extract_items:{len(invalid)}"]
            if candidate_record_gaps or candidate_review_risks:
                remaining_risks.append("unresolved_candidate_document_slices")
            return DocEvidencePackage(
                records=tuple(payload.get("records") or []),
                record_schema=payload.get("target_schema") or {"fields": list(task.target_fields)},
                source_refs=selected_source_ids,
                evidence_refs=processed,
                processed_slice_ids=processed,
                no_relevant_slice_ids=tuple(payload.get("no_relevant_slice_ids") or []),
                ambiguous_slice_ids=tuple(payload.get("ambiguous_slice_ids") or []),
                coverage_summary=coverage,
                remaining_risks=tuple(remaining_risks),
                validation_warnings=validation_warnings,
                uncertain_slices=uncertain_slices,
                agent_trace=tuple(trace_events),
            )

        return DocEvidencePackage(
            records=tuple(payload.get("records") or []),
            record_schema=payload.get("record_schema") or {"fields": list(task.target_fields)},
            source_refs=selected_source_ids,
            evidence_refs=processed,
            processed_slice_ids=processed,
            no_relevant_slice_ids=tuple(payload.get("no_relevant_slice_ids") or []),
            ambiguous_slice_ids=tuple(payload.get("ambiguous_slice_ids") or []),
            coverage_summary=coverage,
            remaining_risks=tuple(
                item
                for item, present in (
                    ("partial_document_coverage", bool(payload.get("partial_coverage"))),
                    ("unresolved_candidate_document_slices", bool(candidate_record_gaps or candidate_review_risks)),
                    ("ambiguous_document_slices", bool(payload.get("ambiguous_slice_ids"))),
                )
                if present
            ),
            validation_warnings=validation_warnings,
            uncertain_slices=uncertain_slices,
            agent_trace=tuple(trace_events),
        )

def _object_schema(
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
    additional_properties: bool = False,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": additional_properties,
    }


_DOC_SCAN_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "record_document_decisions",
            "description": (
                "Record the LLM's decisions for the current document scan batch. "
                "The tool stores records and per-slice decisions only; it does not infer "
                "field meaning, normalize values, choose sources, or compute answers."
            ),
            "parameters": _object_schema(
                {
                    "records": {
                        "type": "array",
                        "description": (
                            "Partial semantic records extracted from the current batch. Include a record whenever "
                            "any target field or merge anchor is supported by the slice; do not wait for all target "
                            "fields to appear in the same slice. Values must be explicit spans supported by the "
                            "slice text, not calculations or normalized rewrites."
                        ),
                        "items": _object_schema(
                            {
                                "record_anchor": {
                                    "type": "string",
                                    "description": "Stable identifier visible in the slice text for cross-section merge.",
                                },
                                "slice_id": {"type": "string"},
                                "slice_ids": {"type": "array", "items": {"type": "string"}},
                                "provenance": {
                                    "type": "object",
                                    "properties": {
                                        "slice_id": {"type": "string"},
                                        "slice_ids": {"type": "array", "items": {"type": "string"}},
                                        "evidence_text": {"type": "string"},
                                    },
                                    "additionalProperties": True,
                                },
                            },
                            additional_properties=True,
                        ),
                    },
                    "slice_decisions": {
                        "type": "array",
                        "items": _object_schema(
                            {
                                "slice_id": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["record_extracted", "no_relevant_record", "ambiguous"],
                                },
                                "reason": {"type": "string"},
                                "candidate_fields": {"type": "array", "items": {"type": "string"}},
                                "evidence_text": {
                                    "type": "string",
                                    "description": "Short text excerpt from this slice that made it ambiguous or relevant.",
                                },
                            },
                            required=["slice_id", "status"],
                            additional_properties=True,
                        ),
                    },
                    "scan_cursor": {"type": "integer"},
                    "notes": {"type": "string"},
                },
                required=["slice_decisions"],
            ),
        },
    }
]
