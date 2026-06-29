from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field as dataclass_field
from typing import Any

import fitz
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

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
    if name == "extract_records_by_plan":
        plan = arguments.get("plan") if isinstance(arguments.get("plan"), dict) else arguments
        return {
            "source_ref": plan.get("source_ref"),
            "source_candidates": list(plan.get("source_candidates") or []),
            "target_fields": list(plan.get("target_fields") or []),
            "record_anchor": plan.get("entity_anchor") or plan.get("record_anchor"),
            "section_scope": plan.get("section_scope") or plan.get("scope"),
            "fields": list((plan.get("fields") or plan.get("field_specs") or {}).keys()),
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
    if name == "inspect_document_index":
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
    elif name == "extract_records_by_plan":
        summary.update(
            {
                "record_count": len(payload.get("records") or []),
                "processed_slice_ids": list(payload.get("processed_slice_ids") or []),
                "invalid_count": len(payload.get("invalid") or []),
                "plan_status": payload.get("plan_status"),
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


def _field_alias_values_from_card(card: dict[str, Any], field: str) -> list[str]:
    raw_values: list[Any] = [
        field,
        card.get("canonical_field"),
        card.get("name"),
        card.get("formula"),
        *(card.get("aliases") or []),
        *(card.get("join_keys") or []),
    ]
    return _ordered_unique(str(value) for value in raw_values if str(value or "").strip())


def _field_aliases_from_task(task: DocTask) -> dict[str, list[str]]:
    target_fields = [str(item) for item in task.target_fields if str(item).strip()]
    by_field: dict[str, list[str]] = {field: [field] for field in target_fields}
    target_lookup = {_normalize_field_name(field): field for field in target_fields}
    for card in task.semantic_cards:
        canonical = str(card.get("canonical_field") or "").strip()
        field = target_lookup.get(_normalize_field_name(canonical))
        if not field:
            continue
        aliases = _field_alias_values_from_card(card, field)
        by_field[field] = _ordered_unique([*by_field.get(field, []), *aliases])
    return by_field


def _source_for_ref(state: LoopState, ref: str) -> SourceRef | None:
    if ref in state.sources:
        return state.sources[ref]
    source_id = state.source_by_path.get(ref)
    if source_id:
        return state.sources.get(source_id)
    return None


def _card_table_and_field(card: dict[str, Any]) -> tuple[str, str]:
    table = str(card.get("canonical_table") or "").strip()
    field = str(card.get("canonical_field") or "").strip()
    name = str(card.get("name") or "").strip()
    if (not table or not field) and "." in name:
        tail = name.split(">")[-1].strip()
        if "." in tail:
            maybe_table, maybe_field = tail.rsplit(".", 1)
            table = table or maybe_table.strip("` ")
            field = field or maybe_field.strip("` ")
    return table, field


def _document_required_fields(state: LoopState, task: DocTask) -> list[str]:
    target_by_norm = {_normalize_field_name(field): field for field in task.target_fields}
    document_stems = {
        _normalize_field_name(source.stem)
        for ref in task.source_candidates
        if (source := _source_for_ref(state, str(ref))) is not None and source.data_form in _DOCUMENT_FORMS
    }
    required: list[str] = []
    for card in task.semantic_cards:
        table, field = _card_table_and_field(card)
        if not table or not field:
            continue
        canonical = target_by_norm.get(_normalize_field_name(field))
        if canonical and _normalize_field_name(table) in document_stems:
            required.append(canonical)
    explicit = task.coverage_policy.get("required_fields")
    if isinstance(explicit, (list, tuple, set)):
        required.extend(str(item) for item in explicit if str(item).strip())
    return _ordered_unique(required)


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
    key_terms = semantic_terms(field_name)
    scored: list[tuple[int, str]] = []
    for field, aliases in aliases_by_field.items():
        alias_norms = {_normalize_field_name(alias) for alias in aliases}
        if normalized in alias_norms:
            return field
        alias_terms: set[str] = set()
        for alias in aliases:
            alias_terms.update(semantic_terms(alias))
        overlap = key_terms & alias_terms
        strong_overlap = {term for term in overlap if len(term) >= 3}
        if strong_overlap:
            scored.append((len(strong_overlap), field))
    scored.sort(reverse=True)
    if len(scored) == 1 or (scored and (len(scored) == 1 or scored[0][0] > scored[1][0])):
        return scored[0][1]
    return None


def _compact_evidence_value(value: Any) -> str:
    return re.sub(r"\s+", "", str(value).casefold())


def _normalized_number(value: str) -> str:
    if "." not in value:
        return value.lstrip("0") or "0"
    whole, frac = value.split(".", 1)
    return f"{whole.lstrip('0') or '0'}.{frac.rstrip('0')}".rstrip(".")


def _evidence_supports_value(value: Any, source_text: str) -> bool:
    compact_value = _compact_evidence_value(value)
    if not compact_value:
        return True
    compact_source = _compact_evidence_value(source_text)
    if compact_value in compact_source:
        return True
    value_numbers = re.findall(r"\d+(?:\.\d+)?", str(value))
    if not value_numbers:
        return False
    source_numbers = {_normalized_number(item) for item in re.findall(r"\d+(?:\.\d+)?", source_text)}
    return all(_normalized_number(item) in source_numbers for item in value_numbers)


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


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Any = [value]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _read_source_text(source: SourceRef) -> str:
    if source.data_form == "markdown_document":
        return source.path.read_text(encoding="utf-8", errors="replace")
    if source.data_form == "pdf_document":
        with fitz.open(source.path) as document:
            return "\n".join(page.get_text() for page in document)
    return ""


def _default_anchor_labels(text: str) -> list[str]:
    labels = ["档案", "记录", "Record", "Archive"]
    scored = [(len(re.findall(rf"{re.escape(label)}\s*[:：#]?\s*[0-9A-Za-z_-]+", text, re.I)), label) for label in labels]
    return [label for count, label in sorted(scored, reverse=True) if count >= 3]


def _anchor_regex(plan: dict[str, Any], text: str) -> re.Pattern[str] | None:
    anchor = _dict_value(plan.get("entity_anchor") or plan.get("record_anchor"))
    pattern = str(anchor.get("pattern") or "").strip()
    if pattern:
        try:
            return re.compile(pattern, re.I)
        except re.error:
            return None
    labels = _string_items(anchor.get("labels") or anchor.get("label"))
    labels.extend(_string_items(plan.get("merge_key")))
    labels.extend(_default_anchor_labels(text))
    ordered_labels = _ordered_unique(labels)
    explicit_labels = _ordered_unique(_string_items(anchor.get("labels") or anchor.get("label")))
    if len(explicit_labels) > 1:
        alternatives: list[str] = []
        for label in explicit_labels:
            if re.fullmatch(r"[\w_-]+", label):
                alternatives.append(rf"{re.escape(label)}\s*[:：#]?\s*([0-9A-Za-z_-]+)")
            else:
                alternatives.append(rf"{re.escape(label)}\s*([0-9A-Za-z_-]+)")
        compiled = re.compile("(?:%s)" % "|".join(alternatives), re.I)
        if len(compiled.findall(text)) >= 3:
            return compiled
    for label in ordered_labels:
        if not label:
            continue
        if re.fullmatch(r"[\w_-]+", label):
            pattern = rf"{re.escape(label)}\s*[:：#]?\s*([0-9A-Za-z_-]+)"
        else:
            pattern = rf"{re.escape(label)}\s*([0-9A-Za-z_-]+)"
        compiled = re.compile(pattern, re.I)
        if len(compiled.findall(text)) >= 3:
            return compiled
    return None


def _anchor_id(match: re.Match[str]) -> str:
    if "id" in match.groupdict():
        return str(match.group("id"))
    groups = [group for group in match.groups() if group is not None]
    return str(groups[-1] if groups else match.group(0)).strip()


def _record_chunks(text: str, pattern: re.Pattern[str], *, offset: int = 0) -> list[dict[str, Any]]:
    matches = list(pattern.finditer(text))
    chunks: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunks.append(
            {
                "merge_id": _anchor_id(match),
                "text": text[match.start() : end],
                "start": offset + match.start(),
                "end": offset + end,
            }
        )
    return chunks


def _hint_position(text: str, hints: list[str], *, start: int = 0) -> int | None:
    lowered = text.casefold()
    positions = [
        lowered.find(hint.casefold(), start)
        for hint in hints
        if hint and lowered.find(hint.casefold(), start) >= 0
    ]
    return min(positions) if positions else None


def _scoped_text_span(
    text: str,
    *,
    scope: dict[str, Any],
    anchor_pattern: re.Pattern[str] | None,
) -> tuple[str, int]:
    include_hints = _string_items(
        scope.get("include_hints")
        or scope.get("include")
        or scope.get("start_hints")
        or scope.get("scope_hints")
    )
    start_after_hints = _string_items(scope.get("start_after_hints") or scope.get("after_hints"))
    exclude_hints = _string_items(scope.get("exclude_hints") or scope.get("exclude") or scope.get("end_hints"))
    start = 0
    if start_after_hints and (after_pos := _hint_position(text, start_after_hints)) is not None:
        start = after_pos
    if include_hints and (include_pos := _hint_position(text, include_hints, start=start)) is not None:
        start = include_pos
    if anchor_pattern is not None:
        previous_anchor = None
        for match in anchor_pattern.finditer(text, 0, start):
            previous_anchor = match.start()
        if previous_anchor is not None and "\n\n" not in text[previous_anchor:start]:
            start = previous_anchor
    end = len(text)
    if exclude_hints and (exclude_pos := _hint_position(text, exclude_hints, start=max(start + 1, 1))) is not None:
        end = exclude_pos
    return text[start:end], start


def _field_spec_map(arguments: dict[str, Any]) -> dict[str, dict[str, Any]]:
    plan = _dict_value(arguments.get("plan")) or arguments
    raw_fields = _dict_value(plan.get("fields") or plan.get("field_specs"))
    specs: dict[str, dict[str, Any]] = {}
    for field, raw in raw_fields.items():
        specs[str(field)] = _dict_value(raw)
    return specs


def _field_value_type(field: str, spec: dict[str, Any]) -> str:
    normalized = _normalize_field_name(field)
    if "code" in normalized or normalized.endswith("id") or "identifier" in normalized:
        return "identifier"
    if any(term in normalized for term in ("amount", "aum", "value", "scale", "nv", "asset", "return", "rr")):
        return "amount"
    value_type = str(spec.get("value_type") or spec.get("type") or "").strip().casefold()
    if value_type:
        return value_type
    return "text"


def _value_aliases(field: str, spec: dict[str, Any], aliases_by_field: dict[str, set[str]]) -> list[str]:
    aliases = [field, *aliases_by_field.get(field, set())]
    aliases.extend(_string_items(spec.get("aliases") or spec.get("alias")))
    aliases.extend(_string_items(spec.get("labels") or spec.get("label")))
    return _ordered_unique(aliases)


def _extract_identifier(chunk: str, aliases: list[str]) -> Any | None:
    def score_candidate(value: str, start: int, end: int, alias: str) -> tuple[int, int]:
        context = chunk[max(0, start - 48) : min(len(chunk), end + 48)]
        score = 0
        if re.search(r"正确|正式|最终|修正|确认|核实|应为", context):
            score += 5
        if re.search(r"错误|误记|过时|初步|暂记|临时|曾将|曾被", context):
            score -= 5
        if re.search(r"[A-Za-z]", alias):
            score += 1
        return score, start

    after_alias: list[tuple[tuple[int, int], str]] = []
    before_alias: list[tuple[tuple[int, int], str]] = []
    for alias in aliases:
        if not alias:
            continue
        escaped = re.escape(alias)
        for match in re.finditer(rf"{escaped}\D{{0,32}}([0-9A-Za-z_-]{{4,}})", chunk, flags=re.I):
            value = match.group(1)
            after_alias.append((score_candidate(value, match.start(1), match.end(1), alias), value))
        for match in re.finditer(rf"([0-9A-Za-z_-]{{4,}})\D{{0,24}}{escaped}", chunk, flags=re.I):
            value = match.group(1)
            before_alias.append((score_candidate(value, match.start(1), match.end(1), alias), value))
    values = after_alias or before_alias
    if not values:
        return None
    value = max(values, key=lambda item: item[0])[1]
    return int(value) if value.isdigit() else value


def _unit_regex(unit: str) -> str:
    normalized = unit.strip().casefold()
    if normalized in {"percentage", "percent", "%", "百分比"}:
        return r"(?:%|百分比|百分点)"
    if normalized in {"currency", "amount", "money"}:
        return r"(?:亿元|万元|元|万)"
    if unit:
        return re.escape(unit)
    return r"(?:亿元|万元|元|万|%|百分比|百分点)?"


def _amount_candidate_score(
    chunk: str,
    *,
    alias: str,
    alias_start: int,
    alias_end: int,
    value_start: int,
    value_end: int,
) -> tuple[int, int, int]:
    distance = min(abs(value_start - alias_end), abs(alias_start - value_end))
    wider = chunk[max(0, value_start - 64) : min(len(chunk), value_end + 64)]
    alias_haystack = alias.casefold()
    score = 1_000 - distance
    if value_start >= alias_start:
        score += 25
    if re.search(r"最终|确认|核准|审计|修正|正式|应为|确认为", wider):
        score += 250
    if re.search(r"错误|误记|初步|暂定|临时|曾为|修正前", wider):
        score -= 180
    before_start = max(0, value_start - 30)
    if value_start >= alias_start:
        before_start = max(before_start, alias_start)
    before_value = chunk[before_start:value_start]
    if re.search(r"最终|确认|核准|审计|修正|正式|应为|确认为", before_value):
        score += 300
    if re.search(r"错误|误记|初步|暂定|临时|曾为|修正前", before_value):
        score -= 260
    if re.search(r"年化|ann", before_value, flags=re.I) and not re.search(r"年化|annual|ann", alias_haystack, flags=re.I):
        score -= 300
    return score, -distance, value_start


def _extract_amount(chunk: str, aliases: list[str], unit: str) -> Any | None:
    unit_pattern = _unit_regex(unit)
    amount_pattern = re.compile(rf"(?<![0-9.])(-?\d+(?:\.\d+)?)\s*{unit_pattern}", re.I)
    after_alias: list[tuple[tuple[int, int, int], str]] = []
    before_alias: list[tuple[tuple[int, int, int], str]] = []
    usable_aliases = [alias for alias in aliases if str(alias).strip()]
    alias_seen = False
    missing_after_alias = False
    for alias in usable_aliases:
        if not alias:
            continue
        escaped = re.escape(alias)
        for match in re.finditer(escaped, chunk, flags=re.I):
            alias_seen = True
            if re.search(r"缺失|NaN|无法获取|无法获得|无法评估|未有记录|无记录|暂无|为空", chunk[match.start() : match.end() + 96]):
                missing_after_alias = True
            window_start = max(0, match.start() - 48)
            window_end = min(len(chunk), match.end() + 160)
            for value_match in amount_pattern.finditer(chunk, window_start, window_end):
                value = value_match.group(1)
                candidate = (
                    _amount_candidate_score(
                        chunk,
                        alias=alias,
                        alias_start=match.start(),
                        alias_end=match.end(),
                        value_start=value_match.start(1),
                        value_end=value_match.end(1),
                    ),
                    value,
                )
                if value_match.start(1) >= match.start():
                    after_alias.append(candidate)
                else:
                    before_alias.append(candidate)
    scored = after_alias or before_alias
    if alias_seen and missing_after_alias and not after_alias:
        return None
    if not scored:
        if usable_aliases and re.search(r"缺失|NaN|无法获取|无法获得|无法评估|未有记录|无记录|暂无|为空", chunk):
            return None
        scored = [((0, 0, match.start(1)), match.group(1)) for match in amount_pattern.finditer(chunk)]
    if not scored:
        return None
    return float(max(scored, key=lambda item: item[0])[1])


def _extract_text_value(chunk: str, aliases: list[str]) -> Any | None:
    for alias in aliases:
        if alias and re.search(re.escape(alias), chunk, re.I):
            return alias
    return None


def _extract_planned_field(chunk: str, *, field: str, spec: dict[str, Any], aliases: list[str]) -> Any | None:
    value_type = _field_value_type(field, spec)
    if value_type in {"identifier", "id", "code"}:
        return _extract_identifier(chunk, aliases)
    if value_type in {"amount", "number", "numeric", "currency", "measure"}:
        return _extract_amount(chunk, aliases, str(spec.get("unit") or "").strip())
    return _extract_text_value(chunk, aliases)


def _supporting_slice_ids_for_chunk(index: DocumentRecordIndex, chunk: str, value: Any) -> list[str]:
    compact_chunk = _compact_evidence_value(chunk)
    chunk_matches: list[RecordSlice] = []
    if compact_chunk:
        for item in index.slices:
            compact_slice = _compact_evidence_value(item.text)
            if not compact_slice:
                continue
            if compact_slice in compact_chunk or compact_chunk in compact_slice:
                chunk_matches.append(item)
    if not chunk_matches:
        anchor_mentions = re.findall(
            r"(?:档案|记录|Record|Archive|战略单元)\s*[:：#]?\s*[0-9A-Za-z_-]+",
            chunk,
            flags=re.I,
        )
        compact_mentions = [_compact_evidence_value(item) for item in anchor_mentions if item.strip()]
        for record_slice in index.slices:
            compact_slice = _compact_evidence_value(record_slice.text)
            if compact_slice and any(mention and mention in compact_slice for mention in compact_mentions):
                chunk_matches.append(record_slice)
    if value is not None and str(value).strip():
        supported = [item.slice_id for item in chunk_matches if _evidence_supports_value(value, item.text)]
        if supported:
            return supported[:4]
        fallback = _supporting_slice_ids(index, value)
        if fallback:
            return fallback
    if chunk_matches:
        return [item.slice_id for item in chunk_matches[:4]]
    if value is None or not str(value).strip():
        return []
    return _supporting_slice_ids(index, value)


def _supporting_slice_ids(index: DocumentRecordIndex, value: Any) -> list[str]:
    matched = [
        item.slice_id
        for item in index.slices
        if _evidence_supports_value(value, item.text)
    ]
    return matched[:4]


def _planned_field_scope(plan: dict[str, Any], *, field: str, spec: dict[str, Any]) -> dict[str, Any]:
    field_scope = _dict_value(spec.get("section_scope") or spec.get("scope"))
    if field_scope:
        return field_scope
    if _field_value_type(field, spec) in {"identifier", "id", "code"}:
        return {}
    return _dict_value(plan.get("section_scope") or plan.get("scope"))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {
            "1",
            "true",
            "yes",
            "y",
            "include",
            "preserve",
            "keep",
            "allow",
            "empty",
            "missing",
        }
    return bool(value)


def _include_missing_records(plan: dict[str, Any], arguments: dict[str, Any]) -> bool:
    for container in (plan, arguments):
        if _truthy(container.get("include_missing_records") or container.get("allow_missing_fields")):
            return True
        policy = container.get("missing_value_policy")
        if isinstance(policy, dict):
            if _truthy(
                policy.get("include_missing_records")
                or policy.get("allow_missing_fields")
                or policy.get("preserve_empty_records")
            ):
                return True
        elif _truthy(policy):
            return True
    return False


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
            package = self._run_model_loop(state, task)
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
        if invalid:
            return {
                "ok": False,
                "tool_name": "extract_semantic_records",
                "summary": "Semantic document extraction failed validation.",
                "payload": {
                    "invalid": invalid,
                    "records": normalized_records,
                    "target_schema": target_schema,
                },
                "negative_scope": {"kind": "invalid_semantic_document_extraction", "invalid": invalid[:20]},
            }
        coverage = self._coverage_summary(state, processed_slice_ids=tuple(sorted(processed)))
        return {
            "ok": True,
            "tool_name": "extract_semantic_records",
            "summary": f"Validated {len(normalized_records)} semantic record(s).",
            "payload": {
                "records": normalized_records,
                "record_schema": target_schema,
                "slice_decisions": decisions,
                "processed_slice_ids": sorted(processed),
                "no_relevant_slice_ids": sorted(no_relevant),
                "ambiguous_slice_ids": sorted(ambiguous),
                "coverage_summary": coverage,
                "partial_coverage": bool(ambiguous) or coverage.get("processed_slice_count", 0) < coverage.get("total_slice_count", 0),
            },
        }

    def extract_records_by_plan(
        self,
        state: LoopState,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        indexes = self.ensure_indexes(state)
        plan = _dict_value(arguments.get("plan")) or dict(arguments)
        source_refs = _string_items(plan.get("source_ref") or plan.get("source_candidates"))
        if not source_refs:
            source_refs = _string_items(arguments.get("source_candidates"))
        selected_source: SourceRef | None = None
        selected_index: DocumentRecordIndex | None = None
        for ref in source_refs:
            source = _source_for_ref(state, ref)
            if source is None or source.data_form not in _DOCUMENT_FORMS:
                continue
            selected_source = source
            selected_index = indexes.get(source.id)
            break
        if selected_source is None or selected_index is None:
            return {
                "ok": False,
                "tool_name": "extract_records_by_plan",
                "summary": "Plan extraction requires a document source_ref/source_candidates entry.",
                "payload": {"arguments": arguments, "plan_status": "missing_document_source"},
                "negative_scope": {"kind": "document_plan_missing_source"},
            }

        text = _read_source_text(selected_source)
        anchor_pattern = _anchor_regex(plan, text)
        if anchor_pattern is None:
            return {
                "ok": False,
                "tool_name": "extract_records_by_plan",
                "summary": "Plan extraction could not identify a repeated record anchor.",
                "payload": {"plan": plan, "plan_status": "missing_record_anchor"},
                "negative_scope": {"kind": "document_plan_missing_record_anchor"},
            }

        target_fields = _ordered_unique(
            [
                *_string_items(plan.get("target_fields")),
                *_string_items(arguments.get("target_fields")),
                *_field_spec_map(arguments).keys(),
            ]
        )
        target_schema = _dict_value(arguments.get("target_schema"))
        if not target_fields:
            fields = target_schema.get("fields")
            target_fields = _ordered_unique(fields if isinstance(fields, list) else [])
        if not target_fields:
            return {
                "ok": False,
                "tool_name": "extract_records_by_plan",
                "summary": "Plan extraction requires target_fields or fields.",
                "payload": {"plan": plan, "plan_status": "missing_target_fields"},
                "negative_scope": {"kind": "document_plan_missing_target_fields"},
            }

        field_specs = _field_spec_map(arguments)
        aliases_by_field = _field_aliases_from_arguments(set(target_fields), arguments)
        include_missing_records = _include_missing_records(plan, arguments)
        records_by_id: dict[str, dict[str, Any]] = {}
        slice_ids_by_id: dict[str, set[str]] = {}
        order_by_id: dict[str, int] = {}

        for field in target_fields:
            spec = field_specs.get(field) or field_specs.get(_normalize_field_name(field)) or {}
            aliases = _value_aliases(field, spec, aliases_by_field)
            scope = _planned_field_scope(plan, field=field, spec=spec)
            field_text, offset = _scoped_text_span(text, scope=scope, anchor_pattern=anchor_pattern)
            for chunk in _record_chunks(field_text, anchor_pattern, offset=offset):
                value = _extract_planned_field(
                    chunk["text"],
                    field=field,
                    spec=spec,
                    aliases=aliases,
                )
                if value is None:
                    if not include_missing_records:
                        continue
                    value = ""
                supporting_slice_ids = _supporting_slice_ids_for_chunk(
                    selected_index,
                    str(chunk["text"]),
                    value,
                )
                if not supporting_slice_ids:
                    continue
                merge_id = str(chunk["merge_id"])
                record = records_by_id.setdefault(merge_id, {"record_anchor": merge_id})
                order_by_id.setdefault(merge_id, int(chunk.get("start") or 0))
                if field not in record or (record.get(field) in {None, ""} and value not in {None, ""}):
                    record[field] = value
                slice_ids_by_id.setdefault(merge_id, set()).update(supporting_slice_ids)

        required_fields = (
            _ordered_unique(_string_items(plan.get("required_fields")))
            if include_missing_records
            else _ordered_unique(
                _string_items(plan.get("required_fields"))
                or _string_items(target_schema.get("required"))
                or target_fields
            )
        )
        generated: list[dict[str, Any]] = []
        for merge_id, record in records_by_id.items():
            if any(field not in record or record[field] in {None, ""} for field in required_fields):
                continue
            slice_ids = sorted(slice_ids_by_id.get(merge_id) or [])
            if not slice_ids:
                continue
            source_slices = [item for item in selected_index.slices if item.slice_id in set(slice_ids)]
            generated.append(
                {
                    **record,
                    "slice_ids": slice_ids,
                    "provenance": {
                        "slice_ids": slice_ids,
                        "source_id": selected_source.id,
                        "path": selected_source.virtual_path,
                        "merge_key": plan.get("merge_key") or merge_id,
                        "merge_id": merge_id,
                        "evidence_text": _compact_text(
                            " ".join(item.preview for item in source_slices),
                            limit=500,
                        ),
                    },
                }
            )
        generated.sort(
            key=lambda item: (
                order_by_id.get(str(item.get("record_anchor") or ""), 10**12),
                str(item.get("record_anchor") or ""),
            )
        )
        if not generated:
            return {
                "ok": False,
                "tool_name": "extract_records_by_plan",
                "summary": (
                    "Plan extraction produced no records."
                    if include_missing_records
                    else "Plan extraction produced no complete records."
                ),
                "payload": {
                    "plan": plan,
                    "plan_status": "no_records" if include_missing_records else "no_complete_records",
                    "target_fields": target_fields,
                    "required_fields": required_fields,
                    "include_missing_records": include_missing_records,
                    "partial_record_count": len(records_by_id),
                },
                "negative_scope": {"kind": "document_plan_no_complete_records"},
            }

        processed_slice_ids = _ordered_unique(
            slice_id
            for record in generated
            for slice_id in record.get("slice_ids", [])
        )
        validation = self.extract_semantic_records(
            state,
            {
                "records": generated,
                "slice_decisions": [
                    {"slice_id": slice_id, "status": "record_extracted", "reason": "plan extraction"}
                    for slice_id in processed_slice_ids
                ],
                "target_schema": {
                    **target_schema,
                    "fields": target_fields,
                    "required": required_fields,
                    "record_grain": target_schema.get("record_grain")
                    or plan.get("record_grain")
                    or arguments.get("required_record_grain")
                    or "",
                },
                "field_aliases": {field: sorted(values) for field, values in aliases_by_field.items()},
            },
        )
        payload = validation.get("payload") if isinstance(validation.get("payload"), dict) else {}
        payload["plan"] = plan
        payload["plan_status"] = "validated" if validation.get("ok") else "validation_failed"
        payload["generated_record_count"] = len(generated)
        payload["include_missing_records"] = include_missing_records
        if validation.get("ok"):
            payload["partial_coverage"] = False
            if isinstance(payload.get("coverage_summary"), dict):
                payload["coverage_summary"]["coverage_basis"] = "validated_extraction_plan"
        return {
            **validation,
            "tool_name": "extract_records_by_plan",
            "summary": (
                f"Plan extraction validated {len(payload.get('records') or [])} semantic record(s)."
                if validation.get("ok")
                else "Plan extraction failed validation."
            ),
            "payload": payload,
            "negative_scope": validation.get("negative_scope"),
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
    ) -> dict[str, Any]:
        indexes = self.ensure_indexes(state)
        all_slices = [item for index in indexes.values() for item in index.slices]
        processed = set(processed_slice_ids)
        summary = {
            "document_count": len(indexes),
            "total_slice_count": len(all_slices),
            "processed_slice_count": sum(1 for item in all_slices if item.slice_id in processed),
            "unprocessed_slice_count": sum(1 for item in all_slices if item.slice_id not in processed),
            "source_slice_counts": {
                source_id: index.slice_count for source_id, index in indexes.items()
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
        )

    def _run_model_loop(self, state: LoopState, task: DocTask) -> DocEvidencePackage:
        tool_model = self.model.bind_tools(_DOC_TOOL_SPECS, parallel_tool_calls=False)
        transcript: list[Any] = []
        last_extract: dict[str, Any] | None = None
        trace_events: list[dict[str, Any]] = []
        read_slice_ids: set[str] = set()
        zero_record_repair_used = False
        invalid_extract_repair_used = False
        touched_source_refs: set[str] = {
            str(ref)
            for ref in task.source_candidates
            if str(ref) in state.sources and state.sources[str(ref)].data_form in _DOCUMENT_FORMS
        }
        inspected = self.inspect_document_index(state, {"limit": 12})
        package_context = {
            "doc_task": task.to_dict(),
            "document_index_summary": inspected["payload"],
            "instructions": (
                "You are DocumentAgent. Use only document tools. Do not final-answer, bind, "
                "or compute. Infer field synonyms, record identity, record grain, units, section scope, "
                "and cross-section merges from the question, semantic_cards, and read slice text. For "
                "documents with repeated natural-language records, prefer extract_records_by_plan: you "
                "decide the semantic plan (source, record anchor, section include/exclude hints, fields, "
                "value types, units, and merge key), and the tool performs extraction/provenance checks. "
                "For all-row data listings where the document explicitly includes missing/NaN values, set "
                "include_missing_records=true so the tool preserves anchored records with empty field values. "
                "Use search/read small batches when needed to design or repair the plan. If a plan is "
                "not appropriate, read slices and call extract_semantic_records with records and one "
                "slice_decision for every read slice. Your final internal tool call must be "
                "extract_records_by_plan or extract_semantic_records; if no relevant records are found, "
                "call extract_semantic_records with records=[] and slice_decisions marking every read "
                "slice as no_relevant_record or ambiguous. DocTask "
                "target_fields may include fields that come from other sources later in the main loop; "
                "do not require every target field to appear in one document record. Extract the "
                "evidence-supported document fields that are present and useful for downstream joins, "
                "filters, or metrics, and omit absent fields. Use canonical target field names in "
                "records whenever possible. Do not invent values or rely on filename/table names as evidence."
            ),
        }
        base_messages: list[Any] = [
            SystemMessage(content="You are a bounded DocumentAgent for PDF/MD evidence extraction."),
            HumanMessage(content=json.dumps(package_context, ensure_ascii=False, default=str)),
        ]
        for _step in range(self.max_steps):
            messages = [*base_messages, *transcript[-8:]]
            response = tool_model.invoke(messages)
            calls = extract_tool_calls(response)
            transcript.append(response)
            trace_events.append(
                {
                    "step": _step + 1,
                    "event": "model_response",
                    "tool_calls": [str(call.get("name") or "") for call in calls],
                    "content": _compact_text(getattr(response, "content", ""), limit=240),
                }
            )
            if not calls:
                break
            call = calls[0]
            name = str(call.get("name") or "")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            if name == "extract_semantic_records":
                args = dict(args)
                schema = args.get("target_schema") if isinstance(args.get("target_schema"), dict) else {}
                schema_fields = (
                    list(schema.get("fields") or [])
                    if isinstance(schema.get("fields"), list)
                    else []
                )
                if not schema_fields and isinstance(schema.get("properties"), dict):
                    schema_fields = list(schema["properties"].keys())
                fields = _ordered_unique(
                    [
                        *task.target_fields,
                        *schema_fields,
                        *(args.get("target_fields") or []),
                    ]
                )
                required_fields = _document_required_fields(state, task)
                args["target_fields"] = fields
                args["target_schema"] = {
                    **schema,
                    "fields": fields,
                    "required": _ordered_unique([*(schema.get("required") or []), *required_fields]),
                    "record_grain": schema.get("record_grain") or task.required_record_grain,
                }
                args["field_aliases"] = _field_aliases_from_task(task)
                if read_slice_ids:
                    args["allowed_slice_ids"] = sorted(read_slice_ids)
            elif name == "extract_records_by_plan":
                args = dict(args)
                plan = _dict_value(args.get("plan")) or args
                plan.setdefault("source_candidates", list(task.source_candidates))
                plan.setdefault("target_fields", list(task.target_fields))
                plan.setdefault("record_grain", task.required_record_grain)
                if plan is not args:
                    args["plan"] = plan
                args.setdefault("source_candidates", list(task.source_candidates))
                args.setdefault("target_fields", list(task.target_fields))
                args.setdefault("required_record_grain", task.required_record_grain)
                args["field_aliases"] = _field_aliases_from_task(task)
                args.setdefault(
                    "target_schema",
                    {
                        "fields": list(task.target_fields),
                        "required": _document_required_fields(state, task),
                        "record_grain": task.required_record_grain,
                    },
                )
            result = self._dispatch_internal_tool(state, name, args)
            result_summary = _summarize_doc_tool_result(name, result)
            trace_events.append(
                {
                    "step": _step + 1,
                    "event": "tool_result",
                    "tool": name,
                    "arguments": _summarize_doc_tool_args(name, args),
                    "result": result_summary,
                }
            )
            payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
            if name == "search_document_records":
                for item in payload.get("matches") or []:
                    if not isinstance(item, dict):
                        continue
                    if item.get("source_id"):
                        touched_source_refs.add(str(item["source_id"]))
            elif name == "read_record_slice":
                for slice_id in payload.get("slice_ids") or []:
                    read_slice_ids.add(str(slice_id))
                for record in payload.get("records") or []:
                    if isinstance(record, dict) and record.get("source_id"):
                        touched_source_refs.add(str(record["source_id"]))
            transcript.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=str(call.get("id") or "doc_agent_call"),
                    name=name,
                    status="success" if result.get("ok") else "error",
                )
            )
            if name in {"extract_semantic_records", "extract_records_by_plan"}:
                last_extract = result
                payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
                if (
                    name == "extract_semantic_records"
                    and result.get("ok")
                    and not payload.get("records")
                    and read_slice_ids
                    and not zero_record_repair_used
                    and _step < self.max_steps - 1
                ):
                    zero_record_repair_used = True
                    repair_message = (
                        "extract_semantic_records returned zero records after read_record_slice succeeded. "
                        "Re-check the read slices: DocTask target_fields may include downstream fields from "
                        "other sources. If a read slice contains evidence-supported document-side join keys, "
                        "filter fields, metrics, or dimensions, call extract_semantic_records again with "
                        "partial records using canonical target field names and slice_id/slice_ids provenance. "
                        f"Document-side required fields are: {_document_required_fields(state, task)}. "
                        "Only return zero records again if the read slices truly contain no relevant "
                        "document-side fields."
                    )
                    trace_events.append(
                        {
                            "step": _step + 1,
                            "event": "zero_record_repair_prompt",
                            "content": repair_message,
                        }
                    )
                    transcript.append(HumanMessage(content=repair_message))
                    continue
                if (
                    name == "extract_semantic_records"
                    and not result.get("ok")
                    and read_slice_ids
                    and not invalid_extract_repair_used
                    and _step < self.max_steps - 1
                ):
                    invalid_extract_repair_used = True
                    invalid_items = payload.get("invalid") if isinstance(payload.get("invalid"), list) else []
                    repair_message = (
                        "extract_semantic_records failed validation. Fix the extraction and call "
                        "extract_semantic_records again. Use only these already-read slice ids: "
                        f"{sorted(read_slice_ids)}. Every non-empty extracted value must appear verbatim "
                        "in at least one of its cited slice texts after whitespace normalization; if one logical "
                        "record is split across slices, include all supporting slice ids in slice_ids. Use canonical target field "
                        f"names: {list(task.target_fields)}. Document-side required fields are: "
                        f"{_document_required_fields(state, task)}. If required fields are split across "
                        "slices, read the missing supporting slices before calling extract again. Validation errors: "
                        f"{json.dumps(invalid_items[:8], ensure_ascii=False, default=str)}"
                    )
                    trace_events.append(
                        {
                            "step": _step + 1,
                            "event": "invalid_extract_repair_prompt",
                            "content": _compact_text(repair_message, limit=1_200),
                        }
                    )
                    transcript.append(HumanMessage(content=repair_message))
                    continue
                if (
                    name == "extract_records_by_plan"
                    and not result.get("ok")
                    and not invalid_extract_repair_used
                    and _step < self.max_steps - 1
                ):
                    invalid_extract_repair_used = True
                    invalid_items = payload.get("invalid") if isinstance(payload.get("invalid"), list) else []
                    repair_message = (
                        "extract_records_by_plan did not produce validated records. Repair the semantic "
                        "plan rather than hand-copying records: adjust record_anchor, section include/exclude "
                        "hints, field aliases, value_type/unit, merge_key, or required_fields, then call "
                        "extract_records_by_plan again. Validation/problem details: "
                        f"{json.dumps((invalid_items or payload) if invalid_items else payload, ensure_ascii=False, default=str)[:1600]}"
                    )
                    trace_events.append(
                        {
                            "step": _step + 1,
                            "event": "plan_extract_repair_prompt",
                            "content": _compact_text(repair_message, limit=1_200),
                        }
                    )
                    transcript.append(HumanMessage(content=repair_message))
                    continue
                break
        if last_extract is None:
            processed = tuple(sorted(read_slice_ids))
            return DocEvidencePackage(
                records=(),
                record_schema={"fields": list(task.target_fields), "record_grain": task.required_record_grain},
                source_refs=tuple(sorted(touched_source_refs)),
                evidence_refs=processed,
                processed_slice_ids=processed,
                no_relevant_slice_ids=(),
                ambiguous_slice_ids=processed,
                coverage_summary=self._coverage_summary(state, processed_slice_ids=processed),
                remaining_risks=("document_agent_model_did_not_call_extract_semantic_records",),
                agent_trace=tuple(trace_events),
            )
        if not last_extract.get("ok"):
            payload = last_extract.get("payload", {}) if isinstance(last_extract.get("payload"), dict) else {}
            invalid = payload.get("invalid") if isinstance(payload.get("invalid"), list) else []
            processed = tuple(sorted(read_slice_ids))
            risk = str(
                (last_extract.get("negative_scope") or {}).get("kind")
                or "document_agent_extract_semantic_records_failed_validation"
            )
            return DocEvidencePackage(
                records=tuple(payload.get("records") or []),
                record_schema=payload.get("target_schema") or {"fields": list(task.target_fields)},
                source_refs=tuple(sorted(touched_source_refs)),
                evidence_refs=processed,
                processed_slice_ids=processed,
                no_relevant_slice_ids=(),
                ambiguous_slice_ids=processed,
                coverage_summary=self._coverage_summary(state, processed_slice_ids=processed),
                remaining_risks=(risk, f"invalid_extract_items:{len(invalid)}"),
                agent_trace=tuple(trace_events),
            )
        payload = last_extract.get("payload", {})
        processed = tuple(str(item) for item in payload.get("processed_slice_ids") or [])
        source_refs = tuple(
            sorted(
                {
                    str(record.get("provenance", {}).get("source_id"))
                    for record in payload.get("records") or []
                    if isinstance(record, dict)
                }
            )
        )
        if not source_refs:
            source_refs = tuple(sorted(touched_source_refs))
        return DocEvidencePackage(
            records=tuple(payload.get("records") or []),
            record_schema=payload.get("record_schema") or {"fields": list(task.target_fields)},
            source_refs=source_refs,
            evidence_refs=processed,
            processed_slice_ids=processed,
            no_relevant_slice_ids=tuple(payload.get("no_relevant_slice_ids") or []),
            ambiguous_slice_ids=tuple(payload.get("ambiguous_slice_ids") or []),
            coverage_summary=payload.get("coverage_summary") or self._coverage_summary(state, processed_slice_ids=processed),
            remaining_risks=(
                ("partial_document_coverage",)
                if payload.get("partial_coverage")
                else ()
            ),
            agent_trace=tuple(trace_events),
        )

    def _dispatch_internal_tool(
        self,
        state: LoopState,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if name == "inspect_document_index":
            return self.inspect_document_index(state, arguments)
        if name == "search_document_records":
            return self.search_document_records(state, arguments)
        if name == "read_record_slice":
            return self.read_record_slice(state, arguments)
        if name == "extract_semantic_records":
            return self.extract_semantic_records(state, arguments)
        if name == "extract_records_by_plan":
            return self.extract_records_by_plan(state, arguments)
        if name == "check_document_coverage":
            return self.check_document_coverage(state, arguments)
        return {
            "ok": False,
            "tool_name": name,
            "summary": f"Unknown DocumentAgent tool: {name}",
            "payload": {"arguments": arguments},
        }


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


_DOC_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_document_index",
            "description": "Inspect bootstrap-built document record indexes without returning full document text.",
            "parameters": _object_schema(
                {
                    "source_ref": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_document_records",
            "description": "Search record slices by query and semantic target fields.",
            "parameters": _object_schema(
                {
                    "query": {"type": "string"},
                    "semantic_fields": {"type": "array", "items": {"type": "string"}},
                    "source_ref": {"type": "string"},
                    "source_candidates": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_record_slice",
            "description": "Read complete record slices by slice_id.",
            "parameters": _object_schema(
                {
                    "slice_ids": {"type": "array", "items": {"type": "string"}},
                    "slice_id": {"type": "string"},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_semantic_records",
            "description": (
                "Submit semantic records extracted from read slices with per-slice decisions. "
                "Every record should include slice_id or provenance.slice_id. Every slice_decision "
                "must use status: record_extracted, no_relevant_record, or ambiguous."
            ),
            "parameters": _object_schema(
                {
                    "records": {
                        "type": "array",
                        "items": _object_schema(
                            {
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
                            },
                            required=["slice_id", "status"],
                            additional_properties=True,
                        ),
                    },
                    "target_schema": {"type": "object", "additionalProperties": True},
                    "target_fields": {"type": "array", "items": {"type": "string"}},
                },
                required=["slice_decisions"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_records_by_plan",
            "description": (
                "Execute an LLM-authored semantic extraction plan over a PDF/MD source. Use this for "
                "repeated natural-language records, cross-section field assembly, and section-scoped "
                "metrics. The LLM decides the source, record anchor, include/exclude section hints, "
                "field aliases/value types/units, and merge key; the tool extracts records and validates "
                "provenance."
            ),
            "parameters": _object_schema(
                {
                    "source_ref": {"type": "string"},
                    "source_candidates": {"type": "array", "items": {"type": "string"}},
                    "target_fields": {"type": "array", "items": {"type": "string"}},
                    "record_grain": {"type": "string"},
                    "merge_key": {"type": "string"},
                    "include_missing_records": {"type": "boolean"},
                    "missing_value_policy": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                    "entity_anchor": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "labels": {"type": "array", "items": {"type": "string"}},
                            "pattern": {"type": "string"},
                        },
                        "additionalProperties": True,
                    },
                    "section_scope": {
                        "type": "object",
                        "properties": {
                            "include_hints": {"type": "array", "items": {"type": "string"}},
                            "exclude_hints": {"type": "array", "items": {"type": "string"}},
                            "start_after_hints": {"type": "array", "items": {"type": "string"}},
                        },
                        "additionalProperties": True,
                    },
                    "fields": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "value_type": {"type": "string"},
                                "unit": {"type": "string"},
                                "aliases": {"type": "array", "items": {"type": "string"}},
                                "missing_indicators": {"type": "array", "items": {"type": "string"}},
                                "section_scope": {"type": "object", "additionalProperties": True},
                            },
                            "additionalProperties": True,
                        },
                    },
                    "required_fields": {"type": "array", "items": {"type": "string"}},
                    "plan": {"type": "object", "additionalProperties": True},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_document_coverage",
            "description": "Check processed slice coverage for the current document task.",
            "parameters": _object_schema(
                {"processed_slice_ids": {"type": "array", "items": {"type": "string"}}}
            ),
        },
    },
]
