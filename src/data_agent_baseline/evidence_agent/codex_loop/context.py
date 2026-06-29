from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.evidence_agent.codex_loop.protocol import LoopState
from data_agent_baseline.evidence_agent.codex_loop.state_views import (
    answer_candidates,
    exhaustion_status,
    final_output_contract,
    requirement_coverage,
    source_coverage_map,
    verifier_decisions,
)


@dataclass(frozen=True, slots=True)
class ContextFragment:
    id: str
    kind: str
    text: str
    truncated: bool = False


def _clip(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[: max(0, limit - 120)] + "\n...[truncated]", True


def _json_fragment(kind: str, value: Any, limit: int) -> ContextFragment:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    clipped, truncated = _clip(text, limit)
    return ContextFragment(id=f"ctx_{kind}", kind=kind, text=clipped, truncated=truncated)


SYSTEM_PROMPT = """You are a Codex-style evidence agent for data tasks.

Use native tool calls only. Do not put tool requests in assistant text.

Allowed flow:
- observe the real environment through tools such as inspect_source, sample_records, search_values, and run_document_agent.
- call bind only after successful evidence proves a usable source/field/value/record set; include canonical_fields / physical_field_mapping when binding evidence for knowledge-defined fields.
- call run_verified_compute only over verified relation names from bindings.
- after a successful compute, call verify_alignment(decision="candidate_answer", target_kind="compute_result") before relying on it as final.
- call submit_final with compute_ref only after candidate_answer verification and with an explicit answer.columns projection.
- submit_final may project or alias values already present in the compute result, but it must not add new values.
- call submit_final(answer=..., binding_refs=..., evidence_refs=...) only for direct answers backed by verified value/document/operation bindings and evidence_refs.
- call blocked when evidence is insufficient, conflicting, or no valid action remains; cite evidence_refs when possible.
- use track_requirements to maintain a generic coverage checklist when the task has multiple required sources, filters, joins, metrics, units, or output constraints.
- use verify_alignment to classify semantic alternatives, document evidence, compute results, final candidates, and blocked/conflict decisions before relying on them.

Rules:
- Do not assume a business domain. Use only the user question, knowledge document text, and observations.
- knowledge.md is an authority for semantics, not a physical schema or data format hint.
- Candidate sources, filenames, document search hits, and semantic similarities are not bindings.
- PDF/MD/video are not structured tables. Documents must go through run_document_agent, which returns a compact DocEvidencePackage with validated records and coverage.
- Use semantic source mapping status as the source priority: exact_structured_source, then document_source, then fallback_candidate. A fallback_candidate is only a discovery hint, even if it is structured.
- Video is unsupported in v1. It can be inspected for metadata but cannot support final evidence.
- Every physical field/table/path used for compute must come from observed evidence and verified bindings, and required knowledge-defined canonical fields must be covered by binding semantic_contract metadata.
- Verifier decisions and requirement tracking are audit evidence, not physical data. They cannot replace real observations.
- A requested answer may require multiple verified sources. If different requested fields are observed in different relations, bind each relation and join or align them using observed shared keys instead of requiring one source to contain every field.
- Use discover_join_paths when multiple verified relations may need joining and the shared key is uncertain.
- Any transformed value, row reduction, aggregation, ordering, join, or direct extraction must be justified by observed evidence, knowledge text, or a verifier decision.
- Final answers should contain only requested answer columns. Drop helper join/filter columns unless the user asked for them as answer dimensions.
- For list/show-data tasks, decide row coverage before compute. Preserve null/empty source rows unless the question asks to filter/rank non-empty values or a metric requires non-null inputs.
- Do not turn unobserved tokens into physical fields, filters, tables, files, or values.
"""


TOOL_GUIDE = {
    "tool_protocol": "Use native tool calls. Text-only answers are not accepted.",
    "final_protocol": "For compute-backed final answers, first verify_alignment(candidate_answer, target_kind=compute_result), then submit_final(compute_ref=..., answer={columns:[...]}) with explicit requested final columns only. For direct document/value evidence, provide answer plus binding_refs and evidence_refs.",
    "binding_protocol": "Use bind(...) with evidence_refs before compute. For knowledge-defined fields, bind with semantic_card_ids, canonical_fields, and physical_field_mapping, or rely on an exact source mapping that can infer the semantic_contract.",
    "requirement_protocol": "Use track_requirements(...) to declare/update generic required answer conditions when the answer depends on multiple constraints.",
    "verifier_protocol": "Use verify_alignment(...) to mark observed evidence/compute as bindable, candidate_answer, intermediate, not_applicable, needs_more_evidence, conflict, or blocked_ok.",
    "relation_protocol": "Use generated relation_name values such as rel_0001 in SQL.",
    "multi_source_protocol": "Requested fields may be assembled from multiple verified relations when observed shared keys support a join/alignment.",
    "join_discovery_protocol": "Use discover_join_paths over verified relations to observe generic same-column or sample-overlap join candidates before uncertain joins.",
    "evidence_protocol": "Use observed evidence, knowledge text, or verifier decisions to justify transformations, filters, joins, aggregation, ordering, and direct extraction.",
    "knowledge_protocol": "Start from semantic_knowledge cards for canonical meaning, aliases, units, record_grain, join_keys, formulas, and ambiguity rules. Use source_resolution or locate_sources for physical source candidates and data formats. Use retrieve_knowledge(mode='semantic') to refresh cards, and retrieve_knowledge(mode='section') only when raw knowledge wording is needed for audit.",
    "source_resolution_protocol": "Use source_resolution and locate_sources to choose physical sources. Source priority is exact_structured_source, then document_source, then fallback_candidate; fallback candidates are not directly bindable canonical fields without extra observed semantic/grain proof.",
    "failure_protocol": "When a tool returns negative_scope or a repeated failure, switch tools or call blocked.",
    "blocked_protocol": "When calling blocked, cite evidence_refs that support the absence/conflict/exhaustion claim whenever possible.",
    "document_protocol": "For document_source mappings or PDF/MD evidence, call run_document_agent with question, target_fields, semantic_cards, source_candidates, required_record_grain, and coverage_policy. The DocumentAgent owns record-slice search/read/extract/coverage internally and returns only a compact DocEvidencePackage to the main loop.",
    "sql_protocol": "Use inspect_relation after bind and before repairing failed SQL.",
}


def _preview(text: str, *, limit: int = 120) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _semantic_field_id(card: Any) -> str | None:
    table = str(getattr(card, "canonical_table", "") or "").strip()
    field = str(getattr(card, "canonical_field", "") or "").strip()
    if not table or not field:
        return None
    return f"{table}.{field}".casefold()


def _compact_mapping(mapping: Any, *, preferred: bool) -> dict[str, Any]:
    status = str(getattr(mapping, "status", "") or "")
    priority = "fallback_only"
    if preferred and status == "exact_structured_source":
        priority = "preferred_exact_structured"
    elif preferred and status == "document_source":
        priority = "document_when_needed"
    return {
        "source_id": getattr(mapping, "source_id", None),
        "source_path": getattr(mapping, "source_path", None),
        "data_form": getattr(mapping, "data_form", None),
        "status": status,
        "matched_table": getattr(mapping, "matched_table", None),
        "matched_field": getattr(mapping, "matched_field", None),
        "warnings": list(getattr(mapping, "warnings", ()) or ()),
        "binding_priority": priority,
        "recommended_tool": (
            "run_document_agent"
            if status == "document_source"
            else "inspect_source"
            if status == "exact_structured_source"
            else "retrieve_knowledge"
        ),
    }


def _semantic_source_plan(state: LoopState) -> dict[str, Any]:
    mappings_by_card: dict[str, list[Any]] = {}
    for mapping in state.source_mappings:
        mappings_by_card.setdefault(mapping.card_id, []).append(mapping)

    required_fields: list[dict[str, Any]] = []
    seen_fields: set[str] = set()
    for card in state.matched_semantic_cards:
        field_id = _semantic_field_id(card)
        if not field_id or field_id in seen_fields:
            continue
        seen_fields.add(field_id)
        mappings = mappings_by_card.get(card.id, [])
        exact_structured = [
            _compact_mapping(mapping, preferred=True)
            for mapping in mappings
            if mapping.status == "exact_structured_source"
        ]
        documents = [
            _compact_mapping(mapping, preferred=True)
            for mapping in mappings
            if mapping.status == "document_source"
        ]
        fallback = [
            _compact_mapping(mapping, preferred=False)
            for mapping in mappings
            if mapping.status == "fallback_candidate"
        ]
        missing = [
            _compact_mapping(mapping, preferred=False)
            for mapping in mappings
            if mapping.status == "unsupported_or_missing"
        ]
        if not (exact_structured or documents or fallback or missing):
            continue
        required_fields.append(
            {
                "card_id": card.id,
                "canonical_field": field_id,
                "canonical_table": card.canonical_table,
                "field": card.canonical_field,
                "unit": card.unit,
                "record_grain": card.record_grain,
                "preferred_mappings": [*exact_structured, *documents],
                "exact_structured_mappings": exact_structured,
                "document_mappings": documents,
                "fallback_candidates": fallback,
                "missing_mappings": missing,
                "binding_instruction": (
                    "Use exact_structured_mappings first, then document_mappings. A fallback_candidate is only "
                    "a discovery hint, even if structured; do not bind it as this canonical field while a direct "
                    "exact/document mapping can answer. Use fallback only after direct mappings are proven "
                    "unusable and observed evidence proves the same semantic field and record grain."
                ),
            }
        )

    return {
        "instruction": (
            "Resolve these canonical fields before choosing physical sources. Source priority is "
            "exact_structured_mappings, then document_mappings, then fallback_candidates. Fallback candidates "
            "are not direct bindings; similarly named tables or columns need extra semantic/grain proof."
        ),
        "required_fields": required_fields[:8],
        "has_required_fields": bool(required_fields),
    }


def build_context_fragments(
    state: LoopState,
    *,
    last_error: str | None = None,
    recovery_hint: dict[str, Any] | None = None,
) -> list[ContextFragment]:
    inventory = [
        {
            "id": source.id,
            "path": source.virtual_path,
            "data_form": source.data_form,
            "tables": list(source.tables[:20]),
            "columns": list(source.columns[:30]),
            "metadata": source.metadata,
        }
        for source in state.sources.values()
    ]
    document_agent = {
        "instruction": (
            "PDF/MD work is isolated behind run_document_agent. The main loop should pass a DocTask "
            "and consume only the returned DocEvidencePackage; do not request line/window document tools."
        ),
        "indexed_document_count": len(state.document_record_indexes),
        "total_record_slice_count": sum(
            int(getattr(index, "slice_count", 0))
            for index in state.document_record_indexes.values()
        ),
        "indexes": [
            {
                "source_id": getattr(index, "source_id", source_id),
                "path": getattr(index, "path", ""),
                "data_form": getattr(index, "data_form", ""),
                "slice_count": getattr(index, "slice_count", 0),
                "page_count": getattr(index, "page_count", None),
            }
            for source_id, index in state.document_record_indexes.items()
        ],
        "latest_packages": state.document_agent_packages[-3:],
        "latest_coverage": state.document_coverage,
    }
    semantic_knowledge = {
        "instruction": (
            "Use these semantic cards for meaning only: canonical fields, aliases, units, record grain, "
            "join keys, formulas, and ambiguity rules. Do not infer physical data format from this fragment."
        ),
        "card_count": len(state.semantic_cards),
        "matched_cards": [
            {
                "id": card.id,
                "kind": card.kind,
                "name": card.name,
                "canonical_table": card.canonical_table,
                "canonical_field": card.canonical_field,
                "definition_preview": _preview(card.definition, limit=300),
                "aliases": list(card.aliases[:8]),
                "unit": card.unit,
                "record_grain": card.record_grain,
                "join_keys": list(card.join_keys),
                "formula": card.formula,
                "section_id": card.section_id,
                "heading_path": card.heading_path,
            }
            for card in state.matched_semantic_cards[:8]
        ],
        "card_catalog": [
            {
                "id": card.id,
                "kind": card.kind,
                "name": card.name,
                "canonical_table": card.canonical_table,
                "canonical_field": card.canonical_field,
                "unit": card.unit,
                "record_grain": card.record_grain,
                "join_keys": list(card.join_keys),
            }
            for card in state.semantic_cards[:30]
        ],
    }
    matched_card_ids = {card.id for card in state.matched_semantic_cards}
    source_resolution = {
        "instruction": (
            "Physical source candidates derived by comparing semantic cards with the observed inventory. "
            "This is source-resolution context, not knowledge. Use it to choose tools and then verify/bind "
            "real sources through evidence."
        ),
        "source_plan": _semantic_source_plan(state),
        "matched_mappings": [
            mapping.to_dict()
            for mapping in state.source_mappings
            if mapping.card_id in matched_card_ids
        ][:80],
        "matched_source_mapping_count": sum(
            1 for mapping in state.source_mappings if mapping.card_id in matched_card_ids
        ),
    }
    knowledge_catalog = {
        "instruction": (
            "Raw /context/knowledge.md section catalog for audit only. Prefer semantic_knowledge "
            "for definitions; use source_resolution or locate_sources for source choices. Call retrieve_knowledge with mode='section' or mode='token' "
            "when exact wording is needed."
        ),
        "section_count": len(state.knowledge_sections),
        "lookup_count": len(state.knowledge_lookup),
        "sections": [
            {
                "id": section.id,
                "heading_path": section.heading_path,
                "line_start": section.line_start,
                "line_end": section.line_end,
                "mention_count": len(section.mentions),
                "mentions": list(section.mentions[:6]),
                "preview": _preview(section.text),
            }
            for section in state.knowledge_sections
        ],
        "lookup_access": "Call retrieve_knowledge(mode='catalog') for lookup tokens, or mode='token' with candidate mentions.",
    }
    matched_knowledge = [
        {
            "id": section.id,
            "heading_path": section.heading_path,
            "line_start": section.line_start,
            "line_end": section.line_end,
            "text": section.text,
            "mentions": list(section.mentions),
        }
        for section in state.matched_sections[:6]
    ]
    candidates = [candidate.to_dict() for candidate in list(state.candidates.values())[-30:]]
    evidence = [item.to_dict() for item in list(state.evidence.values())[-20:]]
    bindings = [binding.to_dict() for binding in state.bindings.values()]
    requirements = requirement_coverage(state)
    verifier = verifier_decisions(state)[-20:]
    compute = [result.to_dict() for result in state.compute_results.values()]
    ok_compute = [result.to_dict() for result in state.compute_results.values() if result.ok]
    candidates_for_answer = answer_candidates(state)
    source_coverage = source_coverage_map(state)
    output_contract = final_output_contract(state)
    direct_bindings = [
        binding.to_dict()
        for binding in state.bindings.values()
        if binding.binding_type in {"document_window", "value", "operation", "answer_candidate"}
    ]
    decision_pressure: dict[str, Any] = {}
    if ok_compute:
        decision_pressure["successful_compute_available"] = {
            "instruction": (
                "If a compute result answers the question, call submit_final with its compute_ref. "
                "If it does not answer the question, run a materially different compute or gather specific missing evidence."
            ),
            "compute_refs": [result["id"] for result in ok_compute[-5:]],
        }
    if direct_bindings:
        decision_pressure["direct_evidence_bindings_available"] = {
            "instruction": (
                "If direct evidence fully supports the answer, call submit_final with an explicit answer table, "
                "binding_refs, evidence_refs, and alignment. Otherwise continue evidence collection or blocked."
            ),
            "binding_refs": [binding["id"] for binding in direct_bindings[-8:]],
        }
    if candidates_for_answer:
        decision_pressure["answer_candidates"] = {
            "instruction": (
                "These are not automatically correct answers. Evaluate them against the question and knowledge. "
                "If one is sufficient, submit_final. If none is sufficient, gather specific missing evidence or blocked."
            ),
            "candidates": candidates_for_answer[-8:],
        }

    fragments = [
        ContextFragment("ctx_question", "question", state.question),
        _json_fragment("tool_guide", TOOL_GUIDE, 4_000),
        _json_fragment("inventory", inventory, 8_000),
        _json_fragment("document_agent", document_agent, 6_000),
        _json_fragment("semantic_knowledge", semantic_knowledge, 12_000),
        _json_fragment("source_resolution", source_resolution, 12_000),
        _json_fragment("knowledge_catalog", knowledge_catalog, 8_000),
        _json_fragment("matched_knowledge_sections", matched_knowledge, 6_000),
        _json_fragment("candidates", candidates, 6_000),
        _json_fragment("latest_evidence", evidence, 12_000),
        _json_fragment("source_coverage", source_coverage, 8_000),
        _json_fragment("requirements", requirements, 8_000),
        _json_fragment("verifier_decisions", verifier, 8_000),
        _json_fragment("bindings", bindings, 8_000),
        _json_fragment("compute_results", compute, 8_000),
        _json_fragment("final_output_contract", output_contract, 4_000),
    ]
    if recovery_hint:
        fragments.append(
            _json_fragment(
                "recovery_hint",
                {
                    "instruction": (
                        "Previous turn failed or made no effective progress. Use this single "
                        "ledger-backed recovery direction, or call blocked with cited evidence."
                    ),
                    "hint": recovery_hint,
                },
                4_000,
            )
        )
    if decision_pressure:
        fragments.append(_json_fragment("decision_pressure", decision_pressure, 4_000))
    if state.evidence or state.negative_scopes:
        fragments.append(_json_fragment("exhaustion_status", exhaustion_status(state), 5_000))
    if state.guard_feedback or last_error:
        fragments.append(
            _json_fragment(
                "guard_feedback",
                {"last_error": last_error, "feedback": state.guard_feedback[-8:]},
                4_000,
            )
        )
    return fragments


def render_user_context(fragments: list[ContextFragment]) -> str:
    parts: list[str] = []
    for fragment in fragments:
        marker = " truncated=true" if fragment.truncated else ""
        parts.append(f"<fragment id='{fragment.id}' kind='{fragment.kind}'{marker}>\n{fragment.text}\n</fragment>")
    return "\n\n".join(parts)
