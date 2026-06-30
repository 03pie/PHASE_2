from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.evidence_agent.codex_loop.protocol import LoopState
from data_agent_baseline.evidence_agent.codex_loop.state_views import (
    answer_contract_view,
    answer_candidates,
    exhaustion_status,
    final_output_contract,
    primary_next_action,
    selected_source_candidates,
    semantic_selection_view,
    source_coverage_map,
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
- at the start of every turn, read next_action_guidance first and choose the next native tool from the current ledger state.
- before choosing knowledge cards or physical fields/sources, use declare_answer_contract to record the LLM's question understanding contract when none is present; include intent_summary, answer_grain, final_outputs, constraints, operations, helper_fields, field_roles, null_policy, transform_intent, document_policy, and unresolved_terms.
- after the answer contract, use select_semantic_cards to choose the knowledge semantic cards for this task; only selected cards are expanded into source candidates.
- observe the real environment through tools such as inspect_source, sample_records, search_values, and run_document_agent.
- call bind only after successful evidence proves a usable source/field/value/record set; include semantic_mappings when binding evidence for knowledge-defined fields.
- call run_verified_compute only over verified relation names from bindings.
- call submit_final with compute_ref only for an existing successful compute result and with an explicit answer.columns projection.
- submit_final may project or alias columns and explicit row_indices already present in the compute result, but it must not add new values.
- call submit_final(answer=..., binding_refs=..., evidence_refs=...) only for direct answers backed by verified value/document/operation bindings and evidence_refs.
- call blocked when evidence is insufficient, conflicting, or no valid action remains; cite evidence_refs when possible.

Rules:
- Do not assume a business domain. Use only the user question, knowledge document text, and observations.
- knowledge.md is an authority for semantics, not a physical schema or data format hint.
- Candidate sources, filenames, document search hits, and semantic similarities are not bindings.
- Physical field names do not need to equal knowledge canonical names. After inspecting samples or extracting document records, the LLM may declare the semantic mapping in bind(... semantic_mappings=..., alignment=...).
- PDF/MD/video are not structured tables. Documents must go through run_document_agent, which returns a compact DocEvidencePackage with validated records and coverage.
- Use selected semantic source mappings only as grounding candidates: unverified_structured_candidate, then unverified_document_candidate, then lexical inventory search when the LLM asks for it. Candidates are not bindings or proof.
- Video is unsupported in v1. It can be inspected for metadata but cannot support final evidence.
- Every physical field/table/path used for compute must come from observed evidence and bindings, and required knowledge-defined canonical fields must be covered by LLM-declared binding semantic mappings.
- A requested answer may require multiple verified sources. If different requested fields are observed in different relations, bind each relation and join or align them using observed shared keys instead of requiring one source to contain every field.
- Use discover_join_paths when multiple verified relations may need joining and the shared key is uncertain.
- Any transformed value, row reduction, aggregation, ordering, join, or direct extraction must be justified by observed evidence, knowledge text, and the LLM-declared question contract.
- Final answers should contain only requested answer columns. Drop helper join/filter columns unless the user asked for them as answer dimensions.
- For list/show-data tasks, decide row coverage before compute. Preserve null/empty source rows unless the question asks to filter/rank non-empty values or a metric requires non-null inputs.
- Do not turn unobserved tokens into physical fields, filters, tables, files, or values.
- Avoid repeated failed actions: if guard_feedback shows a failed action, choose a materially different ledger-backed action from next_action_guidance instead of repeating the same call.
"""


TOOL_GUIDE = {
    "tool_protocol": "Use native tool calls. Text-only answers are not accepted.",
    "answer_contract_protocol": "If no answer_contract is present, call declare_answer_contract first. It is the question understanding contract: intent_summary, answer_grain, final_outputs, constraints, operations, helper_fields, field_roles, null_policy, transform_intent, document_policy, and unresolved_terms. final_outputs must be final answer columns only; helper filter/sort/join/row-selection fields belong in helper_fields and field_roles. This records semantic intent only; it is not a physical source, schema, or binding.",
    "semantic_selection_protocol": "After answer_contract, call select_semantic_cards(card_ids, rationale, unmapped_intents). Only selected cards are expanded into source candidates. If no card maps an intent, declare it in unmapped_intents and use locate_sources/search_values.",
    "final_protocol": "For compute-backed final answers, call submit_final(compute_ref=..., answer={columns:[...], row_indices:[...]}) with explicit requested final columns and optional explicit compute row projection. For direct document/value evidence, provide answer plus binding_refs and evidence_refs.",
    "binding_protocol": "Use bind(...) with evidence_refs before compute. For knowledge-defined fields, bind with semantic_mappings grounded in observed columns/records; physical names may differ from canonical names when the LLM declares the semantic mapping. Never rely on knowledge formatting alone.",
    "relation_protocol": "Use generated relation_name values such as rel_0001 in SQL.",
    "multi_source_protocol": "Requested fields may be assembled from multiple verified relations when observed shared keys support a join/alignment.",
    "join_discovery_protocol": "Use discover_join_paths over verified relations to observe generic same-column or sample-overlap join candidates before uncertain joins.",
    "evidence_protocol": "Use observed evidence, knowledge text, and the LLM-declared question contract to justify transformations, filters, joins, aggregation, ordering, and direct extraction.",
    "knowledge_protocol": "Start from semantic_knowledge cards for canonical meaning, aliases, units, record_grain, join_keys, formulas, and ambiguity rules. Knowledge is not a physical schema; selected_source_candidates only appears after select_semantic_cards.",
    "source_resolution_protocol": "Use selected_source_candidates and locate_sources to collect physical grounding candidates. Candidate priority is unverified_structured_candidate, then unverified_document_candidate. Lexical search is only for LLM-declared unmapped intents; no candidate is directly bindable until inspect_source/sample_records or run_document_agent produces real evidence.",
    "failure_protocol": "When a tool returns negative_scope or a repeated failure, switch tools or call blocked.",
    "blocked_protocol": "When calling blocked, cite evidence_refs that support the absence/conflict/exhaustion claim whenever possible.",
    "document_protocol": "For unverified_document_candidate mappings or PDF/MD evidence, call run_document_agent with question, target_fields, semantic_cards, source_candidates, required_record_grain, and coverage_policy. The DocumentAgent scans document slices, lets the LLM record relevant decisions, then returns only a compact DocEvidencePackage to the main loop.",
    "sql_protocol": "Use inspect_relation after bind and before changing failed SQL. In run_verified_compute SQL, use parse_date_key(value) to sort/filter common date strings, including Chinese numeral dates, as YYYYMMDD integers.",
}


def _preview(text: str, *, limit: int = 120) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _semantic_field_id(card: Any) -> str | None:
    table = str(getattr(card, "semantic_scope", getattr(card, "canonical_table", "")) or "").strip()
    field = str(getattr(card, "semantic_slot", getattr(card, "canonical_field", "")) or "").strip()
    if not table or not field:
        return None
    return f"{table}.{field}".casefold()


def _compact_mapping(mapping: Any, *, preferred: bool) -> dict[str, Any]:
    status = str(getattr(mapping, "status", "") or "")
    priority = "fallback_only"
    if preferred and status == "unverified_structured_candidate":
        priority = "preferred_unverified_structured_candidate"
    elif preferred and status == "unverified_document_candidate":
        priority = "document_when_needed"
    return {
        "source_id": getattr(mapping, "source_id", None),
        "source_path": getattr(mapping, "source_path", None),
        "data_form": getattr(mapping, "data_form", None),
        "status": status,
        "semantic_scope": getattr(mapping, "semantic_scope", None),
        "semantic_slot": getattr(mapping, "semantic_slot", None),
        "physical_table": getattr(mapping, "physical_table", None),
        "physical_field": getattr(mapping, "physical_field", None),
        "warnings": list(getattr(mapping, "warnings", ()) or ()),
        "binding_priority": priority,
        "recommended_tool": (
            "run_document_agent"
            if status == "unverified_document_candidate"
            else "inspect_source"
            if status == "unverified_structured_candidate"
            else "retrieve_knowledge"
        ),
    }


def _semantic_source_plan(state: LoopState) -> dict[str, Any]:
    mappings_by_card: dict[str, list[Any]] = {}
    for mapping in state.source_mappings:
        mappings_by_card.setdefault(mapping.card_id, []).append(mapping)

    candidate_fields: list[dict[str, Any]] = []
    seen_fields: set[str] = set()
    selected_ids = (
        set(state.semantic_selection.card_ids)
        if state.semantic_selection is not None
        else set()
    )
    cards_for_plan = [card for card in state.semantic_cards if card.id in selected_ids]
    for card in cards_for_plan:
        field_id = _semantic_field_id(card)
        if not field_id or field_id in seen_fields:
            continue
        seen_fields.add(field_id)
        mappings = mappings_by_card.get(card.id, [])
        observed_structured = [
            _compact_mapping(mapping, preferred=True)
            for mapping in mappings
            if mapping.status == "unverified_structured_candidate"
        ][:4]
        documents = [
            _compact_mapping(mapping, preferred=True)
            for mapping in mappings
            if mapping.status == "unverified_document_candidate"
        ][:4]
        observed_lexical = [
            _compact_mapping(mapping, preferred=False)
            for mapping in mappings
            if mapping.status == "unverified_lexical_candidate"
        ][:4]
        missing = [
            _compact_mapping(mapping, preferred=False)
            for mapping in mappings
            if mapping.status == "unresolved_grounding"
        ][:2]
        if not (observed_structured or documents or observed_lexical or missing):
            continue
        candidate_fields.append(
            {
                "card_id": card.id,
                "semantic_slot_id": field_id,
                "semantic_scope": card.semantic_scope,
                "semantic_slot": card.semantic_slot,
                "unit": card.unit,
                "record_grain": card.record_grain,
                "grounding_candidates": [*observed_structured, *documents],
                "has_document_candidates": bool(documents),
                "has_structured_candidates": bool(observed_structured),
                "unverified_structured_candidates": observed_structured,
                "document_candidates": documents,
                "unverified_lexical_candidates": observed_lexical,
                "missing_mappings": missing,
                "binding_instruction": (
                    "Use these only as grounding candidates. First inspect/sample structured candidates "
                    "or run DocumentAgent for document candidates. Bind only real observed columns/records with "
                    "semantic mappings; lexical candidates can be used after the LLM declares semantic/grain "
                    "alignment from observed evidence."
                ),
            }
        )

    selection_required = bool(state.semantic_cards) and state.semantic_selection is None
    return {
        "instruction": (
            "These candidate fields come only from LLM-selected semantic cards. If this list is empty, "
            "call select_semantic_cards first or use locate_sources for unmapped intents. Source-resolution "
            "entries are grounding candidates, not physical proof."
        ),
        "candidate_fields": candidate_fields[:16],
        "has_candidate_fields": bool(candidate_fields),
        "selection_required": selection_required,
        "used_catalog_fallback": False,
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
            "Use these semantic cards for meaning only: semantic slots, aliases, units, record grain, "
            "join keys, formulas, and ambiguity rules. Do not infer physical data format from this fragment."
        ),
        "card_count": len(state.semantic_cards),
        "matched_cards": [
            {
                "id": card.id,
                "kind": card.kind,
                "name": card.name,
                "semantic_scope": card.semantic_scope,
                "semantic_slot": card.semantic_slot,
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
                "semantic_scope": card.semantic_scope,
                "semantic_slot": card.semantic_slot,
                "unit": card.unit,
                "record_grain": card.record_grain,
                "join_keys": list(card.join_keys),
            }
            for card in state.semantic_cards[:30]
        ],
    }
    matched_card_ids = (
        set(state.semantic_selection.card_ids)
        if state.semantic_selection is not None
        else set()
    )
    source_resolution = {
        "instruction": (
            "Grounding candidates derived by comparing semantic cards with the observed inventory. "
            "This is source-resolution context, not knowledge, and not proof. Use it to choose tools "
            "and then observe/bind real sources through evidence."
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
    compute = [result.to_dict() for result in state.compute_results.values()]
    ok_compute = [result.to_dict() for result in state.compute_results.values() if result.ok]
    candidates_for_answer = answer_candidates(state)
    source_coverage = source_coverage_map(state)
    answer_contract = answer_contract_view(state)
    semantic_selection = semantic_selection_view(state)
    source_candidates = selected_source_candidates(state)
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
    next_action_guidance = {
        "instruction": (
            "Read this before choosing the next native tool. It is generated before the model action "
            "from the current ledger, so use it as the primary next-step guide instead of relying on "
            "post-action correction."
        ),
        "primary_next_action": primary_next_action(state),
        "answer_contract": answer_contract,
        "semantic_selection": semantic_selection,
        "selected_source_candidates": source_candidates,
        "final_submission_rules": {
            "compute_final": (
                "For a compute-backed final, submit_final with compute_ref and answer.columns containing only requested output columns. "
                "If the LLM has judged that only specific compute rows answer the question, include zero-based "
                "answer.row_indices; submit_final will only validate/project those rows mechanically."
            ),
            "direct_final": (
                "For direct document/value evidence, submit_final must include answer, binding_refs, and evidence_refs "
                "that come from successful observed evidence."
            ),
            "column_projection": (
                "Never submit every compute column by default. Drop helper/filter/sort columns unless the user asked "
                "for them as answer columns."
            ),
        },
        "compute_helpers": {
            "parse_date_key(value)": (
                "Use in run_verified_compute SQL when the selected semantic field is a date string that needs "
                "ordering or filtering; returns YYYYMMDD for common Arabic and Chinese numeral date forms."
            ),
        },
        "recent_failed_actions": state.guard_feedback[-5:],
    }

    fragments = [
        ContextFragment("ctx_question", "question", state.question),
        _json_fragment("tool_guide", TOOL_GUIDE, 4_000),
        _json_fragment("next_action_guidance", next_action_guidance, 6_000),
        _json_fragment("answer_contract", answer_contract, 4_000),
        _json_fragment("semantic_selection", semantic_selection, 6_000),
        _json_fragment("inventory", inventory, 8_000),
        _json_fragment("document_agent", document_agent, 6_000),
        _json_fragment("semantic_knowledge", semantic_knowledge, 12_000),
        _json_fragment("source_resolution", source_resolution, 12_000),
        _json_fragment("selected_source_candidates", source_candidates, 8_000),
        _json_fragment("knowledge_catalog", knowledge_catalog, 8_000),
        _json_fragment("matched_knowledge_sections", matched_knowledge, 6_000),
        _json_fragment("candidates", candidates, 6_000),
        _json_fragment("latest_evidence", evidence, 12_000),
        _json_fragment("source_coverage", source_coverage, 8_000),
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
