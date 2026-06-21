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
- observe the real environment through tools such as inspect_source, sample_records, search_values, read_document_window.
- call bind only after successful evidence proves a usable source/field/value/record set.
- call run_verified_compute only over verified relation names from bindings.
- call submit_final with either a successful compute_ref or a direct answer backed by verified value/document/operation bindings and evidence_refs.
- call blocked when evidence is insufficient, conflicting, or no valid action remains; cite evidence_refs when possible.
- use track_requirements to maintain a generic coverage checklist when the task has multiple required sources, filters, joins, metrics, units, or output constraints.
- use verify_alignment to classify semantic alternatives, document evidence, compute results, final candidates, and blocked/conflict decisions before relying on them.

Rules:
- Do not assume a business domain. Use only the user question, knowledge document text, and observations.
- knowledge.md is an authority for semantics, not a physical schema.
- Candidate sources, filenames, document windows, and semantic similarities are not bindings.
- PDF/MD/video are not structured tables. Documents require window evidence and either extracted record-set evidence before compute or direct document-window/value bindings before direct final.
- Video is unsupported in v1. It can be inspected for metadata but cannot support final evidence.
- Every physical field/table/path used for compute must come from observed evidence and verified bindings.
- Verifier decisions and requirement tracking are audit evidence, not physical data. They cannot replace real observations.
"""


TOOL_GUIDE = {
    "tool_protocol": "Use native tool calls. Text-only answers are not accepted.",
    "final_protocol": "Use submit_final(compute_ref=...) after run_verified_compute succeeds, or submit_final(answer=..., binding_refs=..., evidence_refs=...) for direct verified document/value evidence.",
    "binding_protocol": "Use bind(...) with evidence_refs before compute.",
    "requirement_protocol": "Use track_requirements(...) to declare/update generic required answer conditions when the answer depends on multiple constraints.",
    "verifier_protocol": "Use verify_alignment(...) to mark observed evidence/compute as bindable, candidate_answer, intermediate, not_applicable, needs_more_evidence, conflict, or blocked_ok.",
    "relation_protocol": "Use generated relation_name values such as rel_0001 in SQL.",
    "failure_protocol": "When a tool returns negative_scope or a repeated failure, switch tools or call blocked.",
    "blocked_protocol": "When calling blocked, cite evidence_refs that support the absence/conflict/exhaustion claim whenever possible.",
    "document_protocol": "For PDF/MD use profile_document or search_document before extract_records.",
    "sql_protocol": "Use inspect_relation after bind and before repairing failed SQL.",
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
    knowledge = [
        {
            "id": section.id,
            "heading_path": section.heading_path,
            "score": section.score,
            "line_start": section.line_start,
            "line_end": section.line_end,
            "text": section.text,
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
        _json_fragment("knowledge", knowledge, 12_000),
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
