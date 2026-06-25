from __future__ import annotations

from pathlib import Path
from typing import Any

from data_agent_baseline.evidence_agent.codex_loop.context import build_context_fragments
from data_agent_baseline.evidence_agent.codex_loop.protocol import LoopState, ModelAction
from data_agent_baseline.evidence_agent.codex_loop.registry import EvidenceActionRegistry
from data_agent_baseline.evidence_agent.knowledge import (
    build_knowledge_catalog,
    match_knowledge_sections,
)
from data_agent_baseline.evidence_agent.text import normalize_key


ROOT = Path(__file__).resolve().parents[1]


def _state_for_task(task_id: str, question: str) -> LoopState:
    context_dir = ROOT / "data" / "input" / task_id / "context"
    sections, lookup, _schema_json, _content_hash = build_knowledge_catalog(context_dir)
    state = LoopState(question=question, context_dir=context_dir)
    state.knowledge_sections = sections
    state.knowledge_lookup = lookup
    state.matched_sections = match_knowledge_sections(question, sections)
    return state


def _retrieve(state: LoopState, arguments: dict[str, Any]):
    registry = EvidenceActionRegistry()
    return registry.dispatch(
        state,
        ModelAction(
            kind="tool_call",
            tool_name="retrieve_knowledge",
            arguments=arguments,
        ),
    )


def test_knowledge_lookup_keeps_section_refs_for_real_tokens() -> None:
    state = _state_for_task(
        "task_40",
        "在我们的其他存款性公司资产负债表中，这些年来，总资产的最大值是多少，归属于哪一年 谢谢啊",
    )

    odc = state.knowledge_lookup[normalize_key("ed_otherdepositorycorpbs")]
    total_assets = state.knowledge_lookup[normalize_key("totalassets")]

    assert odc.section_refs
    assert total_assets.section_refs
    assert any(ref in total_assets.section_refs for ref in odc.section_refs)


def test_retrieve_knowledge_catalog_exposes_sections_when_question_match_is_empty() -> None:
    state = _state_for_task(
        "task_40",
        "在我们的其他存款性公司资产负债表中，这些年来，总资产的最大值是多少，归属于哪一年 谢谢啊",
    )

    evidence = _retrieve(state, {"mode": "catalog"})

    assert evidence.ok
    assert evidence.payload["mode"] == "catalog"
    section_text = "\n".join(
        f"{section['heading_path']} {section['preview']}"
        for section in evidence.payload["catalog"]["sections"]
    )
    assert "ed_otherdepositorycorpbs" in section_text
    assert "totalassets" in section_text.casefold()


def test_loop_context_exposes_catalog_even_without_matched_sections() -> None:
    state = _state_for_task(
        "task_40",
        "在我们的其他存款性公司资产负债表中，这些年来，总资产的最大值是多少，归属于哪一年 谢谢啊",
    )
    state.matched_sections = []

    fragments = build_context_fragments(state)
    catalog_fragment = next(fragment for fragment in fragments if fragment.kind == "knowledge_catalog")

    assert not catalog_fragment.truncated
    assert "ed_otherdepositorycorpbs" in catalog_fragment.text
    assert "totalassets" in catalog_fragment.text.casefold()
    assert "retrieve_knowledge" in catalog_fragment.text


def test_retrieve_knowledge_token_returns_complete_task40_slice() -> None:
    state = _state_for_task(
        "task_40",
        "在我们的其他存款性公司资产负债表中，这些年来，总资产的最大值是多少，归属于哪一年 谢谢啊",
    )

    evidence = _retrieve(
        state,
        {"mode": "token", "tokens": ["ed_otherdepositorycorpbs", "totalassets"]},
    )

    assert evidence.ok
    returned_text = "\n\n".join(section["text"] for section in evidence.payload["sections"])
    assert "ed_otherdepositorycorpbs" in returned_text
    assert "totalassets" in returned_text.casefold()
    assert evidence.payload["resolved_tokens"]


def test_retrieve_knowledge_token_returns_complete_task15_field_slice() -> None:
    state = _state_for_task(
        "task_15",
        "显示300707的出让前持股数量和出让前持股比例 谢谢",
    )

    evidence = _retrieve(
        state,
        {"mode": "token", "tokens": ["lc_sharetransfer", "SumBeforeTran", "PCTBeforeTran"]},
    )

    assert evidence.ok
    returned_text = "\n\n".join(section["text"] for section in evidence.payload["sections"])
    assert "lc_sharetransfer" in returned_text
    assert "sumbeforetran" in returned_text.casefold()
    assert "pctbeforetran" in returned_text.casefold()


def test_retrieve_knowledge_section_id_returns_exact_full_text() -> None:
    state = _state_for_task(
        "task_15",
        "显示300707的出让前持股数量和出让前持股比例 谢谢",
    )
    target = next(
        section for section in state.knowledge_sections if "sumbeforetran" in section.text.casefold()
    )

    evidence = _retrieve(state, {"mode": "section", "section_ids": [target.id]})

    assert evidence.ok
    assert len(evidence.payload["sections"]) == 1
    assert evidence.payload["sections"][0]["id"] == target.id
    assert evidence.payload["sections"][0]["line_start"] == target.line_start
    assert evidence.payload["sections"][0]["line_end"] == target.line_end
    assert evidence.payload["sections"][0]["text"] == target.text
