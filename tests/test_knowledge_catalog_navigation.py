from __future__ import annotations

from pathlib import Path
from typing import Any

from data_agent_baseline.evidence_agent.codex_loop.context import build_context_fragments
from data_agent_baseline.evidence_agent.codex_loop.inventory import build_inventory
from data_agent_baseline.evidence_agent.codex_loop.protocol import LoopState, ModelAction
from data_agent_baseline.evidence_agent.codex_loop.registry import EvidenceActionRegistry
from data_agent_baseline.evidence_agent.knowledge import (
    build_knowledge_catalog,
    build_semantic_cards,
    build_source_mappings,
    expand_semantic_card_dependencies,
    match_knowledge_sections,
    match_semantic_cards,
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
    state.sources = build_inventory(context_dir)
    for source in state.sources.values():
        state.source_by_path[source.id] = source.id
        state.source_by_path[source.virtual_path] = source.id
        state.source_by_path[source.path.as_posix()] = source.id
    state.semantic_cards = build_semantic_cards(sections)
    state.matched_semantic_cards = expand_semantic_card_dependencies(
        match_semantic_cards(question, state.semantic_cards),
        state.semantic_cards,
    )
    state.source_mappings = build_source_mappings(state.semantic_cards, tuple(state.sources.values()))
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
    semantic_fragment = next(fragment for fragment in fragments if fragment.kind == "semantic_knowledge")

    assert "ed_otherdepositorycorpbs" in catalog_fragment.text
    assert "retrieve_knowledge" in catalog_fragment.text
    assert not semantic_fragment.truncated
    assert "ed_otherdepositorycorpbs" in semantic_fragment.text
    assert "totalassets" in semantic_fragment.text.casefold()


def test_semantic_cards_cover_every_task3_knowledge_section() -> None:
    state = _state_for_task(
        "task_3",
        "基金经理 QDII 管理规模和学历相关分析",
    )

    card_section_ids = {card.section_id for card in state.semantic_cards}
    section_ids = {section.id for section in state.knowledge_sections}

    assert section_ids <= card_section_ids
    assert any(card.kind == "business_context" for card in state.semantic_cards)
    assert any(card.kind == "field" and card.name == "mf_personalinfo.education" for card in state.semantic_cards)


def test_semantic_cards_parse_three_column_field_tables() -> None:
    state = _state_for_task(
        "task_18",
        "我能看一下基金的十年回报率数据吗",
    )

    by_name = {card.name: card for card in state.semantic_cards}

    assert "mf_netvalueperformancehis.rrintenyear" in by_name
    assert by_name["mf_netvalueperformancehis.rrintenyear"].semantic_slot == "rrintenyear"
    assert "mf_netvalueperformancehis.mf_netvalueperformancehis" not in by_name


def test_semantic_cards_do_not_promote_ambiguity_tables_to_physical_fields() -> None:
    state = _state_for_task(
        "task_18",
        "我能看一下基金的十年回报率数据吗",
    )

    by_name = {card.name: card for card in state.semantic_cards}

    assert "mf_netvalueperformancehis.rrintenyear" in by_name
    assert "rrsincethisyear.rrintenyear" not in by_name
    assert "rrsincethisyear.annualizedrrsincestart" not in by_name


def test_semantic_units_require_explicit_unit_markers() -> None:
    state = _state_for_task(
        "task_32",
        "Help me check these data. Show the trading volume data for each trading day of 601908.",
    )
    turnover = next(card for card in state.semantic_cards if card.name == "qt_dailyquote.turnoverdeals")

    assert turnover.unit is None


def test_semantic_units_strip_markdown_punctuation_noise() -> None:
    state = _state_for_task(
        "task_3",
        "管理基金规模超过100亿的基金经理最高学历分布情况",
    )
    total = next(card for card in state.semantic_cards if card.name == "mf_fmscaleanalysisn.totalfundnv")

    assert total.unit == "亿元"


def test_source_mapping_does_not_infer_physical_fields_from_semantic_text_overlap() -> None:
    state = _state_for_task(
        "task_3",
        "管理基金规模超过100亿的基金经理最高学历分布情况",
    )
    total = next(card for card in state.semantic_cards if card.name == "mf_fmscaleanalysisn.totalfundnv")

    mappings = [mapping for mapping in state.source_mappings if mapping.card_id == total.id]

    assert any(
        mapping.status == "unverified_document_candidate"
        and mapping.source_path == "/context/doc/mf_fmscaleanalysisn.pdf"
        and mapping.physical_field is None
        for mapping in mappings
    )
    assert not any("semantic_term_overlap" in mapping.match_reason for mapping in mappings)
    assert not any(mapping.physical_field == "TotalAUM" for mapping in mappings)


def test_sqlite_source_mapping_uses_table_columns_for_field_grounding() -> None:
    state = _state_for_task(
        "task_40",
        "给咱们搜一下2005年以后，国内信贷和外币存款的数据 感谢",
    )
    domestic = next(
        card
        for card in state.semantic_cards
        if card.name == "ed_chinamoneyandbanking.domesticloans"
    )

    mappings = [mapping for mapping in state.source_mappings if mapping.card_id == domestic.id]

    assert any(
        mapping.status == "unverified_structured_candidate"
        and mapping.source_path == "/context/db/sub_db.sqlite"
        and mapping.physical_field == "DomesticLoans"
        for mapping in mappings
    )


def test_document_source_mapping_is_table_level_until_extracted() -> None:
    state = _state_for_task(
        "task_38",
        "给咱们搜一下2005年以后，国内信贷和外币存款的数据 感谢",
    )
    domestic = next(
        card
        for card in state.semantic_cards
        if card.name == "ed_chinamoneyandbanking.domesticloans"
    )

    mappings = [mapping for mapping in state.source_mappings if mapping.card_id == domestic.id]

    assert any(
        mapping.status == "unverified_document_candidate"
        and mapping.source_path == "/context/doc/ed_chinamoneyandbanking.pdf"
        and mapping.physical_field is None
        and "document_mapping_requires_extraction_plan" in mapping.warnings
        for mapping in mappings
    )


def test_semantic_source_mapping_distinguishes_rules_structured_and_documents() -> None:
    state = _state_for_task(
        "task_3",
        "找出 QDII 管理规模超过 200 亿且学历为硕士的基金经理",
    )

    by_card = {card.name: card for card in state.semantic_cards}
    business_card = next(card for card in state.semantic_cards if card.kind == "business_context")
    education_card = by_card["mf_personalinfo.education"]
    qdii_card = by_card["mf_fmscaleanalysisn.qdiinv"]

    business_mappings = [m for m in state.source_mappings if m.card_id == business_card.id]
    education_mappings = [m for m in state.source_mappings if m.card_id == education_card.id]
    qdii_mappings = [m for m in state.source_mappings if m.card_id == qdii_card.id]

    assert any(mapping.status == "semantic_only" for mapping in business_mappings)
    assert any(mapping.status == "unverified_structured_candidate" for mapping in education_mappings)
    assert any(mapping.status == "unverified_document_candidate" for mapping in qdii_mappings)


def test_retrieve_knowledge_semantic_returns_pure_cards_without_source_mappings() -> None:
    state = _state_for_task(
        "task_3",
        "QDII 管理规模超过 200 亿",
    )
    qdii_card = next(card for card in state.semantic_cards if card.name == "mf_fmscaleanalysisn.qdiinv")

    evidence = _retrieve(state, {"mode": "semantic", "card_ids": [qdii_card.id]})

    assert evidence.ok
    assert evidence.payload["mode"] == "semantic"
    assert evidence.payload["cards"][0]["name"] == "mf_fmscaleanalysisn.qdiinv"
    assert "source_mappings" not in evidence.payload["cards"][0]
    assert "data_form" not in evidence.payload["cards"][0]
    assert "locate_sources" in evidence.payload["usage_note"]


def test_loop_context_exposes_semantic_knowledge_as_primary_fragment() -> None:
    state = _state_for_task(
        "task_3",
        "QDII 管理规模超过 200 亿",
    )

    fragments = build_context_fragments(state)
    semantic_fragment = next(fragment for fragment in fragments if fragment.kind == "semantic_knowledge")
    source_resolution_fragment = next(fragment for fragment in fragments if fragment.kind == "source_resolution")

    assert "mf_fmscaleanalysisn.qdiinv" in semantic_fragment.text
    assert "source_mappings" not in semantic_fragment.text
    assert "source_plan" not in semantic_fragment.text
    assert "data_form" not in semantic_fragment.text
    assert "semantic cards" in semantic_fragment.text
    assert "source_plan" in source_resolution_fragment.text
    assert "selection_required" in source_resolution_fragment.text
    assert "unverified_document_candidate" not in source_resolution_fragment.text

    qdii_card = next(card for card in state.semantic_cards if card.name == "mf_fmscaleanalysisn.qdiinv")
    EvidenceActionRegistry().dispatch(
        state,
        ModelAction(
            kind="tool_call",
            tool_name="select_semantic_cards",
            arguments={"card_ids": [qdii_card.id], "rationale": "QDII scale is the relevant semantic field."},
        ),
    )
    selected_fragments = build_context_fragments(state)
    selected_resolution = next(fragment for fragment in selected_fragments if fragment.kind == "source_resolution")
    selected_candidates = next(fragment for fragment in selected_fragments if fragment.kind == "selected_source_candidates")
    assert "unverified_document_candidate" in selected_resolution.text
    assert "mf_fmscaleanalysisn" in selected_candidates.text
    assert "qdiinv" in selected_candidates.text


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
