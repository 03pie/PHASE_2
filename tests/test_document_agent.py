from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import fitz

from data_agent_baseline.evidence_agent.codex_loop.context import build_context_fragments
from data_agent_baseline.evidence_agent.codex_loop.document_agent import DocumentAgent
from data_agent_baseline.evidence_agent.codex_loop.guard import guard_action
from data_agent_baseline.evidence_agent.codex_loop.inventory import build_inventory
from data_agent_baseline.evidence_agent.codex_loop.protocol import LoopState, ModelAction, SourceRef
from data_agent_baseline.evidence_agent.codex_loop.registry import EvidenceActionRegistry
from data_agent_baseline.evidence_agent.knowledge import (
    build_knowledge_catalog,
    build_semantic_cards,
    build_source_mappings,
    expand_semantic_card_dependencies,
    match_semantic_cards,
)


ROOT = Path(__file__).resolve().parents[1]


def _state_with_doc(task_id: str, relative_path: str, data_form: str = "markdown_document") -> LoopState:
    path = ROOT / "data" / "input" / task_id / "context" / relative_path
    virtual_path = "/context/" + relative_path.replace("\\", "/")
    source = SourceRef(
        id="src_0001",
        path=path,
        virtual_path=virtual_path,
        basename=path.name,
        stem=path.stem,
        suffix=path.suffix,
        data_form=data_form,  # type: ignore[arg-type]
        size_bytes=path.stat().st_size,
    )
    state = LoopState(question="Total Assets", context_dir=path.parent.parent)
    state.sources[source.id] = source
    state.source_by_path[source.id] = source.id
    state.source_by_path[source.virtual_path] = source.id
    state.source_by_path[source.path.as_posix()] = source.id
    return state


def _dispatch(state: LoopState, tool_name: str, arguments: dict[str, Any]):
    return EvidenceActionRegistry().dispatch(
        state,
        ModelAction(kind="tool_call", tool_name=tool_name, arguments=arguments),
    )


def _task_state_with_knowledge(task_id: str, question: str) -> LoopState:
    context_dir = ROOT / "data" / "input" / task_id / "context"
    sections, lookup, _schema_json, _content_hash = build_knowledge_catalog(context_dir)
    state = LoopState(question=question, context_dir=context_dir)
    state.sources = build_inventory(context_dir)
    for source in state.sources.values():
        state.source_by_path[source.id] = source.id
        state.source_by_path[source.virtual_path] = source.id
        state.source_by_path[source.path.as_posix()] = source.id
    state.knowledge_sections = sections
    state.knowledge_lookup = lookup
    state.semantic_cards = build_semantic_cards(sections)
    state.matched_semantic_cards = expand_semantic_card_dependencies(
        match_semantic_cards(question, state.semantic_cards),
        state.semantic_cards,
    )
    state.source_mappings = build_source_mappings(state.semantic_cards, tuple(state.sources.values()))
    return state


def test_document_tool_surface_is_document_agent_only() -> None:
    registry = EvidenceActionRegistry()

    assert "run_document_agent" in registry.tool_names
    assert "preview_document" not in registry.tool_names
    assert "search_document" not in registry.tool_names
    assert "read_document_slice" not in registry.tool_names
    assert "extract_records" not in registry.tool_names


def test_markdown_document_index_uses_record_slices_not_line_windows() -> None:
    state = _state_with_doc("task_40", "doc/ed_otherdepositorycorpbs.md")
    agent = DocumentAgent()

    indexes = agent.ensure_indexes(state)
    index = indexes["src_0001"]

    assert index.slice_count > 10
    first = index.slices[0]
    assert first.slice_id == "src_0001_record_0001"
    assert first.line_start is not None
    assert first.line_end is not None
    assert first.text
    assert all("slice_lines" not in item.public_dict(include_text=True) for item in index.slices[:5])


def test_document_agent_search_read_extracts_validated_semantic_records() -> None:
    state = _state_with_doc("task_40", "doc/ed_otherdepositorycorpbs.md")
    agent = DocumentAgent()
    agent.ensure_indexes(state)

    search = agent.search_document_records(
        state,
        {"query": "Total Assets", "semantic_fields": ["metric"], "source_ref": "src_0001"},
    )
    assert search["ok"]
    slice_id = ""
    read: dict[str, Any] | None = None
    for match in search["payload"]["matches"]:
        candidate = agent.read_record_slice(state, {"slice_ids": [match["slice_id"]]})
        if "Total Assets" in candidate["payload"]["text"]:
            slice_id = match["slice_id"]
            read = candidate
            break
    assert slice_id

    assert read is not None
    assert read["ok"]
    assert "Total Assets" in read["payload"]["text"]

    extraction = agent.extract_semantic_records(
        state,
        {
            "records": [{"slice_id": slice_id, "metric": "Total Assets"}],
            "slice_decisions": [{"slice_id": slice_id, "status": "record_extracted"}],
            "target_schema": {"fields": ["metric"], "record_grain": "document_record"},
        },
    )

    assert extraction["ok"]
    assert extraction["payload"]["records"][0]["metric"] == "Total Assets"
    assert extraction["payload"]["records"][0]["provenance"]["slice_id"] == slice_id
    assert extraction["payload"]["processed_slice_ids"] == [slice_id]


def test_run_document_agent_returns_compact_package_for_main_loop() -> None:
    state = _state_with_doc("task_40", "doc/ed_otherdepositorycorpbs.md")
    agent = DocumentAgent()
    matches = agent.search_document_records(
        state,
        {"query": "Total Assets", "semantic_fields": ["metric"], "source_ref": "src_0001"},
    )["payload"]["matches"]
    slice_id = next(
        match["slice_id"]
        for match in matches
        if "Total Assets"
        in agent.read_record_slice(state, {"slice_ids": [match["slice_id"]]})["payload"]["text"]
    )

    evidence = _dispatch(
        state,
        "run_document_agent",
        {
            "question": "Total Assets",
            "target_fields": ["metric"],
            "source_candidates": ["src_0001"],
            "required_record_grain": "document_record",
            "records": [{"slice_id": slice_id, "metric": "Total Assets"}],
            "slice_decisions": [{"slice_id": slice_id, "status": "record_extracted"}],
        },
    )

    assert evidence.ok
    assert evidence.tool_name == "run_document_agent"
    assert evidence.payload["records"][0]["metric"] == "Total Assets"
    assert evidence.payload["processed_slice_ids"] == [slice_id]
    assert "text" not in evidence.payload
    assert state.document_agent_packages


def test_main_context_exposes_document_agent_summary_not_document_text() -> None:
    state = _state_with_doc("task_40", "doc/ed_otherdepositorycorpbs.md")
    DocumentAgent().ensure_indexes(state)

    fragments = build_context_fragments(state)
    doc_fragment = next(fragment for fragment in fragments if fragment.kind == "document_agent")

    assert "run_document_agent" in doc_fragment.text
    assert "indexed_document_count" in doc_fragment.text
    assert len(doc_fragment.text) < 6_000


def test_pdf_index_can_merge_record_across_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "cross_page.pdf"
    document = fitz.open()
    page_1 = document.new_page()
    page_1.insert_text((72, 72), "记录 001 Total")
    page_2 = document.new_page()
    page_2.insert_text((72, 72), "Assets 123 million")
    document.save(pdf_path)
    document.close()

    source = SourceRef(
        id="src_0001",
        path=pdf_path,
        virtual_path="/context/doc/cross_page.pdf",
        basename=pdf_path.name,
        stem=pdf_path.stem,
        suffix=".pdf",
        data_form="pdf_document",
        size_bytes=pdf_path.stat().st_size,
    )
    state = LoopState(question="Total Assets", context_dir=tmp_path)
    state.sources[source.id] = source
    state.source_by_path[source.id] = source.id
    state.source_by_path[source.virtual_path] = source.id
    state.source_by_path[source.path.as_posix()] = source.id

    index = DocumentAgent().ensure_indexes(state)["src_0001"]

    assert index.page_count == 2
    assert index.slices
    assert any(item.page_start == 1 and item.page_end == 2 for item in index.slices)


def test_partial_document_record_set_is_rejected_for_compute() -> None:
    state = _state_with_doc("task_40", "doc/ed_otherdepositorycorpbs.md")
    evidence = state.add_evidence(
        tool_name="run_document_agent",
        ok=True,
        summary="partial package",
        payload={
            "records": [{"metric": "Total Assets"}],
            "partial_coverage": True,
            "coverage_summary": {"processed_slice_count": 1, "total_slice_count": 10},
        },
        source_id="src_0001",
        data_form="markdown_document",
    )
    binding = state.add_binding(
        binding_type="document_record_set",
        evidence_refs=(evidence.id,),
        source_id="src_0001",
        allowed_columns=("metric",),
        metadata={"records": [{"metric": "Total Assets"}], "partial_coverage": True},
    )
    action = ModelAction(
        kind="compute",
        sql=f"SELECT * FROM {binding.relation_name}",
        binding_refs=(binding.id,),
    )

    decision = guard_action(state, action, EvidenceActionRegistry())

    assert not decision.allowed
    assert "partial document record-set" in decision.reason
    assert "run_document_agent" in decision.allowed_next_tools


def test_document_agent_without_model_does_not_synthesize_records() -> None:
    state = _task_state_with_knowledge(
        "task_3",
        "管理基金规模超过100亿的基金经理最高学历分布情况",
    )
    target_card = next(card for card in state.semantic_cards if card.name == "3.2 Total Fund Net Value (AUM per Manager)")

    evidence = EvidenceActionRegistry(document_agent=DocumentAgent()).dispatch(
        state,
        ModelAction(
            kind="tool_call",
            tool_name="run_document_agent",
            arguments={
                "question": state.question,
                "target_fields": ["personalcode", "totalfundnv"],
                "semantic_cards": [target_card.to_dict()],
                "source_candidates": ["/context/doc/mf_fmscaleanalysisn.pdf"],
                "required_record_grain": "fund_manager",
                "coverage_policy": {"initial_slice_limit": 12},
            },
        ),
    )

    assert evidence.ok
    assert evidence.payload["records"] == []
    assert evidence.payload["partial_coverage"]
    assert "semantic_extraction_requires_document_agent_model_or_provided_records" in evidence.payload["remaining_risks"]


def test_extract_records_by_plan_joins_cross_section_pdf_records_to_gold_counts() -> None:
    state = _task_state_with_knowledge(
        "task_3",
        "管理基金规模超过100亿的基金经理最高学历分布情况",
    )

    result = DocumentAgent().extract_records_by_plan(
        state,
        {
            "source_ref": "/context/doc/mf_fmscaleanalysisn.pdf",
            "target_fields": ["personalcode", "totalfundnv"],
            "record_grain": "fund_manager",
            "merge_key": "档案号",
            "entity_anchor": {"label": "档案"},
            "section_scope": {
                "include_hints": ["资产总规模", "总资产净值", "基金总净值"],
                "exclude_hints": ["权益", "混合", "债券", "QDII", "货币"],
            },
            "fields": {
                "personalcode": {
                    "value_type": "identifier",
                    "aliases": ["PersonalCode", "个人识别码"],
                },
                "totalfundnv": {
                    "value_type": "amount",
                    "unit": "亿元",
                    "aliases": ["总资产净值", "资产总规模", "基金总净值", "资产总净值"],
                },
            },
            "required_fields": ["personalcode", "totalfundnv"],
        },
    )

    assert result["ok"]
    payload = result["payload"]
    assert payload["plan_status"] == "validated"
    records = payload["records"]
    assert len(records) == 50
    assert all(record["provenance"]["slice_ids"] for record in records)

    db_source = next(source for source in state.sources.values() if source.virtual_path.endswith("sub_db.sqlite"))
    with sqlite3.connect(db_source.path) as connection:
        education_by_code = {
            int(personalcode): education
            for personalcode, education in connection.execute(
                "SELECT PersonalCode, Education FROM mf_personalinfo"
            )
        }
    counts = Counter(
        education_by_code[int(record["personalcode"])]
        for record in records
        if float(record["totalfundnv"]) > 100
    )
    assert counts == Counter(
        {
            "Doctoral degree": 5,
            "Master's degree": 44,
            "Postdoctoral researcher": 1,
        }
    )


def test_extract_records_by_plan_preserves_missing_doc_rows_and_disambiguates_returns() -> None:
    state = _task_state_with_knowledge(
        "task_18",
        "我能看一下基金的十年回报率数据吗",
    )

    result = DocumentAgent().extract_records_by_plan(
        state,
        {
            "source_ref": "/context/doc/mf_netvalueperformancehis.md",
            "target_fields": ["rrintenyear"],
            "record_grain": "fund",
            "merge_key": "档案号",
            "entity_anchor": {"labels": ["档案", "战略单元"]},
            "section_scope": {
                "include_hints": ["五年及十年维度", "超长期业绩分析阶段"],
                "exclude_hints": ["自成立以来的整体表现"],
            },
            "fields": {
                "rrintenyear": {
                    "value_type": "amount",
                    "unit": "%",
                    "aliases": ["十年回报率", "近十年累计回报率"],
                },
            },
            "include_missing_records": True,
            "required_fields": [],
        },
    )

    assert result["ok"]
    records = result["payload"]["records"]
    assert len(records) == 49
    assert [
        (index, record["rrintenyear"])
        for index, record in enumerate(records)
        if str(record.get("rrintenyear")).strip()
    ] == [
        (0, 166.097944),
        (8, 216.829262),
        (12, 78.852695),
    ]
    assert all(record["provenance"]["slice_ids"] for record in records)
    assert not any(
        "基金战略定位" in record["provenance"]["evidence_text"]
        for record in records
    )


def test_run_document_agent_filters_cross_source_fields_and_enriches_cards() -> None:
    state = _task_state_with_knowledge(
        "task_3",
        "管理基金规模超过100亿的基金经理最高学历分布情况",
    )

    evidence = EvidenceActionRegistry(document_agent=DocumentAgent()).dispatch(
        state,
        ModelAction(
            kind="tool_call",
            tool_name="run_document_agent",
            arguments={
                "question": state.question,
                "target_fields": ["personalcode", "totalfundnv", "education"],
                "semantic_cards": [
                    {
                        "id": "sem_0009",
                        "canonical_table": "mf_personalinfo",
                        "canonical_field": "education",
                    },
                    {
                        "id": "sem_0012",
                        "canonical_table": "mf_fmscaleanalysisn",
                        "canonical_field": "personalcode",
                    },
                    {
                        "id": "sem_0013",
                        "canonical_table": "mf_fmscaleanalysisn",
                        "canonical_field": "totalfundnv",
                    },
                ],
                "source_candidates": ["src_0012"],
                "required_record_grain": "fund_manager",
            },
        ),
    )

    assert evidence.ok
    assert evidence.payload["doc_task"]["target_fields"] == ["personalcode", "totalfundnv"]
    card_names = {card.get("name") for card in evidence.payload["doc_task"]["semantic_cards"]}
    assert "mf_fmscaleanalysisn.personalcode" in card_names
    assert "mf_personalinfo.education" not in card_names
    assert any(
        "Foreign key referencing the fund manager" in card.get("definition", "")
        for card in evidence.payload["doc_task"]["semantic_cards"]
    )


def test_locate_sources_frontloads_canonical_semantic_source_plan() -> None:
    state = _task_state_with_knowledge(
        "task_3",
        "管理基金规模超过100亿的基金经理最高学历分布情况",
    )
    evidence = _dispatch(
        state,
        "locate_sources",
        {"query": "管理基金规模超过100亿的基金经理最高学历分布情况"},
    )

    assert evidence.ok
    total_plan = next(
        item
        for item in evidence.payload["semantic_source_plan"]
        if item["canonical_field"] == "mf_fmscaleanalysisn.totalfundnv"
    )
    assert any(
        mapping["status"] == "document_source"
        and mapping["source_path"] == "/context/doc/mf_fmscaleanalysisn.pdf"
        and mapping["binding_priority"] == "preferred"
        for mapping in total_plan["source_mappings"]
    )
    personalcode_plan = next(
        item
        for item in evidence.payload["semantic_source_plan"]
        if item["canonical_field"] == "mf_fmscaleanalysisn.personalcode"
    )
    assert any(
        mapping["status"] == "fallback_candidate"
        and mapping["source_path"] == "/context/json/mf_fmretscaleanalysis.json"
        and mapping["binding_priority"] == "fallback_only"
        for mapping in personalcode_plan["source_mappings"]
    )
    assert any(
        candidate.get("canonical_field") == "mf_fmscaleanalysisn.totalfundnv"
        and candidate.get("mapping_status") == "document_source"
        and candidate.get("recommended_tool") == "run_document_agent"
        for candidate in evidence.payload["candidates"]
    )


def test_compute_guard_no_longer_blocks_canonical_gap_after_source_planning() -> None:
    state = _task_state_with_knowledge(
        "task_3",
        "管理基金规模超过100亿的基金经理最高学历分布情况",
    )
    json_source = next(source for source in state.sources.values() if source.virtual_path.endswith("mf_fmretscaleanalysis.json"))
    evidence = state.add_evidence(
        tool_name="inspect_source",
        ok=True,
        summary="JSON observed",
        payload={
            "source_id": json_source.id,
            "path": json_source.virtual_path,
            "data_form": json_source.data_form,
            "columns": ["PersonalCode", "TotalAUM"],
        },
        source_id=json_source.id,
        data_form=json_source.data_form,
    )
    binding = state.add_binding(
        binding_type="structured_source",
        evidence_refs=(evidence.id,),
        source_id=json_source.id,
        allowed_columns=("PersonalCode", "TotalAUM"),
    )

    decision = guard_action(
        state,
        ModelAction(
            kind="compute",
            sql=f"SELECT MAX(TotalAUM) FROM {binding.relation_name}",
            binding_refs=(binding.id,),
        ),
        EvidenceActionRegistry(),
    )

    assert decision.allowed
