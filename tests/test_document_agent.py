from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz

from data_agent_baseline.evidence_agent.codex_loop.controller import CodexEvidenceController, _audit_final
from data_agent_baseline.evidence_agent.codex_loop.context import build_context_fragments
from data_agent_baseline.evidence_agent.codex_loop.compute import (
    parse_date_key,
    run_sql_over_bindings,
)
from data_agent_baseline.evidence_agent.codex_loop.document_agent import (
    DocEvidencePackage,
    DocTask,
    DocumentAgent,
    RecordSlice,
    _candidate_fields_for_slice,
    _candidate_record_gap_snippets,
    _candidate_review_snippets,
    _merge_scan_records,
    _task_candidate_terms,
)
from data_agent_baseline.evidence_agent.codex_loop.guard import guard_action
from data_agent_baseline.evidence_agent.codex_loop.inventory import build_inventory
from data_agent_baseline.evidence_agent.codex_loop.protocol import LoopState, ModelAction, SourceRef
from data_agent_baseline.evidence_agent.codex_loop.registry import (
    EvidenceActionRegistry,
    _enrich_document_agent_arguments,
)
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


def test_document_full_scan_ambiguous_slices_are_not_partial_coverage() -> None:
    state = _state_with_doc("task_40", "doc/ed_otherdepositorycorpbs.md")
    agent = DocumentAgent()
    index = agent.ensure_indexes(state)["src_0001"]
    slice_decisions = [
        {
            "slice_id": item.slice_id,
            "status": "ambiguous",
            "reason": "LLM could not determine relevance from this slice.",
        }
        for item in index.slices
    ]

    extraction = agent.extract_semantic_records(
        state,
        {
            "records": [],
            "slice_decisions": slice_decisions,
            "target_schema": {"fields": ["metric"], "record_grain": "document_record"},
        },
    )

    assert extraction["ok"]
    assert not extraction["payload"]["partial_coverage"]
    assert len(extraction["payload"]["ambiguous_slice_ids"]) == index.slice_count
    assert extraction["payload"]["coverage_summary"]["processed_slice_count"] == index.slice_count


def test_document_unsupported_value_invalid_preserves_scan_coverage() -> None:
    state = _state_with_doc("task_40", "doc/ed_otherdepositorycorpbs.md")
    agent = DocumentAgent()
    index = agent.ensure_indexes(state)["src_0001"]
    first_slice = index.slices[0]
    decisions = [
        {
            "slice_id": item.slice_id,
            "status": "record_extracted" if item.slice_id == first_slice.slice_id else "no_relevant_record",
        }
        for item in index.slices
    ]

    extraction = agent.extract_semantic_records(
        state,
        {
            "records": [{"slice_id": first_slice.slice_id, "metric": "definitely-not-present-12345"}],
            "slice_decisions": decisions,
            "target_schema": {"fields": ["metric"], "record_grain": "document_record"},
        },
    )

    assert not extraction["ok"]
    assert extraction["payload"]["invalid"][0]["error"] == "unsupported_extracted_values"
    assert not extraction["payload"]["partial_coverage"]
    assert extraction["payload"]["coverage_summary"]["processed_slice_count"] == index.slice_count


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


def test_scan_ledger_merges_partial_records_by_llm_anchor() -> None:
    records = _merge_scan_records(
        [
            {"record_anchor": "unit-1", "enddate": "2020-01-02", "slice_id": "s1"},
            {"record_anchor": "unit-1", "totalassets": "123", "slice_id": "s2"},
            {"record_anchor": "unit-2", "enddate": "2021-03-04", "slice_id": "s3"},
        ]
    )

    assert records[0]["record_anchor"] == "unit-1"
    assert records[0]["enddate"] == "2020-01-02"
    assert records[0]["totalassets"] == "123"
    assert records[0]["slice_ids"] == ["s1", "s2"]
    assert records[1]["record_anchor"] == "unit-2"


def test_text_validation_accepts_evidence_supported_cjk_values(tmp_path: Path) -> None:
    doc_path = tmp_path / "records.md"
    doc_path.write_text(
        "记录 1\n基金简称：贝莱德基金\n成立日期：2020年9月10日\n",
        encoding="utf-8",
    )
    source = SourceRef(
        id="src_0001",
        path=doc_path,
        virtual_path="/context/doc/records.md",
        basename=doc_path.name,
        stem=doc_path.stem,
        suffix=".md",
        data_form="markdown_document",
        size_bytes=doc_path.stat().st_size,
    )
    state = LoopState(question="records", context_dir=tmp_path)
    state.sources[source.id] = source
    state.source_by_path[source.id] = source.id
    state.source_by_path[source.virtual_path] = source.id
    index = DocumentAgent().ensure_indexes(state)["src_0001"]
    slice_id = index.slices[0].slice_id

    result = DocumentAgent().extract_semantic_records(
        state,
        {
            "records": [
                {
                    "investadvisorabbrname": "贝莱德基金",
                    "establishmentdate": "2020年9月10日",
                    "slice_ids": [slice_id],
                }
            ],
            "slice_decisions": [{"slice_id": slice_id, "status": "record_extracted"}],
            "target_schema": {
                "fields": ["investadvisorabbrname", "establishmentdate"],
                "required": ["investadvisorabbrname", "establishmentdate"],
            },
            "fields": {
                "investadvisorabbrname": {"value_type": "text", "aliases": ["基金简称"]},
                "establishmentdate": {"value_type": "date", "aliases": ["成立日期"]},
            },
        },
    )

    assert result["ok"], result
    assert result["payload"]["records"][0]["investadvisorabbrname"] == "贝莱德基金"


def test_document_validation_does_not_guess_field_semantics(tmp_path: Path) -> None:
    doc_path = tmp_path / "records.md"
    doc_path.write_text("Total Assets: 123\n", encoding="utf-8")
    source = SourceRef(
        id="src_0001",
        path=doc_path,
        virtual_path="/context/doc/records.md",
        basename=doc_path.name,
        stem=doc_path.stem,
        suffix=".md",
        data_form="markdown_document",
        size_bytes=doc_path.stat().st_size,
    )
    state = LoopState(question="records", context_dir=tmp_path)
    state.sources[source.id] = source
    state.source_by_path[source.id] = source.id
    state.source_by_path[source.virtual_path] = source.id
    slice_id = DocumentAgent().ensure_indexes(state)["src_0001"].slices[0].slice_id

    result = DocumentAgent().extract_semantic_records(
        state,
        {
            "records": [{"assets": "123", "slice_id": slice_id}],
            "slice_decisions": [{"slice_id": slice_id, "status": "record_extracted"}],
            "target_schema": {"fields": ["totalassets"]},
        },
    )

    assert result["ok"], result
    assert "totalassets" not in result["payload"]["records"][0]


def test_document_validation_accepts_generic_record_field_wrappers(tmp_path: Path) -> None:
    doc_path = tmp_path / "records.md"
    doc_path.write_text("Total Assets: 123\nReporting period end date: 2022\n", encoding="utf-8")
    source = SourceRef(
        id="src_0001",
        path=doc_path,
        virtual_path="/context/doc/records.md",
        basename=doc_path.name,
        stem=doc_path.stem,
        suffix=".md",
        data_form="markdown_document",
        size_bytes=doc_path.stat().st_size,
    )
    state = LoopState(question="records", context_dir=tmp_path)
    state.sources[source.id] = source
    state.source_by_path[source.id] = source.id
    state.source_by_path[source.virtual_path] = source.id
    slice_id = DocumentAgent().ensure_indexes(state)["src_0001"].slices[0].slice_id

    result = DocumentAgent().extract_semantic_records(
        state,
        {
            "records": [{"record_totalassets": "123", "record_enddate": "2022", "slice_id": slice_id}],
            "slice_decisions": [{"slice_id": slice_id, "status": "record_extracted"}],
            "target_schema": {"fields": ["totalassets", "enddate"]},
        },
    )

    assert result["ok"], result
    assert result["payload"]["records"][0]["totalassets"] == "123"
    assert result["payload"]["records"][0]["enddate"] == "2022"


def test_document_validation_rejects_when_llm_changes_evidence_format(tmp_path: Path) -> None:
    doc_path = tmp_path / "records.md"
    doc_path.write_text("Total Assets: 1,234\n", encoding="utf-8")
    source = SourceRef(
        id="src_0001",
        path=doc_path,
        virtual_path="/context/doc/records.md",
        basename=doc_path.name,
        stem=doc_path.stem,
        suffix=".md",
        data_form="markdown_document",
        size_bytes=doc_path.stat().st_size,
    )
    state = LoopState(question="records", context_dir=tmp_path)
    state.sources[source.id] = source
    state.source_by_path[source.id] = source.id
    state.source_by_path[source.virtual_path] = source.id
    slice_id = DocumentAgent().ensure_indexes(state)["src_0001"].slices[0].slice_id

    result = DocumentAgent().extract_semantic_records(
        state,
        {
            "records": [{"totalassets": "1234", "slice_id": slice_id}],
            "slice_decisions": [{"slice_id": slice_id, "status": "record_extracted"}],
            "target_schema": {"fields": ["totalassets"]},
        },
    )

    assert not result["ok"]
    assert result["payload"]["invalid"][0]["error"] == "unsupported_extracted_values"
    assert result["payload"]["records"] == []


def test_compute_parse_date_key_orders_mixed_date_strings(tmp_path: Path) -> None:
    assert parse_date_key("2020年9月10日") == 20200910
    assert parse_date_key("二零一五年十二月二十四日") == 20151224

    state = LoopState(question="latest date", context_dir=tmp_path)
    evidence = state.add_evidence(
        tool_name="run_document_agent",
        ok=True,
        summary="records",
        payload={},
    )
    binding = state.add_binding(
        binding_type="document_record_set",
        evidence_refs=(evidence.id,),
        allowed_columns=("name", "establishmentdate"),
        metadata={
            "records": [
                {"name": "东证融汇", "establishmentdate": "二零一五年十二月二十四日"},
                {"name": "贝莱德基金", "establishmentdate": "2020年9月10日"},
            ],
        },
    )

    columns, rows, evidence_refs = run_sql_over_bindings(
        state,
        sql=(
            f"SELECT name, establishmentdate FROM {binding.relation_name} "
            "ORDER BY parse_date_key(establishmentdate) DESC NULLS LAST LIMIT 1"
        ),
        binding_refs=(binding.id,),
    )

    assert columns == ("name", "establishmentdate")
    assert rows == (("贝莱德基金", "2020年9月10日"),)
    assert evidence_refs == (evidence.id,)


def test_partial_document_record_set_is_not_blocked_for_compute() -> None:
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

    assert decision.allowed


def test_partial_document_record_set_compute_allowed_from_evidence_lineage() -> None:
    state = _state_with_doc("task_40", "doc/ed_otherdepositorycorpbs.md")
    evidence = state.add_evidence(
        tool_name="run_document_agent",
        ok=True,
        summary="partial package",
        payload={
            "records": [{"metric": "Total Assets"}],
            "partial_coverage": True,
            "uncertain_slices": [{"slice_id": "src_0001_record_0106"}],
            "doc_task": {
                "question": state.question,
                "target_fields": ["metric"],
                "source_candidates": ["src_0001"],
                "semantic_cards": [],
                "coverage_policy": {},
            },
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
    decision = guard_action(
        state,
        ModelAction(kind="compute", sql=f"SELECT metric FROM {binding.relation_name}", binding_refs=(binding.id,)),
        EvidenceActionRegistry(),
    )

    assert decision.allowed


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
    assert evidence.payload["uncertain_slices"]
    assert evidence.payload["uncertain_slices"][0]["candidate_fields"] == ["personalcode", "totalfundnv"]


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
                        "semantic_scope": "mf_personalinfo",
                        "semantic_slot": "education",
                    },
                    {
                        "id": "sem_0012",
                        "semantic_scope": "mf_fmscaleanalysisn",
                        "semantic_slot": "personalcode",
                    },
                    {
                        "id": "sem_0013",
                        "semantic_scope": "mf_fmscaleanalysisn",
                        "semantic_slot": "totalfundnv",
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
    selected_card_ids = [
        next(card.id for card in state.semantic_cards if card.name == "mf_fmscaleanalysisn.totalfundnv"),
        next(card.id for card in state.semantic_cards if card.name == "mf_fmscaleanalysisn.personalcode"),
    ]
    _dispatch(
        state,
        "select_semantic_cards",
        {"card_ids": selected_card_ids, "rationale": "scale and manager key are relevant to the requested answer."},
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
        if item["semantic_slot_id"] == "mf_fmscaleanalysisn.totalfundnv"
    )
    assert any(
        mapping["status"] == "unverified_document_candidate"
        and mapping["source_path"] == "/context/doc/mf_fmscaleanalysisn.pdf"
        and mapping["binding_priority"] == "grounding_candidate"
        for mapping in total_plan["grounding_candidates"]
    )
    personalcode_plan = next(
        item
        for item in evidence.payload["semantic_source_plan"]
        if item["semantic_slot_id"] == "mf_fmscaleanalysisn.personalcode"
    )
    assert any(
        mapping["status"] == "unverified_lexical_candidate"
        and mapping["source_path"] == "/context/json/mf_fmretscaleanalysis.json"
        and mapping["binding_priority"] == "lexical_candidate"
        for mapping in personalcode_plan["grounding_candidates"]
    )
    assert any(
        candidate.get("semantic_slot_id") == "mf_fmscaleanalysisn.totalfundnv"
        and candidate.get("mapping_status") == "unverified_document_candidate"
        and candidate.get("recommended_tool") == "run_document_agent"
        for candidate in evidence.payload["candidates"]
    )


def test_document_agent_argument_enrichment_adds_target_field_cards() -> None:
    state = _task_state_with_knowledge(
        "task_8",
        "Which fund company was established most recently?",
    )
    date_card = next(card for card in state.semantic_cards if card.name == "mf_investadvisoroutline.establishmentdate")

    enriched = _enrich_document_agent_arguments(
        state,
        {
            "question": state.question,
            "target_fields": ["investadvisorabbrname", "establishmentdate"],
            "semantic_cards": [date_card.to_dict()],
            "source_candidates": ["/context/doc/mf_investadvisoroutline.md"],
        },
    )

    card_names = {card["name"] for card in enriched["semantic_cards"]}
    assert "mf_investadvisoroutline.investadvisorabbrname" in card_names
    assert "mf_investadvisoroutline.establishmentdate" in card_names


def test_document_agent_argument_enrichment_prefers_semantic_slots_over_final_aliases() -> None:
    state = _task_state_with_knowledge(
        "task_40",
        "在我们的其他存款性公司资产负债表中，这些年来，总资产的最大值是多少，归属于哪一年 谢谢啊",
    )
    total_card = next(card for card in state.semantic_cards if card.name == "ed_otherdepositorycorpbs.totalassets")
    date_card = next(card for card in state.semantic_cards if card.name == "ed_otherdepositorycorpbs.enddate")

    enriched = _enrich_document_agent_arguments(
        state,
        {
            "question": state.question,
            "target_fields": ["total_assets_max", "year"],
            "semantic_cards": [total_card.to_dict(), date_card.to_dict()],
            "source_candidates": ["/context/doc/ed_otherdepositorycorpbs.md"],
        },
    )

    assert enriched["target_fields"] == ["enddate", "totalassets"]


def test_document_agent_argument_enrichment_does_not_require_all_targets_when_preserving_missing() -> None:
    state = _task_state_with_knowledge(
        "task_40",
        "在我们的其他存款性公司资产负债表中，这些年来，总资产的最大值是多少，归属于哪一年 谢谢啊",
    )
    _dispatch = EvidenceActionRegistry().dispatch
    _dispatch(
        state,
        ModelAction(
            kind="tool_call",
            tool_name="declare_answer_contract",
            arguments={
                "intent_summary": "find the maximum total assets value and its year",
                "answer_grain": "one row for the selected maximum record",
                "final_outputs": ["total_assets_max", "year"],
                "constraints": [],
                "operations": {
                    "row_shape": "single_row",
                    "sort_by": ["totalassets"],
                    "top_n": 1,
                    "reason": "select the row with maximum total assets",
                },
                "helper_fields": {"sort_fields": ["totalassets"], "row_selection_fields": ["enddate"]},
                "row_shape": "single_row",
                "null_policy": "preserve",
                "transform_intent": "find maximum total assets and corresponding year",
                "document_policy": {"include_missing_records": True, "required_fields": []},
                "unresolved_terms": [],
            },
        ),
    )
    total_card = next(card for card in state.semantic_cards if card.name == "ed_otherdepositorycorpbs.totalassets")
    date_card = next(card for card in state.semantic_cards if card.name == "ed_otherdepositorycorpbs.enddate")

    enriched = _enrich_document_agent_arguments(
        state,
        {
            "question": state.question,
            "target_fields": ["totalassets", "enddate"],
            "semantic_cards": [total_card.to_dict(), date_card.to_dict()],
            "source_candidates": ["/context/doc/ed_otherdepositorycorpbs.md"],
            "coverage_policy": {
                "include_missing_records": True,
                "required_fields": ["totalassets", "enddate"],
            },
        },
    )

    assert enriched["coverage_policy"]["include_missing_records"] is True
    assert enriched["coverage_policy"]["required_fields"] == []


def test_document_candidate_hints_are_derived_from_task_semantics() -> None:
    task = DocTask(
        question="find the largest total assets value",
        target_fields=("totalassets", "enddate"),
        semantic_cards=(
            {
                "name": "ledger.totalassets",
                "semantic_scope": "ledger",
                "semantic_slot": "totalassets",
                "definition": "Total assets",
            },
            {
                "name": "ledger.enddate",
                "semantic_scope": "ledger",
                "semantic_slot": "enddate",
                "definition": "Reporting period end date",
            },
        ),
    )
    terms = _task_candidate_terms(task)

    candidates = _candidate_fields_for_slice("Unit A began 2022 with Total Assets of 1,234,567.", terms)

    assert "totalassets" in candidates


def test_document_candidate_no_relevant_decision_becomes_uncertain_slice() -> None:
    target_slice = RecordSlice(
        slice_id="slice_1",
        source_id="src_1",
        path="/context/doc.md",
        data_form="markdown_document",
        slice_index=1,
        text="Unit A began 2022 with Total Assets of 1,234,567.",
    )

    risks = _candidate_review_snippets(
        [target_slice],
        [{"slice_id": target_slice.slice_id, "status": "no_relevant_record", "reason": "missed"}],
        {target_slice.slice_id: ("totalassets",)},
    )

    assert risks
    assert risks[0]["slice_id"] == target_slice.slice_id
    assert risks[0]["candidate_fields"] == ["totalassets"]
    assert "1,234,567" in risks[0]["evidence_text"]


def test_document_candidate_record_extracted_without_field_becomes_gap() -> None:
    target_slice = RecordSlice(
        slice_id="slice_1",
        source_id="src_1",
        path="/context/doc.md",
        data_form="markdown_document",
        slice_index=1,
        text="Unit A began 2022 with Total Assets of 1,234,567.",
    )

    risks = _candidate_record_gap_snippets(
        [target_slice],
        [{"slice_id": target_slice.slice_id, "status": "record_extracted"}],
        [{"provenance": {"slice_ids": [target_slice.slice_id]}}],
        {target_slice.slice_id: ("totalassets",)},
    )

    assert risks
    assert risks[0]["slice_id"] == target_slice.slice_id
    assert risks[0]["candidate_fields"] == ["totalassets"]


def test_document_agent_remaining_candidate_risk_marks_payload_partial(tmp_path: Path) -> None:
    class RiskyDocumentAgent:
        def run(self, state: LoopState, task: DocTask) -> DocEvidencePackage:
            del state, task
            return DocEvidencePackage(
                records=({"provenance": {"slice_ids": ["s1"]}},),
                record_schema={"fields": ["metric"]},
                source_refs=("src_0001",),
                evidence_refs=("s1",),
                processed_slice_ids=("s1",),
                no_relevant_slice_ids=(),
                ambiguous_slice_ids=(),
                coverage_summary={"total_slice_count": 1, "processed_slice_count": 1},
                remaining_risks=("unresolved_candidate_document_slices",),
            )

    state = LoopState(question="extract metric", context_dir=tmp_path)
    path = tmp_path / "doc.md"
    path.write_text("metric evidence", encoding="utf-8")
    source = SourceRef(
        id="src_0001",
        path=path,
        virtual_path="/context/doc.md",
        basename=path.name,
        stem=path.stem,
        suffix=path.suffix,
        data_form="markdown_document",
        size_bytes=path.stat().st_size,
    )
    state.sources[source.id] = source

    evidence = EvidenceActionRegistry(document_agent=RiskyDocumentAgent()).dispatch(
        state,
        ModelAction(
            kind="tool_call",
            tool_name="run_document_agent",
            arguments={"question": state.question, "source_candidates": [source.id], "target_fields": ["metric"]},
        ),
    )

    assert evidence.ok
    assert evidence.payload["partial_coverage"] is True
    assert evidence.recommended_next_actions[0]["tool_name"] == "bind"


def test_compute_guard_does_not_infer_required_fields_without_contract() -> None:
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


def test_compute_guard_allows_lexical_candidate_with_explicit_physical_mapping() -> None:
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
        metadata={
            "semantic_contract": {
                "canonical_fields": ["mf_fmscaleanalysisn.totalfundnv"],
                "physical_field_mapping": {
                    "mf_fmscaleanalysisn.totalfundnv": {"field": "TotalAUM"},
                },
            },
        },
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


def test_compute_guard_rejects_semantic_contract_without_physical_column() -> None:
    state = _task_state_with_knowledge(
        "task_8",
        "Which fund company was established most recently?",
    )
    evidence = state.add_evidence(
        tool_name="run_document_agent",
        ok=True,
        summary="document records missing one requested field",
        payload={
            "records": [{"establishmentdate": "2020年9月10日"}],
            "partial_coverage": False,
        },
        source_id="src_0013",
        data_form="markdown_document",
    )
    binding = state.add_binding(
        binding_type="document_record_set",
        evidence_refs=(evidence.id,),
        source_id="src_0013",
        allowed_columns=("establishmentdate",),
        metadata={
            "records": [{"establishmentdate": "2020年9月10日"}],
            "semantic_contract": {
                "canonical_fields": [
                    "mf_investadvisoroutline.establishmentdate",
                    "mf_investadvisoroutline.investadvisorabbrname",
                ],
                "physical_field_mapping": {
                    "mf_investadvisoroutline.establishmentdate": {"field": "establishmentdate"},
                    "mf_investadvisoroutline.investadvisorabbrname": {"field": "investadvisorabbrname"},
                },
            },
        },
    )

    decision = guard_action(
        state,
        ModelAction(
            kind="compute",
            sql=f"SELECT establishmentdate FROM {binding.relation_name}",
            binding_refs=(binding.id,),
        ),
        EvidenceActionRegistry(),
    )

    assert not decision.allowed
    assert "mf_investadvisoroutline.investadvisorabbrname" in decision.reason


def test_direct_final_allowed_from_evidence_backed_document_binding() -> None:
    state = _task_state_with_knowledge(
        "task_8",
        "Which fund company was established most recently?",
    )
    evidence = state.add_evidence(
        tool_name="run_document_agent",
        ok=True,
        summary="document records extracted",
        payload={"records": [{"establishmentdate": "2020年9月10日"}]},
        source_id="src_0013",
        data_form="markdown_document",
    )
    binding = state.add_binding(
        binding_type="document_record_set",
        evidence_refs=(evidence.id,),
        source_id="src_0013",
        allowed_columns=("establishmentdate",),
        metadata={"records": [{"establishmentdate": "2020年9月10日"}]},
    )
    decision = guard_action(
        state,
        ModelAction(
            kind="final",
            answer={"fund_company": "贝莱德基金", "establishmentdate": "2020年9月10日"},
            binding_refs=(binding.id,),
            evidence_refs=(evidence.id,),
        ),
        EvidenceActionRegistry(),
    )

    assert decision.allowed


def test_document_record_bind_uses_union_of_all_record_columns(tmp_path: Path) -> None:
    state = LoopState(question="max metric by date", context_dir=tmp_path)
    evidence = state.add_evidence(
        tool_name="run_document_agent",
        ok=True,
        summary="document records extracted",
        payload={
            "records": [
                {"metric": "141.37", "provenance": {"slice_ids": ["s1"]}},
                {"metric": "147.38", "date": "2013", "provenance": {"slice_ids": ["s2"]}},
            ],
            "partial_coverage": False,
        },
        source_id="src_0001",
        data_form="markdown_document",
    )

    result = CodexEvidenceController._apply_bind(
        object(),
        state,
        ModelAction(
            kind="bind",
            binding_type="document_record_set",
            evidence_refs=(evidence.id,),
            arguments={"allowed_columns": ["date"]},
        ),
    )

    assert result.ok
    binding = next(iter(state.bindings.values()))
    assert binding.allowed_columns == ("date", "metric")
    columns, rows, _refs = run_sql_over_bindings(
        state,
        sql=f"SELECT metric, date FROM {binding.relation_name}",
        binding_refs=(binding.id,),
    )
    assert columns == ("metric", "date")
    assert rows[0] == ("141.37", "")
    assert rows[1] == ("147.38", "2013")


def test_final_audit_accepts_evidence_backed_direct_document_answer() -> None:
    state = _task_state_with_knowledge(
        "task_8",
        "Which fund company was established most recently?",
    )
    evidence = state.add_evidence(
        tool_name="run_document_agent",
        ok=True,
        summary="document records extracted",
        payload={"records": [{"fund_company": "贝莱德基金", "establishmentdate": "2020年9月10日"}]},
        source_id="src_0013",
        data_form="markdown_document",
    )
    binding = state.add_binding(
        binding_type="document_record_set",
        evidence_refs=(evidence.id,),
        source_id="src_0013",
        allowed_columns=("fund_company", "establishmentdate"),
        metadata={"records": [{"fund_company": "贝莱德基金", "establishmentdate": "2020年9月10日"}]},
    )
    state.final_answer = {
        "columns": ["fund_company", "establishmentdate"],
        "rows": [["贝莱德基金", "2020年9月10日"]],
        "binding_refs": [binding.id],
        "evidence_refs": [evidence.id],
    }

    audit = _audit_final(state)

    assert "direct_final_without_direct_binding" not in audit["unsupported_operations"]
    assert audit["passed"], audit
