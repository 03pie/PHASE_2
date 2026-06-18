from __future__ import annotations

import json
from types import SimpleNamespace

from data_agent_baseline.agents.middleware import _fact_targets_request
from data_agent_baseline.agents.middleware import _discovery_state
from data_agent_baseline.agents.semantic_layer import query_semantic_context
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.prompts.loader import (
    build_knowledge_bundle,
    build_task_prompt,
    read_knowledge_content,
)
from data_agent_baseline.tools.agent_tools.extract_narrative_records import _extract_rows
from data_agent_baseline.tools.agent_tools.execute_python import (
    _try_preserve_source_projection,
)
from data_agent_baseline.tools.answer import validate_prepared_answer


def test_fact_target_matching_uses_field_level_evidence() -> None:
    requirements = [
        {
            "quote": "其他国外资产和国外负债的数据记录",
            "statement": "Other foreign assets and foreign liabilities data records",
            "requirement_type": "measure",
        },
        {
            "quote": "还是在货币当局资产负债表中",
            "statement": "monetary authority balance sheet",
            "requirement_type": "entity",
        },
    ]
    original_request = (
        "你好，那你再查一下对其他国外资产和国外负债的数据记录，"
        "还是在货币当局资产负债表中"
    )

    assert _fact_targets_request(
        fact=SimpleNamespace(
            field_key="otherforeignassets",
            quote="Foreign assets not classified as forex or gold",
        ),
        original_request=original_request,
        requirements=requirements,
    )
    assert _fact_targets_request(
        fact=SimpleNamespace(
            field_key="abroadliability",
            quote="Total liabilities owed to foreign entities",
        ),
        original_request=original_request,
        requirements=requirements,
    )
    assert not _fact_targets_request(
        fact=SimpleNamespace(
            field_key="totalassets",
            quote="Total assets held by the monetary authority",
        ),
        original_request=original_request,
        requirements=requirements,
    )
    assert not _fact_targets_request(
        fact=SimpleNamespace(
            field_key="forex",
            quote="Foreign exchange reserves held as assets",
        ),
        original_request=original_request,
        requirements=requirements,
    )


def test_fact_target_matching_does_not_translate_cross_language_filters() -> None:
    requirements = [
        {
            "quote": "准备金存款",
            "statement": "Filter by '准备金存款' item",
            "requirement_type": "filter",
        }
    ]
    original_request = "列出准备金存款在其他存款性公司资产负债表中的数据记录"

    assert not _fact_targets_request(
        fact=SimpleNamespace(
            section_key="ed_otherdepositorycorpbs",
            field_key="depositswithcentralbank",
            quote="Reserves and deposits placed by ODCs at the central bank (PBoC)",
        ),
        original_request=original_request,
        requirements=requirements,
    )
    assert not _fact_targets_request(
        fact=SimpleNamespace(
            section_key="ed_chinafibalancesheetrmb",
            field_key="corporatesavings",
            quote="Name could imply investment savings | corporate deposit balances (存款), not investment",
        ),
        original_request=original_request,
        requirements=requirements,
    )
    assert not _fact_targets_request(
        fact=SimpleNamespace(
            section_key="ed_moneyauthoritybs",
            field_key="reservedeposits",
            quote="Deposits held by banks at the monetary authority",
        ),
        original_request=original_request,
        requirements=requirements,
    )


def test_semantic_bindings_keep_section_scope(tmp_path) -> None:
    context = tmp_path / "context"
    doc_dir = context / "doc"
    csv_dir = context / "csv"
    doc_dir.mkdir(parents=True)
    csv_dir.mkdir()
    (context / "knowledge.md").write_text(
        "\n".join(
            [
                "### Other Depository Corporations Balance Sheet (`ed_otherdepositorycorpbs`)",
                "| Column | Semantic Definition | Business Role |",
                "|---|---|---|",
                "| `depositswithcentralbank` | Reserves and deposits placed by ODCs at the central bank (PBoC) | reserve deposits |",
                "",
                "### RMB Credit Balance Sheet (`ed_chinafibalancesheetrmb`)",
                "| Column | Semantic Definition | Business Role |",
                "|---|---|---|",
                "| `corporatesavings` | corporate deposit balances (存款), not investment | deposits |",
            ]
        ),
        encoding="utf-8",
    )
    (doc_dir / "ed_otherdepositorycorpbs.md").write_text(
        "For unit 16, DepositsWithCentralBank was 1,817,970.",
        encoding="utf-8",
    )
    (csv_dir / "ed_moneyauthoritybs.csv").write_text(
        "id,ReserveDeposits\n1,288460\n",
        encoding="utf-8",
    )

    semantic = query_semantic_context(context, "准备金存款", max_matches=10)

    assert semantic["source_candidates"] == []

    semantic = query_semantic_context(
        context,
        "准备金存款",
        max_matches=10,
        scope="ed_otherdepositorycorpbs",
    )

    assert [
        fact["field_key"]
        for fact in semantic["knowledge_facts"]
    ] == ["depositswithcentralbank"]
    assert [
        item["source_path"]
        for item in semantic["source_candidates"]
    ] == ["/context/doc/ed_otherdepositorycorpbs.md"]
    assert [
        item["source_path"]
        for item in semantic["section_bindings"]["ed_otherdepositorycorpbs"]
    ] == ["/context/doc/ed_otherdepositorycorpbs.md"]


def test_narrative_extractor_handles_field_adjacent_values() -> None:
    rows, evidence = _extract_rows(
        [
            (
                "For strategic unit 16, total ReserveAssets were recorded at "
                "1,986,140. This position was composed of 1,817,970 in "
                "DepositsWithCentralBank, supplemented by 168,170 in CashInVault."
            ),
            (
                "This, combined with its DepositsWithCentralBank of "
                "20,138,585.53, resulted in total ReserveAssets of 21,065,020."
            ),
            (
                "DepositsWithCentralBank were initially logged at 21.5 million. "
                "The final audited number was confirmed to be 21,606,280."
            ),
            (
                "The official CashInVault was corrected to 613,778.92. This "
                "amount, combined with 22,381,441.08 in DepositsWithCentralBank, "
                "resulted in the final ReserveAssets total of 22,995,220."
            ),
        ],
        source_field="depositswithcentralbank",
        knowledge_quote="Reserves and deposits placed by ODCs at the central bank",
        start_line=None,
        end_line=None,
        max_records=10,
    )

    assert rows == [[1817970.0], [20138585.53], [21606280.0], [22381441.08]]
    assert [item["record_count"] for item in evidence] == [1, 1, 1, 1]


def test_narrative_extractor_uses_field_value_across_line_breaks() -> None:
    rows, evidence = _extract_rows(
        [
            "Archive 91. The profile identifier is PersonalCode",
            "101001473 and the record is now verified.",
            "Archive 92. PersonalCode is unavailable.",
        ],
        source_field="personalcode",
        knowledge_quote="Unique identifier for a person",
        start_line=None,
        end_line=None,
        max_records=10,
    )

    assert rows == [[101001473.0]]
    assert evidence[0]["line_number"] == 1


def test_task_prompt_injects_structured_knowledge_schema(tmp_path) -> None:
    task_dir = tmp_path / "task"
    context = task_dir / "context"
    doc_dir = context / "doc"
    doc_dir.mkdir(parents=True)
    (context / "knowledge.md").write_text(
        "\n".join(
            [
                "### Example Table (`example_table`)",
                "| Column | Semantic Definition |",
                "|---|---|",
                "| `MetricValue` | Target metric value |",
            ]
        ),
        encoding="utf-8",
    )
    (doc_dir / "example_table.md").write_text(
        "MetricValue was 42.",
        encoding="utf-8",
    )
    task = PublicTask(
        TaskRecord("task_schema", "public", "列出指标值"),
        TaskAssets(task_dir, context),
    )

    bundle = build_knowledge_bundle(context)
    prompt = build_task_prompt(
        task,
        question_structure="<structure>",
        knowledge_bundle=bundle,
    )

    assert "<knowledge_schema>" in prompt
    assert "<context_knowledge>" not in prompt
    assert '"knowledge_status_for_plan": "authoritative"' in prompt
    assert f'"content_hash": "{bundle.content_hash}"' in prompt
    assert '"binding_status": "narrative_only"' in prompt
    assert '"fact_id": "kf_1"' in prompt
    assert '"section_key": "example_table"' in prompt
    assert '"source_path": "/context/doc/example_table.md"' in prompt
    assert bundle.raw_content == read_knowledge_content(context)
    assert "| `MetricValue` | Target metric value |" in read_knowledge_content(context)


def test_discovery_uses_state_knowledge_without_raw_prompt() -> None:
    raw_knowledge = "\n".join(
        [
            "### Example Table (`example_table`)",
            "| Column | Semantic Definition |",
            "|---|---|",
            "| `MetricValue` | Target metric value |",
        ]
    )

    discovery = _discovery_state([], {"knowledge_content": raw_knowledge})

    assert discovery.knowledge_present
    assert discovery.knowledge_available
    assert raw_knowledge in discovery.knowledge_content


def test_answer_validation_projects_demoted_supporting_columns() -> None:
    plan = {
        "output_spec": {
            "columns": [{"name": "ThirdIndustryGDP", "source_fields": ["ThirdIndustryGDP"]}],
            "row_policy": "preserve",
            "transformations": [],
            "expected_row_count": 2,
        },
        "execution_spec": {
            "supporting_fields": [
                {"name": "EndDate", "source_fields": ["EndDate"], "purpose": "context"},
                {"name": "Province", "source_fields": ["Province"], "purpose": "context"},
            ]
        },
    }
    columns = ["EndDate", "Province", "ThirdIndustryGDP"]
    rows = [
        ["2000-12-31 00:00:00", "北京", 144528.0],
        ["2000-12-31 00:00:00", "天津", 74565.0],
    ]

    answer, error = validate_prepared_answer(columns, rows, plan)

    assert error is None
    assert answer is not None
    assert answer.columns == ["ThirdIndustryGDP"]
    assert answer.rows == [[144528.0], [74565.0]]


def test_preserve_source_projection_uses_plan_columns(tmp_path) -> None:
    context = tmp_path / "context"
    context.mkdir()
    (tmp_path / "scratch").mkdir()
    source = context / "records.json"
    source.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "EndDate": "2000-12-31 00:00:00",
                        "Province": "北京",
                        "ThirdIndustryGDP": 144528.0,
                    },
                    {
                        "EndDate": "2000-12-31 00:00:00",
                        "Province": "天津",
                        "ThirdIndustryGDP": 74565.0,
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plan = {
        "output_spec": {
            "columns": [{"name": "ThirdIndustryGDP", "source_fields": ["ThirdIndustryGDP"]}],
            "row_policy": "preserve",
            "transformations": [],
            "expected_row_count": 2,
        },
        "evidence": {
            "context_sources": [{"path": "/context/records.json"}],
        },
    }

    answer = _try_preserve_source_projection(workspace=tmp_path, analysis_plan=plan)

    assert answer is not None
    assert answer.columns == ["ThirdIndustryGDP"]
    assert answer.rows == [[144528.0], [74565.0]]
