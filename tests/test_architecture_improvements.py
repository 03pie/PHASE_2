from __future__ import annotations

import json

from data_agent_baseline.agents.middleware import _discovery_state
from data_agent_baseline.agents.middleware import _recover_analyze_plan_tagged_arguments
from data_agent_baseline.agents.semantic_layer import query_semantic_context
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.prompts.loader import (
    build_knowledge_bundle,
    build_task_prompt,
    read_knowledge_content,
)
from data_agent_baseline.tools.agent_tools.extract_narrative_records import (
    _coerce_string_list,
    _extract_multi_field_rows,
    _extract_rows,
)
from data_agent_baseline.tools.agent_tools.execute_python import (
    _try_preserve_source_projection,
)
from data_agent_baseline.tools.answer import validate_prepared_answer


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


def test_narrative_extractor_handles_before_quantity_and_percentage_records() -> None:
    rows, evidence, stats = _extract_multi_field_rows(
        [
            "记录 1 的转让方为甲。交易前，该实体持有 2,441,732 股，占总股本的 1.55%。交易后，其持股数降至 2,141,732 股，持股比例为 1.36%。",
            "记录 2 的转让方信息缺失。其交易前后的持股数量与比例也无法确定。",
        ],
        source_fields=["sumbeforetran", "pctbeforetran"],
        field_aliases=None,
        record_anchor="转让方",
        start_line=None,
        end_line=None,
        max_records=10,
    )

    assert rows == [[2441732.0, 0.0155], ["", ""]]
    assert [item["line_number"] for item in evidence] == [1, 2]
    assert stats["missing_counts"] == {
        "sumbeforetran": 1,
        "pctbeforetran": 1,
    }


def test_narrative_extractor_coerces_json_string_field_lists() -> None:
    assert _coerce_string_list('["field_a", "field_b"]') == [
        "field_a",
        "field_b",
    ]
    assert _coerce_string_list("field_a|field_b") == ["field_a", "field_b"]


def test_sql_knowledge_examples_expose_join_operation(tmp_path) -> None:
    context = tmp_path / "context"
    context.mkdir()
    (context / "knowledge.md").write_text(
        "\n".join(
            [
                "```sql",
                "SELECT a.key, COUNT(*)",
                "FROM source_a AS a",
                "JOIN source_b AS b ON a.key = b.key",
                "WHERE b.metric > 10",
                "GROUP BY a.key;",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    semantic = query_semantic_context(context, "metric distribution", max_matches=10)

    operations = {
        fact["operation"]
        for fact in semantic["knowledge_facts"]
        if fact["kind"] == "example_query"
    }
    assert "join,filter,aggregate" in operations


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


def test_task_prompt_focuses_knowledge_schema_sections(tmp_path) -> None:
    task_dir = tmp_path / "task"
    context = task_dir / "context"
    doc_dir = context / "doc"
    json_dir = context / "json"
    doc_dir.mkdir(parents=True)
    json_dir.mkdir()
    (context / "knowledge.md").write_text(
        "\n".join(
            [
                "### Primary Metrics (`metric_source`)",
                "| Column | Semantic Definition |",
                "|---|---|",
                "| `targetmetric` | Target metric value |",
                "",
                "```sql",
                "SELECT targetmetric FROM metric_source;",
                "```",
                "",
                "### Neighbor Metrics (`metric_neighbor`)",
                "| Column | Semantic Definition |",
                "|---|---|",
                "| `othermetric` | Unrelated neighbor value |",
            ]
        ),
        encoding="utf-8",
    )
    (doc_dir / "metric_source.md").write_text("targetmetric was 42.", encoding="utf-8")
    (json_dir / "metric_neighbor.json").write_text(
        '{"records":[{"OtherMetric": 7}]}',
        encoding="utf-8",
    )
    task = PublicTask(
        TaskRecord("task_focused_schema", "public", "Show target metric records."),
        TaskAssets(task_dir, context),
    )

    prompt = build_task_prompt(
        task,
        question_structure='{"output":{"requested_columns":["target_metric"]}}',
        knowledge_bundle=build_knowledge_bundle(context),
    )
    schema_text = prompt.split("<knowledge_schema>", 1)[1].split(
        "</knowledge_schema>",
        1,
    )[0]

    assert '"section_key": "metric_source"' in schema_text
    assert '"section_key": "metric_neighbor"' not in schema_text
    assert '"focus"' in schema_text


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


def test_answer_validation_rejects_ambiguous_structured_cell_projection() -> None:
    plan = {
        "output_spec": {
            "columns": [
                {"name": "company", "source_fields": ["Commission", "ChiNameAbbr"]}
            ],
            "row_policy": "transform",
            "transformations": [
                {
                    "operation": "sort",
                    "description": "Select the row authorized by the user.",
                    "authorization": {"source": "user", "quote": "highest"},
                }
            ],
            "expected_row_count": 1,
        },
        "evidence": {
            "context_sources": [{"path": "/context/json/lc_financialexpense.json"}],
        },
        "execution_spec": {
            "sources": [{"path": "/context/json/lc_financialexpense.json"}],
            "operations": [
                {
                    "operation": "sort",
                    "description": "Sort by Commission.",
                    "authorization": {"source": "user", "quote": "highest"},
                }
            ],
        },
    }
    columns = ["company"]
    rows = [[{"Commission": 144059466.29, "ChiNameAbbr": "广汇能源"}]]
    audit = {
        "source_paths": ["/context/json/lc_financialexpense.json"],
        "operations": ["sort by Commission desc", "limit 1"],
    }

    answer, error = validate_prepared_answer(columns, rows, plan, audit)

    assert answer is None
    assert error is not None
    assert "could not be uniquely projected" in error
    assert "Commission" in error
    assert "ChiNameAbbr" in error


def test_answer_validation_uses_unique_source_field_as_final_column() -> None:
    plan = {
        "output_spec": {
            "columns": [
                {
                    "name": "annualized_2yr_return",
                    "source_fields": ["AnnualizedRRInTwoYear"],
                }
            ],
            "row_policy": "preserve",
            "transformations": [],
            "expected_row_count": 2,
        }
    }

    answer, error = validate_prepared_answer(
        ["annualized_2yr_return"],
        [[15.97], [-6.42]],
        plan,
    )

    assert error is None
    assert answer is not None
    assert answer.columns == ["AnnualizedRRInTwoYear"]


def test_recovers_plan_sections_from_mixed_json_and_tags() -> None:
    recovered = _recover_analyze_plan_tagged_arguments(
        {
            "evidence": {
                "knowledge_status": "unavailable",
                "knowledge_rules": [],
                "context_sources": [{"path": "/context/db/sub_db.sqlite::treatment"}],
            },
            "execution_spec": (
                '{"sources":[{"path":"/context/db/sub_db.sqlite::treatment"}],'
                '"operations":[{"operation":"limit","authorization":'
                '{"source":"user","quote":"first"}}]}'
                "\n<intent>\n"
                '{"requirements":[{"quote":"first","requirement_type":"limit"}],'
                '"unresolved":[]}'
                "\n</intent>\n<output_spec>\n"
                '{"columns":[{"name":"first_procedure_time",'
                '"source_fields":["treatmenttime"]}],'
                '"row_policy":"transform","transformations":[]}'
                "\n</output_spec>"
            ),
        }
    )

    assert recovered["execution_spec"]["operations"][0]["operation"] == "limit"
    assert recovered["intent"]["requirements"][0]["quote"] == "first"
    assert recovered["output_spec"]["columns"][0]["source_fields"] == ["treatmenttime"]


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
