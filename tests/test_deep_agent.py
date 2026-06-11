from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from data_agent_baseline.agents.knowledge_schema import markdown_knowledge_to_schema
from data_agent_baseline.agents.prompts import BENCHMARK_SYSTEM_PROMPT, SUBAGENT_SYSTEM_PROMPT
from data_agent_baseline.agents.react import DeepAgent, DeepAgentConfig, _task_prompt
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]
    auto_bootstrap: bool = True
    call_count: int = 0
    bound_tool_sets: list[set[str]] = Field(default_factory=list)
    bound_tool_choices: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling-model"

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if not self.auto_bootstrap:
            return
        first_tool_name = ""
        if self.responses and self.responses[0].tool_calls:
            first_tool_name = str(self.responses[0].tool_calls[0].get("name") or "")
        if first_tool_name != "analyze_plan":
            self.responses.insert(
                0,
                AIMessage(
                    content="Analyze the request and establish the execution plan.",
                    tool_calls=[
                        {
                            "name": "analyze_plan",
                            "args": {
                                "intent": "Solve the requested benchmark task.",
                                "output_spec": "Return the requested result as a table.",
                                "intent_confidence": 0.9,
                                "confidence_reason": (
                                    "The complete question directly states the requested result."
                                ),
                                "steps": ["Inspect evidence", "Compute and validate", "Submit"],
                                "delegation_candidates": [],
                            },
                            "id": "plan-call",
                            "type": "tool_call",
                        }
                    ],
                ),
            )
        second_tool_name = ""
        if len(self.responses) > 1 and self.responses[1].tool_calls:
            second_tool_name = str(self.responses[1].tool_calls[0].get("name") or "")
        if second_tool_name != "write_todos":
            self.responses.insert(
                1,
                AIMessage(
                    content="Convert the analysis plan into an actionable todo list.",
                    tool_calls=[
                        {
                            "name": "write_todos",
                            "args": {
                                "todos": [
                                    {"content": "Inspect evidence", "status": "in_progress"},
                                    {"content": "Compute and validate", "status": "pending"},
                                    {"content": "Submit the result", "status": "pending"},
                                ]
                            },
                            "id": "todos-call",
                            "type": "tool_call",
                        }
                    ],
                ),
            )

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ScriptedChatModel:
        del kwargs
        names = {
            str(getattr(tool, "name", tool.get("name") if isinstance(tool, dict) else ""))
            for tool in tools
        }
        self.bound_tool_sets.append(names)
        self.bound_tool_choices.append(tool_choice)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        if self.call_count >= len(self.responses):
            raise RuntimeError("No scripted model responses remaining.")
        message = self.responses[self.call_count]
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_benchmark_prompt_is_only_the_caller_policy_layer() -> None:
    assert BENCHMARK_SYSTEM_PROMPT
    assert "You are a deep agent" not in BENCHMARK_SYSTEM_PROMPT
    assert "answer" in BENCHMARK_SYSTEM_PROMPT


def test_subagent_prompt_cannot_submit_final_answer() -> None:
    assert "answer" in SUBAGENT_SYSTEM_PROMPT
    assert "/context/" in SUBAGENT_SYSTEM_PROMPT


def test_knowledge_markdown_is_converted_to_schema_entries() -> None:
    schema = markdown_knowledge_to_schema(
        """
# Knowledge Guide: demo

Database ID: ignored

## Gross Domestic Product (`ed_grossdomesticproduct`)

| Column | Semantic Definition | Unit |
|---|---|---|
| `thirdindustrygdp` | GDP contribution from the tertiary sector | 浜垮厓 |

- Annual records use the period end date.
"""
    )

    assert {
        "id": "0001",
        "path": "Gross Domestic Product (ed_grossdomesticproduct)",
        "type": "table_row",
        "text": (
            "In Gross Domestic Product (ed_grossdomesticproduct), Column: "
            "thirdindustrygdp, Semantic Definition: GDP contribution from the tertiary "
            "sector, Unit: 浜垮厓"
        ),
    } in schema
    assert any(entry["type"] == "list_item" for entry in schema)
    assert all("Database ID" not in entry["text"] for entry in schema)


def test_task_prompt_attaches_usable_knowledge_schema(tmp_path: Path) -> None:
    task = _make_task_with_knowledge(
        tmp_path,
        """
# Knowledge Guide: demo

## Loans (`loan_table`)

| Column | Semantic Definition | Unit |
|---|---|---|
| `amount` | New loan amount | 浜垮厓 |
""",
    )

    prompt = _task_prompt(task)

    assert "Knowledge schema extracted from /context/knowledge.md" in prompt
    assert "authoritative reference for data definitions" in prompt
    assert "loan_table" in prompt
    assert "New loan amount" in prompt


def test_task_prompt_handles_invalid_knowledge_with_data_consistency(tmp_path: Path) -> None:
    task = _make_task_with_knowledge(
        tmp_path,
        "# Step3 skipped\n\nevidence 涓虹┖锛屾湭鐢熸垚 knowledge guide銆俓n",
    )

    prompt = _task_prompt(task)

    assert "no usable schema could be extracted" in prompt
    assert "keep every conclusion consistent with observed data" in prompt


def _tool_response(
    name: str,
    args: dict[str, Any],
    tool_call_id: str,
    *,
    content: str = "",
) -> AIMessage:
    args = dict(args)
    if name in {"analyze_plan", "revise_plan"}:
        args.setdefault("intent_confidence", 0.9)
        args.setdefault(
            "confidence_reason",
            "The complete question and inspected task evidence support this plan.",
        )
    return AIMessage(
        content=content,
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": tool_call_id,
                "type": "tool_call",
            }
        ],
    )


def _answer_response(
    *,
    columns: list[str],
    rows: list[list[Any]],
    tool_call_id: str = "answer-call",
) -> AIMessage:
    return _tool_response(
        "answer",
        {"columns": columns, "rows": rows},
        tool_call_id,
        content="The result table is ready.",
    )


def _llm_steps(result: Any) -> list[Any]:
    return [
        step
        for step in result.steps
        if "llm" in step.action_input
    ]


def _tool_calls(result: Any) -> list[tuple[Any, dict[str, Any]]]:
    return [
        (step, tool_call)
        for step in result.steps
        for tool_call in step.observation.get("tool_calls", [])
    ]


def _tool_call(result: Any, tool_call_id: str) -> tuple[Any, dict[str, Any]]:
    return next(
        (step, tool_call)
        for step, tool_call in _tool_calls(result)
        if tool_call.get("tool_call_id") == tool_call_id
    )


def _make_task_with_knowledge(tmp_path: Path, knowledge_text: str) -> PublicTask:
    task_dir = tmp_path / "task_knowledge"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "knowledge.md").write_text(knowledge_text, encoding="utf-8")
    (context_dir / "data.csv").write_text("amount\n1\n", encoding="utf-8")
    return PublicTask(
        record=TaskRecord(
            task_id="task_knowledge",
            difficulty="easy",
            question="Return the amount.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


@pytest.fixture
def public_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "data.txt").write_text("hello from context\n", encoding="utf-8")
    (context_dir / "sample.csv").write_text("name,value\nalpha,1\n", encoding="utf-8")
    (context_dir / "sample.json").write_text('{"value": 1}\n', encoding="utf-8")
    (context_dir / "sample.sqlite").write_bytes(b"")
    (context_dir / "sample.pdf").write_bytes(b"%PDF-1.4\n")
    (context_dir / "sample.mp4").write_bytes(b"video")
    return PublicTask(
        record=TaskRecord(
            task_id="task_1",
            difficulty="easy",
            question="Return the observed value.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_reads_context_and_submits_answer(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "read_file",
                {"file_path": "/context/data.txt"},
                "read-call",
            ),
            _answer_response(columns=["value"], rows=[["hello from context"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.to_dict() == {
        "columns": ["value"],
        "rows": [["hello from context"]],
    }
    assert [step.action for step in result.steps[:3]] == [
        "system_prompt",
        "user_prompt",
        "analyze_plan",
    ]
    tool_calls = _tool_calls(result)
    assert [tool_call["name"] for _, tool_call in tool_calls] == [
        "analyze_plan",
        "write_todos",
        "read_file",
        "answer",
    ]
    assert [tool_call["tool_call_id"] for _, tool_call in tool_calls] == [
        "plan-call",
        "todos-call",
        "read-call",
        "answer-call",
    ]
    assert [step.action_input["llm_call_index"] for step, _ in tool_calls] == [1, 2, 3, 4]


def test_analyze_plan_accepts_json_string_list_arguments(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Extract the observed value records.",
                    "output_spec": "Return the value column.",
                    "steps": '["Inspect evidence", "Submit"]',
                    "delegation_candidates": "[]",
                    "operation_type": "column_extract",
                    "target_tables": '["sample.csv"]',
                    "target_fields": '["value"]',
                    "filters": "[]",
                    "group_by": "[]",
                    "output_columns": '["value"]',
                    "ambiguities": "[]",
                    "requested_outputs": '["value"]',
                    "scope_evidence": "[]",
                    "request_mode_evidence": '["Return"]',
                    "field_bindings": '{"value": "value"}',
                },
                "json-string-plan",
            ),
            _answer_response(columns=["value"], rows=[["1"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "json-string-plan")
    assert plan_call["name"] == "analyze_plan"
    assert plan_call["ok"] is True
    assert "sample.csv" in json.dumps(plan_call["result"])


def test_aggregate_plan_for_raw_record_question_is_rejected(
    public_task: PublicTask,
) -> None:
    raw_record_task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question="Find value records.",
        ),
        assets=public_task.assets,
    )
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Construct a national total from regional rows.",
                    "output_spec": "Return yearly totals.",
                    "steps": ["Group by year", "Sum values"],
                    "delegation_candidates": [],
                    "operation_type": "aggregate",
                    "target_tables": ["sample.csv"],
                    "target_fields": ["value"],
                    "group_by": ["year"],
                    "aggregation": "sum(value)",
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Find"],
                    "field_bindings": {},
                    "grouping_evidence": ["records"],
                    "transformation_evidence": ["No national record exists."],
                },
                "bad-aggregate-plan",
            ),
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return the observed raw value records.",
                    "output_spec": "One-column table containing value.",
                    "steps": ["Inspect raw value", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "column_extract",
                    "target_tables": ["sample.csv"],
                    "target_fields": ["value"],
                    "output_columns": ["value"],
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Find"],
                    "field_bindings": {"value": "value"},
                },
                "raw-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit raw value", "status": "completed"}]},
                "todos-complete",
            ),
            _answer_response(columns=["value"], rows=[["1"]]),
        ],
    )

    result = DeepAgent(model=model).run(raw_record_task)

    assert result.succeeded
    bad_step, bad_call = _tool_call(result, "bad-aggregate-plan")
    assert bad_step.ok is False
    assert bad_call["ok"] is False
    assert "Request contract rejected" in json.dumps(bad_call["result"])
    _, raw_call = _tool_call(result, "raw-plan")
    assert raw_call["ok"] is True


def test_rejected_revision_does_not_block_a_corrected_revision(
    public_task: PublicTask,
) -> None:
    raw_record_task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question="Find value records.",
        ),
        assets=public_task.assets,
    )
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return raw records.",
                    "output_spec": "Return the value column.",
                    "steps": ["Inspect", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "column_extract",
                    "target_fields": ["value"],
                    "output_columns": ["value"],
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Find"],
                    "field_bindings": {"value": "value"},
                    "intent_confidence": 0.65,
                    "confidence_reason": (
                        "The complete question is clear about value but the source "
                        "mapping has not yet been inspected."
                    ),
                },
                "initial-raw-plan",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Inspect", "status": "in_progress"},
                        {"content": "Submit", "status": "pending"},
                    ]
                },
                "initial-todos",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/sample.csv"},
                "revision-evidence",
            ),
            _tool_response(
                "revise_plan",
                {
                    "revision_reason": "No national row exists.",
                    "evidence": ["Only regional rows were observed."],
                    "intent": "Construct a national total.",
                    "output_spec": "Return yearly totals.",
                    "steps": ["Aggregate", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "aggregate",
                    "target_fields": ["value"],
                    "group_by": ["year"],
                    "aggregation": "sum(value)",
                    "output_columns": ["year", "total"],
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Find"],
                    "field_bindings": {
                        "year": "records",
                        "total": "value",
                    },
                    "grouping_evidence": ["records"],
                    "transformation_evidence": ["No national row exists."],
                },
                "rejected-revision",
            ),
            _tool_response(
                "revise_plan",
                {
                    "revision_reason": "The question asks for raw records.",
                    "evidence": ["The source contains the requested value column."],
                    "intent": "Return the source value column without aggregation.",
                    "output_spec": "Return one value column in source order.",
                    "steps": ["Extract the source column", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "column_extract",
                    "target_fields": ["value"],
                    "preserve_raw_rows": True,
                    "output_columns": ["value"],
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Find"],
                    "field_bindings": {"value": "value"},
                    "intent_confidence": 0.9,
                    "confidence_reason": (
                        "The complete question and inspected source both support "
                        "a raw value projection."
                    ),
                },
                "corrected-revision",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Extract the source column", "status": "completed"},
                        {"content": "Submit", "status": "completed"},
                    ]
                },
                "corrected-todos",
            ),
            _answer_response(columns=["value"], rows=[["1"]]),
        ],
    )

    result = DeepAgent(model=model).run(raw_record_task)

    assert result.succeeded
    rejected_step, rejected_call = _tool_call(result, "rejected-revision")
    assert rejected_step.ok is False
    assert rejected_call["ok"] is False
    _, corrected_call = _tool_call(result, "corrected-revision")
    assert corrected_call["ok"] is True
    revised_plan = corrected_call["result"]["update"]["analysis_plan"]
    assert revised_plan["intent_confidence"] == 0.9
    assert revised_plan["confidence_history"][-1]["previous"] == 0.65
    assert revised_plan["confidence_history"][-1]["new"] == 0.9


def test_analyze_plan_normalizes_safe_reversed_field_bindings(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return the observed value.",
                    "output_spec": "Return the source metric column for the requested value.",
                    "steps": ["Project metric", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "column_extract",
                    "target_fields": ["metric"],
                    "output_columns": ["metric"],
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Return"],
                    "field_bindings": {"value": "metric"},
                },
                "reversed-bindings-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "completed"}]},
                "reversed-bindings-todos",
            ),
            _answer_response(columns=["metric"], rows=[["1"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "reversed-bindings-plan")
    plan = plan_call["result"]["update"]["analysis_plan"]
    assert plan["request_contract"]["field_bindings"] == {"metric": "value"}
    assert plan["output_columns"] == ["metric"]
    assert plan["question_audit"]["coverage_complete"] is True


def test_repeated_analyze_plan_contract_errors_fail_fast(
    public_task: PublicTask,
) -> None:
    bad_plan = _tool_response(
        "analyze_plan",
        {
            "intent": "Return a fabricated field.",
            "output_spec": "Return value.",
            "steps": ["Submit"],
            "delegation_candidates": [],
            "operation_type": "column_extract",
            "target_fields": ["value"],
            "output_columns": ["value"],
            "requested_outputs": ["fabricated"],
            "request_mode_evidence": ["Return"],
            "field_bindings": {"value": "fabricated"},
        },
        "bad-plan-placeholder",
    )
    responses = []
    for index in range(8):
        message = bad_plan.model_copy(deep=True)
        message.tool_calls[0]["id"] = f"bad-plan-{index}"
        responses.append(message)
    model = ScriptedChatModel(auto_bootstrap=False, responses=responses)

    result = DeepAgent(
        model=model,
        config=DeepAgentConfig(max_steps=30),
    ).run(public_task)

    assert not result.succeeded
    assert result.failure_reason is not None
    assert "Repeated analyze_plan contract rejection" in result.failure_reason
    assert model.call_count == 4


def test_revise_plan_records_conflict_evidence_and_forces_new_todos(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return the observed value.",
                    "output_spec": "Return the value column.",
                    "steps": ["Inspect data", "Verify plan", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "column_extract",
                    "target_fields": ["value"],
                    "output_columns": ["value"],
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Return"],
                    "field_bindings": {"value": "value"},
                    "verification_questions": ["Confirm the source field exists."],
                },
                "initial-iterative-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Inspect data", "status": "in_progress"}]},
                "initial-iterative-todos",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/sample.csv"},
                "iterative-read",
            ),
            _tool_response(
                "revise_plan",
                {
                    "revision_reason": (
                        "The first execution pass verified the concrete source binding."
                    ),
                    "evidence": ["sample.csv contains the value column."],
                    "conflict_points": [
                        "The initial plan had not yet verified the source field."
                    ],
                    "question_evidence": ["value"],
                    "superseded_plan_reason": (
                        "The source binding was provisional before data inspection."
                    ),
                    "intent": "Return the observed value.",
                    "output_spec": "Return the verified value column.",
                    "steps": ["Project verified value", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "column_extract",
                    "target_fields": ["value"],
                    "output_columns": ["value"],
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Return"],
                    "field_bindings": {"value": "value"},
                    "verification_questions": [],
                },
                "iterative-revise",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit verified value", "status": "completed"}]},
                "revised-iterative-todos",
            ),
            _answer_response(columns=["value"], rows=[["1"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, revise_call = _tool_call(result, "iterative-revise")
    revision = revise_call["result"]["update"]["plan_revisions"][-1]
    assert revision["conflict_points"] == [
        "The initial plan had not yet verified the source field."
    ]
    assert revision["question_evidence"] == ["value"]
    assert revision["revised_plan"]["question_audit"]["coverage_complete"] is True
    tool_ids = [tool_call["tool_call_id"] for _, tool_call in _tool_calls(result)]
    assert tool_ids.index("revised-iterative-todos") > tool_ids.index("iterative-revise")


def test_raw_record_plan_rejects_unrequested_context_columns_and_null_filter(
    public_task: PublicTask,
) -> None:
    task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question="Find country GDP records for recent years.",
        ),
        assets=public_task.assets,
    )
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return tertiary-industry GDP records.",
                    "output_spec": "Return date, province, and GDP.",
                    "steps": ["Filter nulls", "Sort by date", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "column_extract",
                    "target_fields": ["Province", "EndDate", "ThirdIndustryGDP"],
                    "filters": ["ThirdIndustryGDP IS NOT NULL"],
                    "requested_outputs": ["GDP"],
                    "scope_evidence": ["country", "recent years"],
                    "request_mode_evidence": ["Find"],
                    "field_bindings": {
                        "Province": "country",
                        "EndDate": "recent years",
                        "ThirdIndustryGDP": "GDP",
                    },
                    "filter_evidence": ["records"],
                },
                "over-shaped-plan",
            ),
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return the requested raw GDP field.",
                    "output_spec": "Return only ThirdIndustryGDP in source order.",
                    "steps": [
                        "Project the source field",
                        "Return records preserving source row count, order, and NULL values",
                        "Submit",
                    ],
                    "delegation_candidates": [],
                    "operation_type": "column_extract",
                    "target_fields": ["ThirdIndustryGDP"],
                    "filters": [],
                    "preserve_raw_rows": True,
                    "output_columns": ["ThirdIndustryGDP"],
                    "requested_outputs": ["GDP"],
                    "scope_evidence": ["country", "recent years"],
                    "request_mode_evidence": ["Find"],
                    "field_bindings": {
                        "ThirdIndustryGDP": "GDP",
                    },
                },
                "exact-projection-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "completed"}]},
                "projection-todos",
            ),
            _answer_response(columns=["ThirdIndustryGDP"], rows=[[None], [2.0], [1.0]]),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    bad_step, bad_call = _tool_call(result, "over-shaped-plan")
    assert bad_step.ok is False
    assert bad_call["ok"] is False
    assert "Request contract rejected" in json.dumps(bad_call["result"])
    _, corrected_call = _tool_call(result, "exact-projection-plan")
    assert corrected_call["ok"] is True


def test_raw_source_projection_is_committed_without_another_model_call(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task_raw_projection"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "data.json").write_text(
        json.dumps(
            {
                "records": [
                    {"Province": "A", "Metric": None},
                    {"Province": "B", "Metric": 2.0},
                    {"Province": "C", "Metric": 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    task = PublicTask(
        record=TaskRecord(
            task_id="task_raw_projection",
            difficulty="easy",
            question="Show the Metric records.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return the raw Metric field.",
                    "output_spec": "One Metric column in source order.",
                    "steps": ["Project Metric", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "column_extract",
                    "target_tables": ["data.json"],
                    "target_fields": ["metric"],
                    "preserve_raw_rows": True,
                    "output_columns": ["metric"],
                    "requested_outputs": ["Metric"],
                    "request_mode_evidence": ["Show"],
                    "field_bindings": {"metric": "Metric"},
                },
                "source-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "completed"}]},
                "source-todos",
            ),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.to_dict() == {
        "columns": ["Metric"],
        "rows": [[None], [2.0], [1.0]],
    }
    assert model.call_count == 2


def test_comma_separated_fields_cannot_bypass_raw_projection_guard(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task_comma_fields"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "ed_grossdomesticproduct.json").write_text(
        json.dumps(
            {
                "records": [
                    {"EndDate": "2000-12-31", "ThirdIndustryGDP": None},
                    {"EndDate": "2001-12-31", "ThirdIndustryGDP": 2.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    task = PublicTask(
        record=TaskRecord(
            task_id="task_comma_fields",
            difficulty="easy",
            question="Find country GDP records for recent years.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return GDP records.",
                    "output_spec": "Return date and GDP.",
                    "steps": ["Project fields"],
                    "delegation_candidates": [],
                    "operation_type": "column_extract",
                    "target_tables": "ed_grossdomesticproduct",
                    "target_fields": "thirdindustrygdp, enddate",
                    "requested_outputs": ["GDP"],
                    "scope_evidence": ["recent years"],
                    "request_mode_evidence": ["Find"],
                    "field_bindings": {
                        "thirdindustrygdp": "GDP",
                        "enddate": "recent years",
                    },
                },
                "comma-fields-plan",
            ),
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return the requested GDP field.",
                    "output_spec": "One GDP column in source order.",
                    "steps": ["Project GDP"],
                    "delegation_candidates": [],
                    "operation_type": "column_extract",
                    "target_tables": "ed_grossdomesticproduct",
                    "target_fields": "thirdindustrygdp",
                    "requested_outputs": ["GDP"],
                    "scope_evidence": ["recent years"],
                    "request_mode_evidence": ["Find"],
                    "field_bindings": {
                        "thirdindustrygdp": "GDP",
                    },
                },
                "corrected-comma-fields-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Project GDP", "status": "completed"}]},
                "comma-fields-todos",
            ),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    rejected_step, rejected_call = _tool_call(result, "comma-fields-plan")
    assert rejected_step.ok is False
    assert rejected_call["ok"] is False
    assert "never to scope wording" in json.dumps(rejected_call["result"])
    assert result.answer is not None
    assert result.answer.to_dict() == {
        "columns": ["ThirdIndustryGDP"],
        "rows": [[None], [2.0]],
    }
    assert model.call_count == 3
    verification_step = next(
        step
        for step in result.steps
        if step.action == "plan_verification"
        and step.tool_call_id == "auto-plan-verification-source-projection"
    )
    assert verification_step.observation["checks"]["todos_completed"] is True
    assert verification_step.observation["checks"]["source_path"] == (
        "/context/ed_grossdomesticproduct.json"
    )
    auto_answer_step, auto_answer_call = _tool_call(result, "auto-source-projection")
    assert auto_answer_step.action == "answer"
    assert auto_answer_call["args"]["mode"] == "source_projection"
    assert auto_answer_call["args"]["source_path"] == (
        "/context/ed_grossdomesticproduct.json"
    )


def test_answer_path_accepts_object_rows(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_staged_answer"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "data.json").write_text('{"records": [{"Metric": 1}]}', encoding="utf-8")
    task = PublicTask(
        record=TaskRecord(
            task_id="task_staged_answer",
            difficulty="easy",
            question="Compute the Metric records.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return Metric.",
                    "output_spec": "One Metric column.",
                    "steps": ["Stage", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "compute",
                    "target_fields": ["Metric"],
                    "output_columns": ["Metric"],
                    "requested_outputs": ["Metric"],
                    "request_mode_evidence": ["Compute"],
                    "field_bindings": {"Metric": "Metric"},
                    "transformation_evidence": ["Compute"],
                },
                "staged-plan",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Stage", "status": "in_progress"},
                        {"content": "Submit", "status": "pending"},
                    ]
                },
                "staged-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "import json\n"
                        "with open('/scratch/result.json', 'w', encoding='utf-8') as f:\n"
                        "    json.dump({'columns': ['Metric'], "
                        "'rows': [{'Metric': None}, {'Metric': 2.0}]}, f)\n"
                    )
                },
                "stage-answer-file",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Stage", "status": "completed"},
                        {"content": "Submit", "status": "completed"},
                    ]
                },
                "staged-complete",
            ),
            _tool_response(
                "answer",
                {
                    "answer_path": "/scratch/result.json",
                    "columns": '["Metric"]',
                },
                "staged-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.to_dict() == {
        "columns": ["Metric"],
        "rows": [[None], [2.0]],
    }


def test_answer_json_is_committed_without_another_model_call(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_auto_staged_answer"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "metrics.json").write_text(
        json.dumps({"records": [{"Metric": None}, {"Metric": 2.0}]}),
        encoding="utf-8",
    )
    task = PublicTask(
        record=TaskRecord(
            task_id="task_auto_staged_answer",
            difficulty="easy",
            question="Compute the Metric records.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return Metric.",
                    "output_spec": "One Metric column.",
                    "steps": ["Stage the result"],
                    "delegation_candidates": [],
                    "operation_type": "compute",
                    "target_tables": ["metrics"],
                    "target_fields": ["Metric"],
                    "output_columns": ["Metric"],
                    "requested_outputs": ["Metric"],
                    "request_mode_evidence": ["Compute"],
                    "field_bindings": {"Metric": "Metric"},
                    "transformation_evidence": ["Compute"],
                },
                "auto-stage-plan",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Stage the result", "status": "in_progress"},
                    ]
                },
                "auto-stage-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "import json\n"
                        "with open('/scratch/answer.json', 'w', encoding='utf-8') as f:\n"
                        "    json.dump({'columns': ['metric'], "
                        "'rows': [[None], [2.0]]}, f)\n"
                    )
                },
                "auto-stage-file",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Stage the result", "status": "completed"},
                    ]
                },
                "auto-stage-complete",
            ),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.to_dict() == {
        "columns": ["Metric"],
        "rows": [[None], [2.0]],
    }
    assert model.call_count == 4
    verification_step = next(
        step
        for step in result.steps
        if step.action == "plan_verification"
        and step.tool_call_id == "auto-plan-verification-staged-answer"
    )
    assert verification_step.observation["checks"]["todos_completed"] is True
    assert verification_step.observation["checks"]["actual_columns"] == ["Metric"]
    auto_answer_step, auto_answer_call = _tool_call(result, "auto-staged-answer")
    assert auto_answer_step.action == "answer"
    assert auto_answer_call["args"]["mode"] == "staged_answer"


def test_auto_staged_answer_must_match_latest_plan(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_invalid_auto_staged_answer"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    task = PublicTask(
        record=TaskRecord(
            task_id="task_invalid_auto_staged_answer",
            difficulty="easy",
            question="Compute the Metric records.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return Metric.",
                    "output_spec": "One Metric column.",
                    "steps": ["Stage", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "compute",
                    "target_fields": ["Metric"],
                    "output_columns": ["Metric"],
                    "requested_outputs": ["Metric"],
                    "request_mode_evidence": ["Compute"],
                    "field_bindings": {"Metric": "Metric"},
                    "transformation_evidence": ["Compute"],
                },
                "invalid-stage-plan",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Stage", "status": "in_progress"},
                        {"content": "Submit", "status": "pending"},
                    ]
                },
                "invalid-stage-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "import json\n"
                        "with open('/scratch/answer.json', 'w', encoding='utf-8') as f:\n"
                        "    json.dump({'columns': ['Other'], 'rows': [[1]]}, f)\n"
                    )
                },
                "invalid-stage-file",
            ),
            _answer_response(
                columns=["Metric"],
                rows=[[2.0]],
                tool_call_id="correct-after-invalid-stage",
            ),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.to_dict() == {
        "columns": ["Metric"],
        "rows": [[2.0]],
    }
    assert model.call_count == 4


def test_answer_columns_must_match_latest_plan(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return only value.",
                    "output_spec": "One value column.",
                    "steps": ["Submit"],
                    "delegation_candidates": [],
                    "operation_type": "column_extract",
                    "target_fields": ["value"],
                    "output_columns": ["value"],
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Return"],
                    "field_bindings": {"value": "value"},
                },
                "answer-shape-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "completed"}]},
                "answer-shape-todos",
            ),
            _answer_response(
                columns=["name", "value"],
                rows=[["alpha", "1"]],
                tool_call_id="wrong-shape-answer",
            ),
            _answer_response(
                columns=["value"],
                rows=[["1"]],
                tool_call_id="correct-shape-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    wrong_step, wrong_call = _tool_call(result, "wrong-shape-answer")
    assert wrong_step.ok is False
    assert wrong_call["ok"] is False
    assert "latest plan" in json.dumps(wrong_call["result"])
    assert result.answer is not None
    assert result.answer.columns == ["value"]


def test_completed_todos_with_non_executable_plan_force_revise_before_answer(
    public_task: PublicTask,
) -> None:
    task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question="Summarize value by year.",
        ),
        assets=public_task.assets,
    )
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Summarize values by year, but the executable schema is not bound yet.",
                    "output_spec": "Return yearly value summary.",
                    "steps": ["Inspect schema", "Revise executable plan", "Submit"],
                    "delegation_candidates": [],
                    "intent_confidence": 0.85,
                    "confidence_reason": "The request intent is clear, but fields still need verification.",
                    "operation_type": "compute",
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Summarize"],
                    "grouping_evidence": ["year"],
                    "transformation_evidence": ["Summarize"],
                },
                "non-executable-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Inspect schema", "status": "in_progress"}]},
                "initial-todos",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/sample.csv"},
                "schema-evidence",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Inspect schema", "status": "completed"}]},
                "completed-before-revise",
            ),
            _tool_response(
                "revise_plan",
                {
                    "revision_reason": "sample.csv provides verified year and value fields.",
                    "evidence": ["sample.csv has columns year,value"],
                    "conflict_points": [
                        "The previous plan had request evidence but no executable output columns, group_by, or aggregation."
                    ],
                    "question_evidence": ["Summarize", "value", "year"],
                    "superseded_plan_reason": "The old plan was an intent contract only.",
                    "intent": "Aggregate values by year.",
                    "output_spec": "Return year and summarized value.",
                    "steps": ["Group by year", "Submit"],
                    "delegation_candidates": [],
                    "intent_confidence": 0.95,
                    "confidence_reason": "The verified file columns match the original request.",
                    "operation_type": "aggregate",
                    "target_tables": ["sample.csv"],
                    "target_fields": ["value"],
                    "group_by": ["year"],
                    "aggregation": "sum(value)",
                    "output_columns": ["year", "value"],
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Summarize"],
                    "field_bindings": {
                        "year": "year",
                        "value": "value",
                    },
                    "grouping_evidence": ["year"],
                    "transformation_evidence": ["Summarize"],
                },
                "forced-revise",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit revised result", "status": "completed"}]},
                "todos-after-forced-revise",
            ),
            _answer_response(columns=["year", "value"], rows=[["2020", "1"]]),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    _, revise_call = _tool_call(result, "forced-revise")
    assert revise_call["ok"] is True
    assert "revise_plan" in model.bound_tool_choices
    revise_index = model.bound_tool_choices.index("revise_plan")
    assert model.bound_tool_sets[revise_index] == {"revise_plan"}
    assert result.answer is not None
    assert result.answer.columns == ["year", "value"]


def test_revise_plan_cannot_replace_schema_locked_source_binding(
    public_task: PublicTask,
) -> None:
    task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question="Summarize value by year.",
        ),
        assets=public_task.assets,
    )
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Use the schema-defined source table and field.",
                    "output_spec": "Return yearly value summary.",
                    "steps": ["Check schema source", "Submit if executable"],
                    "delegation_candidates": [],
                    "intent_confidence": 0.9,
                    "confidence_reason": "The schema binding is the semantic baseline.",
                    "operation_type": "aggregate",
                    "target_tables": ["schema_scale_table"],
                    "target_fields": ["schema_value", "year"],
                    "filters": ["schema_value > 100"],
                    "group_by": ["year"],
                    "aggregation": "count(*)",
                    "output_columns": ["year", "count"],
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Summarize"],
                    "field_bindings": {"year": "year"},
                    "derived_field_bindings": {"count": "Summarize"},
                    "filter_evidence": ["value"],
                    "grouping_evidence": ["year"],
                },
                "schema-locked-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Check schema source", "status": "in_progress"}]},
                "schema-locked-todos",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/sample.csv"},
                "observed-source",
            ),
            _tool_response(
                "revise_plan",
                {
                    "revision_reason": (
                        "The schema table is unavailable, so switch to a different file."
                    ),
                    "evidence": [
                        "schema_scale_table is unavailable; sample.csv has other_value."
                    ],
                    "conflict_points": [
                        "The schema source cannot be found in the observed data."
                    ],
                    "question_evidence": ["Summarize", "value", "year"],
                    "superseded_plan_reason": "Attempting to replace the source binding.",
                    "intent": "Use a substitute file.",
                    "output_spec": "Return substitute yearly counts.",
                    "steps": ["Use substitute file", "Submit"],
                    "delegation_candidates": [],
                    "intent_confidence": 0.9,
                    "confidence_reason": "A substitute file has similar-looking columns.",
                    "operation_type": "aggregate",
                    "target_tables": ["sample.csv"],
                    "target_fields": ["other_value", "year"],
                    "filters": ["other_value > 100"],
                    "group_by": ["year"],
                    "aggregation": "count(*)",
                    "output_columns": ["year", "count"],
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Summarize"],
                    "field_bindings": {"year": "year"},
                    "derived_field_bindings": {"count": "Summarize"},
                    "filter_evidence": ["value"],
                    "grouping_evidence": ["year"],
                },
                "schema-rebinding-revise",
            ),
            _answer_response(
                columns=["year", "count"],
                rows=[["2020", 1]],
                tool_call_id="must-not-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert not result.succeeded
    assert result.failure_reason is not None
    assert "schema source binding rejected" in result.failure_reason
    _, revise_call = _tool_call(result, "schema-rebinding-revise")
    assert revise_call["ok"] is False
    assert revise_call["status"] == "error"
    assert model.call_count == 4


def test_aggregate_plan_with_explicit_question_cue_is_allowed(
    public_task: PublicTask,
) -> None:
    aggregate_task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question="Summarize value by year.",
        ),
        assets=public_task.assets,
    )
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Aggregate values by year.",
                    "output_spec": "Return yearly totals.",
                    "steps": ["Group by year", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "aggregate",
                    "target_tables": ["sample.csv"],
                    "target_fields": ["value"],
                    "group_by": ["year"],
                    "aggregation": "sum(value)",
                    "output_columns": ["year", "value"],
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Summarize"],
                    "field_bindings": {
                        "year": "year",
                        "value": "value",
                    },
                    "grouping_evidence": ["year"],
                    "transformation_evidence": ["Summarize"],
                },
                "aggregate-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit aggregate", "status": "completed"}]},
                "todos-complete",
            ),
            _answer_response(columns=["year", "value"], rows=[["2020", "1"]]),
        ],
    )

    result = DeepAgent(model=model).run(aggregate_task)

    assert result.succeeded
    _, aggregate_call = _tool_call(result, "aggregate-plan")
    assert aggregate_call["ok"] is True


def test_chinese_contract_allows_whitespace_normalized_evidence_and_requires_explicit_derived_binding(
    public_task: PublicTask,
) -> None:
    question = "\u7ba1\u7406\u57fa\u91d1\u89c4\u6a21\u8d85\u8fc7100\u4ebf\u7684\u57fa\u91d1\u7ecf\u7406\u6700\u9ad8\u5b66\u5386\u5206\u5e03\u60c5\u51b5"
    task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question=question,
        ),
        assets=public_task.assets,
    )
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Reject computed outputs placed in direct field bindings.",
                    "output_spec": "Invalid plan.",
                    "steps": ["Submit"],
                    "delegation_candidates": [],
                    "operation_type": "aggregation",
                    "target_fields": ["totalfundnv", "education", "personalcode"],
                    "filters": ["totalfundnv > 100"],
                    "group_by": ["education"],
                    "aggregation": "COUNT()",
                    "output_columns": ["education", "count"],
                    "requested_outputs": ["\u6700\u9ad8\u5b66\u5386"],
                    "request_mode_evidence": ["\u5206\u5e03\u60c5\u51b5"],
                    "field_bindings": {
                        "education": "\u6700\u9ad8\u5b66\u5386",
                        "count": "\u5206\u5e03\u60c5\u51b5",
                    },
                    "derived_field_bindings": {},
                    "scope_evidence": ["\u7ba1\u7406\u57fa\u91d1\u89c4\u6a21\u8d85\u8fc7 100 \u4ebf"],
                    "filter_evidence": ["\u8d85\u8fc7 100 \u4ebf"],
                    "grouping_evidence": ["\u6700\u9ad8\u5b66\u5386"],
                },
                "computed-output-as-field-plan",
            ),
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Use whitespace-normalized original evidence and explicit derived binding.",
                    "output_spec": "Return education groups and counts.",
                    "steps": ["Filter managers", "Group by education", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "aggregation",
                    "target_fields": ["totalfundnv", "education", "personalcode"],
                    "filters": ["totalfundnv > 100"],
                    "group_by": ["education"],
                    "aggregation": "COUNT()",
                    "output_columns": ["education", "count"],
                    "requested_outputs": ["\u6700\u9ad8\u5b66\u5386"],
                    "request_mode_evidence": ["\u5206\u5e03\u60c5\u51b5"],
                    "field_bindings": {"education": "\u6700\u9ad8\u5b66\u5386"},
                    "derived_field_bindings": {"count": "\u5206\u5e03\u60c5\u51b5"},
                    "scope_evidence": ["\u7ba1\u7406\u57fa\u91d1\u89c4\u6a21\u8d85\u8fc7 100 \u4ebf"],
                    "filter_evidence": ["\u8d85\u8fc7 100 \u4ebf"],
                    "grouping_evidence": ["\u6700\u9ad8\u5b66\u5386"],
                },
                "spaced-derived-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "completed"}]},
                "exact-derived-todos",
            ),
            _answer_response(columns=["education", "count"], rows=[["master", 1]]),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    rejected_step, rejected_call = _tool_call(result, "computed-output-as-field-plan")
    assert rejected_step.ok is False
    assert rejected_call["ok"] is False
    assert "derived_field_bindings" in json.dumps(rejected_call["result"])
    _, plan_call = _tool_call(result, "spaced-derived-plan")
    assert plan_call["ok"] is True
    plan = plan_call["result"]["update"]["analysis_plan"]
    assert plan["operation_type"] == "aggregate"
    assert plan["request_contract"]["scope_evidence"] == [
        "\u7ba1\u7406\u57fa\u91d1\u89c4\u6a21\u8d85\u8fc7100\u4ebf"
    ]
    assert plan["request_contract"]["filter_evidence"] == ["\u8d85\u8fc7100\u4ebf"]
    assert plan["request_contract"]["derived_field_bindings"] == {
        "count": "\u5206\u5e03\u60c5\u51b5"
    }
    assert all(
        "\u8d85\u8fc7 100 \u4ebf" not in json.dumps(segment, ensure_ascii=False)
        for segment in plan["question_audit"]["segments"]
    )

def test_rank_plan_with_highest_cue_is_allowed(public_task: PublicTask) -> None:
    rank_task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question="Which company has the highest commission fees?",
        ),
        assets=public_task.assets,
    )
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Find the row with the highest commission fees.",
                    "output_spec": "Return the company and fee.",
                    "steps": ["Rank by fee", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "rank",
                    "target_tables": ["sample.csv"],
                    "target_fields": ["name", "value"],
                    "aggregation": "max(value)",
                    "output_columns": ["name", "value"],
                    "requested_outputs": ["company", "commission fees"],
                    "request_mode_evidence": ["Which"],
                    "field_bindings": {
                        "name": "company",
                        "value": "commission fees",
                    },
                    "transformation_evidence": ["highest"],
                },
                "rank-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit ranked result", "status": "completed"}]},
                "todos-complete",
            ),
            _answer_response(columns=["name", "value"], rows=[["alpha", "1"]]),
        ],
    )

    result = DeepAgent(model=model).run(rank_task)

    assert result.succeeded
    _, rank_call = _tool_call(result, "rank-plan")
    assert rank_call["ok"] is True


def test_statistics_request_with_data_word_is_not_rejected(
    public_task: PublicTask,
) -> None:
    statistics_task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question="Compute max and min values.",
        ),
        assets=public_task.assets,
    )
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Compute maximum and minimum values.",
                    "output_spec": "Return max and min.",
                    "steps": ["Compute extrema", "Submit"],
                    "delegation_candidates": [],
                    "operation_type": "aggregate",
                    "target_tables": ["sample.csv"],
                    "target_fields": ["value"],
                    "aggregation": "max(value), min(value)",
                    "output_columns": ["max", "min"],
                    "requested_outputs": ["max", "min"],
                    "request_mode_evidence": ["Compute"],
                    "field_bindings": {
                        "max": "max",
                        "min": "min",
                    },
                    "transformation_evidence": ["Compute"],
                },
                "statistics-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit statistics", "status": "completed"}]},
                "todos-complete",
            ),
            _answer_response(columns=["max", "min"], rows=[["1", "1"]]),
        ],
    )

    result = DeepAgent(model=model).run(statistics_task)

    assert result.succeeded
    _, statistics_call = _tool_call(result, "statistics-plan")
    assert statistics_call["ok"] is True


def test_completed_todos_force_answer_tool_only(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return the observed value.",
                    "output_spec": "One-column table containing value.",
                    "steps": ["Submit value"],
                    "delegation_candidates": [],
                    "operation_type": "column_extract",
                },
                "plan-call",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit value", "status": "completed"}]},
                "todos-complete",
            ),
            _answer_response(columns=["value"], rows=[["1"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.bound_tool_sets[2] == {"answer"}
    assert model.bound_tool_choices[2] == "answer"


def test_can_revise_plan_after_data_exploration(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "read_file",
                {"file_path": "/context/sample.csv"},
                "read-before-revise",
            ),
            _tool_response(
                "revise_plan",
                {
                    "revision_reason": "The file shows raw rows, so the answer should extract records.",
                    "evidence": ["sample.csv has one observed row with columns name,value"],
                    "intent": "Return the observed raw value.",
                    "output_spec": "One-column table containing the value field.",
                    "operation_type": "column_extract",
                    "target_tables": ["sample.csv"],
                    "target_fields": ["value"],
                    "filters": [],
                    "group_by": [],
                    "aggregation": None,
                    "preserve_raw_rows": True,
                    "output_columns": ["value"],
                    "requested_outputs": ["value"],
                    "request_mode_evidence": ["Return"],
                    "field_bindings": {"value": "value"},
                    "ambiguities": [],
                    "steps": ["Use the raw value column", "Submit"],
                    "delegation_candidates": [],
                },
                "revise-call",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Use the raw value column", "status": "in_progress"},
                        {"content": "Submit", "status": "pending"},
                    ]
                },
                "todos-after-revise",
            ),
            _answer_response(columns=["value"], rows=[["1"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, revise_call = _tool_call(result, "revise-call")
    assert revise_call["name"] == "revise_plan"
    assert "raw rows" in json.dumps(revise_call["args"])
    _, todos_call = _tool_call(result, "todos-after-revise")
    assert todos_call["name"] == "write_todos"
    assert result.answer is not None
    assert result.answer.columns == ["value"]


def test_parallel_tool_calls_are_correlated_by_id(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="Inspect both the directory and the target file.",
                tool_calls=[
                    {
                        "name": "ls",
                        "args": {"path": "/context"},
                        "id": "parallel-ls",
                        "type": "tool_call",
                    },
                    {
                        "name": "read_file",
                        "args": {"file_path": "/context/data.txt"},
                        "id": "parallel-read",
                        "type": "tool_call",
                    },
                ],
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    parallel_step, parallel_ls = _tool_call(result, "parallel-ls")
    same_step, parallel_read = _tool_call(result, "parallel-read")
    assert parallel_step is same_step
    assert "data.txt" in json.dumps(parallel_ls["result"])
    assert "hello from context" in json.dumps(parallel_read["result"])
    assert parallel_ls["name"] == "ls"
    assert parallel_read["name"] == "read_file"


def test_invalid_answer_returns_tool_error_and_can_be_corrected(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        responses=[
            _answer_response(columns=[], rows=[], tool_call_id="invalid-answer"),
            _answer_response(columns=["value"], rows=[["correct"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.call_count == 4
    invalid_step, invalid_call = _tool_call(result, "invalid-answer")
    assert invalid_call["name"] == "answer"
    assert invalid_call["ok"] is False
    assert invalid_step.ok is False
    assert "non-empty" in json.dumps(invalid_call["result"])
    assert result.steps[-1].ok is True


def test_empty_answer_rows_are_rejected(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _answer_response(columns=["value"], rows=[], tool_call_id="empty-rows"),
            _answer_response(columns=["value"], rows=[["correct"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    empty_step, empty_call = _tool_call(result, "empty-rows")
    assert empty_step.ok is False
    assert empty_call["ok"] is False
    assert "at least one row" in json.dumps(empty_call["result"])


def test_successful_answer_stops_before_another_model_call(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _answer_response(columns=["value"], rows=[["done"]]),
            AIMessage(content="This response must never be used."),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.call_count == 3


def test_plain_text_completion_retries_with_answer_tool(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="The answer is one."),
            _answer_response(columns=["value"], rows=[["one"]], tool_call_id="retried-answer"),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["one"]]
    _, answer_call = _tool_call(result, "retried-answer")
    assert answer_call["name"] == "answer"
    assert model.bound_tool_sets[-1] == {"answer"}
    assert model.bound_tool_choices[-1] == "answer"


def test_plain_text_before_plan_does_not_skip_planning(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        auto_bootstrap=False,
        responses=[AIMessage(content="The answer is one.")],
    )

    result = DeepAgent(model=model).run(public_task)

    assert not result.succeeded
    assert result.answer is None
    assert model.bound_tool_sets == [{"analyze_plan"}]
    assert result.failure_reason == "Agent completed without calling the answer tool."


def test_default_subagent_does_not_receive_answer_tool(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "task",
                {
                    "description": "Inspect the context and return a short report.",
                    "subagent_type": "general-purpose",
                },
                "task-call",
            ),
            AIMessage(content="Subagent report."),
            _answer_response(columns=["value"], rows=[["from report"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert any(
        "answer" in tool_names and "task" in tool_names for tool_names in model.bound_tool_sets
    )
    assert any(
        "answer" not in tool_names and "task" not in tool_names
        for tool_names in model.bound_tool_sets
    )
    _, task_call = _tool_call(result, "task-call")
    assert "Subagent report." in json.dumps(task_call["result"])
    subagent_steps = [
        step
        for step in result.steps
        if step.action_input.get("scope") == "subagent:general-purpose"
    ]
    assert any(step.action == "system_prompt" for step in subagent_steps)
    assert any("llm" in step.action_input for step in subagent_steps)
    assert not any("request" in step.action_input for step in subagent_steps)


def test_model_call_limit_returns_failure(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[_tool_response("ls", {"path": "/context"}, f"ls-{index}") for index in range(2)]
    )

    result = DeepAgent(
        model=model,
        config=DeepAgentConfig(max_steps=2),
    ).run(public_task)

    assert not result.succeeded
    assert model.call_count == 2
    assert result.failure_reason == "Agent did not submit an answer within 2 model calls."


def test_inline_python_execution_is_isolated(
    public_task: PublicTask,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "must-not-reach-shell")
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "import os\n"
                        "from pathlib import Path\n"
                        "Path('context/data.txt').write_text('changed', encoding='utf-8')\n"
                        "print(os.getenv('API_KEY'))\n"
                    ),
                },
                "run-code",
            ),
            _answer_response(columns=["value"], rows=[["isolated"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert public_task.context_dir.joinpath("data.txt").read_text(encoding="utf-8") == (
        "hello from context\n"
    )
    _, environment_call = _tool_call(result, "run-code")
    assert "must-not-reach-shell" not in json.dumps(environment_call["result"])
    assert "None" in json.dumps(environment_call["result"])


def test_python_virtual_context_paths_are_mapped(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "from pathlib import Path\n"
                        "print(Path('/context/data.txt').read_text(encoding='utf-8'))\n"
                        "print([path.name for path in Path('/context').iterdir()])\n"
                    )
                },
                "virtual-paths",
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, virtual_path_call = _tool_call(result, "virtual-paths")
    output = json.dumps(virtual_path_call["result"], ensure_ascii=False)
    assert virtual_path_call["ok"] is True
    assert "hello from context" in output
    assert "sample.csv" in output


def test_python_virtual_scratch_path_is_mapped(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "from pathlib import Path\n"
                        "target = Path('/scratch/result.txt')\n"
                        "target.write_text('瀹屾垚', encoding='utf-8')\n"
                        "print(target.read_text(encoding='utf-8'))\n"
                    )
                },
                "virtual-scratch",
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, virtual_scratch_call = _tool_call(result, "virtual-scratch")
    output = json.dumps(virtual_scratch_call["result"], ensure_ascii=False)
    assert virtual_scratch_call["ok"] is True
    assert "瀹屾垚" in output


def test_empty_python_source_is_rejected(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {"code": " \n"},
                "invalid-code",
            ),
            _answer_response(columns=["value"], rows=[["recovered"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    invalid_step, invalid_call = _tool_call(result, "invalid-code")
    assert invalid_step.ok is False
    assert invalid_call["ok"] is False
    assert "non-empty" in json.dumps(invalid_call["result"])


def test_python_nonzero_exit_is_a_tool_error(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {"code": "raise RuntimeError('expected failure')\n"},
                "run-failure",
            ),
            _answer_response(columns=["value"], rows=[["recovered"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    failed_step, failed_call = _tool_call(result, "run-failure")
    assert failed_step.ok is False
    assert failed_call["ok"] is False
    assert "Exit code: 1" in json.dumps(failed_call["result"])
    assert "expected failure" in json.dumps(failed_call["result"])


def test_unavailable_tools_are_hidden_from_main_and_subagent(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "task",
                {
                    "description": "Inspect the context.",
                    "subagent_type": "general-purpose",
                },
                "task-call",
            ),
            AIMessage(content="Subagent report."),
            _answer_response(columns=["value"], rows=[["done"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.bound_tool_sets
    assert model.bound_tool_sets[0] == {"analyze_plan"}
    assert model.bound_tool_sets[1] == {"write_todos"}
    for tool_names in model.bound_tool_sets[2:]:
        assert {"execute", "ls", "write_file", "edit_file"}.isdisjoint(tool_names)
    assert all("execute_python" in tool_names for tool_names in model.bound_tool_sets[2:])


def test_python_output_is_utf8(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {"code": "print('鍖椾含 涓婃捣 鍏ㄥ浗')\n"},
                "unicode-output",
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, output_call = _tool_call(result, "unicode-output")
    output = json.dumps(output_call["result"], ensure_ascii=False)
    assert "鍖椾含 涓婃捣 鍏ㄥ浗" in output
    assert "\ufffd" not in output


def test_builtin_write_is_denied_for_context(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "write_file",
                {"file_path": "/context/new.txt", "content": "blocked"},
                "blocked-write",
            ),
            _answer_response(columns=["value"], rows=[["unchanged"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    blocked_step, blocked_call = _tool_call(result, "blocked-write")
    assert blocked_step.ok is False
    assert blocked_call["ok"] is False
    assert "permission" in json.dumps(blocked_call["result"]).lower()
    assert not public_task.context_dir.joinpath("new.txt").exists()
