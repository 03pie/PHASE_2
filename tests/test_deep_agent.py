from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from data_agent_baseline.agents.deep_agent import DeepAgent
from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.analyze_plan import analyze_plan_tool


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]
    auto_discovery_plan: bool = True
    call_count: int = 0
    bound_tool_sets: list[set[str]] = Field(default_factory=list)
    bound_tool_choices: list[str | None] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling-model"

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if not self.auto_discovery_plan:
            return
        discovery_and_plan = [
            _tool_response(
                "read_file",
                {"file_path": "/context/knowledge.md"},
                "knowledge-call",
                content="Read the authoritative task knowledge first.",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/data.txt"},
                "schema-call",
                content="Inspect a relevant candidate data source.",
            ),
            AIMessage(
                content="Create an evidence-based execution plan.",
                tool_calls=[
                    {
                        "name": "analyze_plan",
                        "args": {
                            "intent": "Solve the requested benchmark task.",
                            "output_spec": "Return the requested result as a table.",
                            "knowledge_status": "authoritative",
                            "knowledge_findings": [
                                "knowledge.md defines the authoritative semantics.",
                            ],
                            "knowledge_issue": "",
                            "context_sources": ["/context/data.txt"],
                            "context_evidence": [
                                "data.txt contains the candidate source values.",
                            ],
                            "cross_validated_inference": "",
                            "uncertainties": [],
                            "steps": ["Compute and validate", "Submit"],
                            "delegation_candidates": [],
                        },
                        "id": "plan-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="Convert the evidence-based plan into actionable todos.",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {
                            "todos": [
                                {"content": "Compute and validate", "status": "in_progress"},
                                {"content": "Submit the result", "status": "pending"},
                            ]
                        },
                        "id": "todos-call",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
        self.responses[0:0] = discovery_and_plan

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


def _tool_response(
    name: str,
    args: dict[str, Any],
    tool_call_id: str,
    *,
    content: str = "",
) -> AIMessage:
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
        if "request" in step.action_input
    ]


def _tool_calls(result: Any) -> list[tuple[Any, dict[str, Any]]]:
    return [
        (step, tool_call)
        for step in _llm_steps(result)
        for tool_call in step.observation.get("tool_calls", [])
    ]


def _tool_call(result: Any, tool_call_id: str) -> tuple[Any, dict[str, Any]]:
    return next(
        (step, tool_call)
        for step, tool_call in _tool_calls(result)
        if tool_call.get("tool_call_id") == tool_call_id
    )


@pytest.fixture
def public_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "data.txt").write_text("hello from context\n", encoding="utf-8")
    (context_dir / "knowledge.md").write_text(
        "Use the observed source value exactly.\n",
        encoding="utf-8",
    )
    (context_dir / "sample.csv").write_text("name,value\nalpha,1\n", encoding="utf-8")
    (context_dir / "sample.json").write_text('{"value": 1}\n', encoding="utf-8")
    with closing(sqlite3.connect(context_dir / "sample.sqlite")) as connection:
        connection.execute("CREATE TABLE metrics (name TEXT, value REAL)")
        connection.execute("CREATE TABLE observations (observed_at TEXT)")
        connection.commit()
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
    assert [step.action for step in result.steps[:4]] == [
        "system_prompt",
        "user_prompt",
        "read_file",
        "read_file",
    ]
    assert result.steps[4].action == "analyze_plan"
    tool_calls = _tool_calls(result)
    assert [tool_call["name"] for _, tool_call in tool_calls] == [
        "read_file",
        "read_file",
        "analyze_plan",
        "write_todos",
        "read_file",
        "answer",
    ]
    assert [tool_call["tool_call_id"] for _, tool_call in tool_calls] == [
        "knowledge-call",
        "schema-call",
        "plan-call",
        "todos-call",
        "read-call",
        "answer-call",
    ]
    assert [step.action_input["llm_call_index"] for step, _ in tool_calls] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    system_prompt_steps = [
        step
        for step in result.steps
        if step.action == "system_prompt"
        and step.action_input.get("scope") == "main"
    ]
    assert len(system_prompt_steps) == 1
    assert all(
        "system_message" not in step.action_input["request"]
        for step in _llm_steps(result)
    )
    user_prompt = next(step for step in result.steps if step.action == "user_prompt")
    prompt_content = user_prompt.action_input["message"]["content"]
    assert "sample.sqlite" in prompt_content
    assert "[tables: metrics, observations]" in prompt_content


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


def test_analyze_plan_is_unavailable_before_discovery(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        responses=[_answer_response(columns=["value"], rows=[["done"]])]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert "analyze_plan" not in model.bound_tool_sets[0]
    assert "analyze_plan" not in model.bound_tool_sets[1]
    assert "analyze_plan" in model.bound_tool_sets[2]
    assert {"execute_python", "grep", "read_file"}.issubset(
        model.bound_tool_sets[2]
    )
    assert model.bound_tool_choices[2] is None
    _, plan_call = _tool_call(result, "plan-call")
    assert plan_call["args"]["knowledge_status"] == "authoritative"
    assert plan_call["args"]["knowledge_findings"] == [
        "knowledge.md defines the authoritative semantics.",
    ]
    assert plan_call["args"]["context_sources"] == ["/context/data.txt"]
    assert plan_call["args"]["context_evidence"] == [
        "data.txt contains the candidate source values.",
    ]


def test_authoritative_knowledge_rejects_inferred_override() -> None:
    result = analyze_plan_tool.func(
        intent="Return the requested metric.",
        output_spec="One result table.",
        knowledge_status="authoritative",
        knowledge_findings=["knowledge.md defines the metric and unit."],
        knowledge_issue="",
        context_sources=["/context/data.csv"],
        context_evidence=["data.csv contains the required field."],
        cross_validated_inference="Use a different unit inferred from data.csv.",
        uncertainties=[],
        steps=["Read values", "Validate", "Submit"],
        delegation_candidates=[],
        tool_call_id="invalid-authoritative-plan",
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "cannot be replaced" in str(result.content)


def test_invalid_knowledge_requires_cross_source_validation() -> None:
    result = analyze_plan_tool.func(
        intent="Return the requested metric.",
        output_spec="One result table.",
        knowledge_status="invalid",
        knowledge_findings=[],
        knowledge_issue="knowledge.md conflicts with the observed schema.",
        context_sources=["/context/data.csv"],
        context_evidence=[
            "data.csv exposes field_a.",
            "A second observation from the same data.csv supports field_a.",
        ],
        cross_validated_inference="Use field_a as the requested measure.",
        uncertainties=[],
        steps=["Read values", "Validate", "Submit"],
        delegation_candidates=[],
        tool_call_id="single-source-plan",
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "at least two distinct" in str(result.content)


def test_invalid_knowledge_reopens_only_required_cross_validation(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_file",
                {"file_path": "/context/knowledge.md"},
                "knowledge-check",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/data.txt"},
                "first-source",
            ),
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Resolve a schema conflict.",
                    "output_spec": "Return one table.",
                    "knowledge_status": "invalid",
                    "knowledge_findings": [],
                    "knowledge_issue": "The knowledge rule conflicts with the schema.",
                    "context_sources": ["/context/data.txt"],
                    "context_evidence": ["data.txt exposes the first observation."],
                    "cross_validated_inference": "Use the observed source field.",
                    "uncertainties": [],
                    "steps": ["Validate", "Submit"],
                    "delegation_candidates": [],
                },
                "invalid-plan",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/sample.csv"},
                "second-source",
            ),
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Resolve a schema conflict.",
                    "output_spec": "Return one table.",
                    "knowledge_status": "invalid",
                    "knowledge_findings": [],
                    "knowledge_issue": "The knowledge rule conflicts with the schema.",
                    "context_sources": [
                        "/context/data.txt",
                        "/context/sample.csv",
                    ],
                    "context_evidence": [
                        "data.txt exposes the first observation.",
                        "sample.csv independently confirms the value field.",
                    ],
                    "cross_validated_inference": "Use the shared observed value.",
                    "uncertainties": [],
                    "steps": ["Validate", "Submit"],
                    "delegation_candidates": [],
                },
                "valid-cross-plan",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Validate", "status": "in_progress"},
                        {"content": "Submit", "status": "pending"},
                    ]
                },
                "cross-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.bound_tool_sets[:6] == [
        {"read_file"},
        {"execute_python", "grep", "read_file"},
        {"analyze_plan", "execute_python", "grep", "read_file"},
        {"execute_python", "grep", "read_file"},
        {"analyze_plan"},
        {"write_todos"},
    ]
    _, invalid_plan = _tool_call(result, "invalid-plan")
    assert invalid_plan["name"] == "analyze_plan"
    assert model.bound_tool_sets[3] == {"execute_python", "grep", "read_file"}


def test_discovery_budget_forces_plan_after_targeted_checks(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_file",
                {"file_path": "/context/knowledge.md"},
                "budget-knowledge",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/data.txt"},
                "budget-source-1",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/sample.csv"},
                "budget-source-2",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/sample.json"},
                "budget-source-3",
            ),
            _tool_response(
                "analyze_plan",
                {
                    "intent": "Return the observed value.",
                    "output_spec": "Return one table.",
                    "knowledge_status": "authoritative",
                    "knowledge_findings": ["knowledge.md defines the output rule."],
                    "knowledge_issue": "",
                    "context_sources": [
                        "/context/data.txt",
                        "/context/sample.csv",
                        "/context/sample.json",
                    ],
                    "context_evidence": [
                        "data.txt contains the direct value.",
                        "sample.csv confirms the value field.",
                        "sample.json confirms the value representation.",
                    ],
                    "cross_validated_inference": "",
                    "uncertainties": [],
                    "steps": ["Validate", "Submit"],
                    "delegation_candidates": [],
                },
                "budget-plan",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Validate", "status": "in_progress"},
                        {"content": "Submit", "status": "pending"},
                    ]
                },
                "budget-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.bound_tool_sets[2] == {
        "analyze_plan",
        "execute_python",
        "grep",
        "read_file",
    }
    assert model.bound_tool_sets[3] == model.bound_tool_sets[2]
    assert model.bound_tool_sets[4] == {"analyze_plan"}
    assert model.bound_tool_choices[4] == "analyze_plan"


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
    assert model.call_count == 6
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
    assert model.call_count == 5


def test_plain_text_completion_is_not_a_valid_answer(public_task: PublicTask) -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="The answer is one.")])

    result = DeepAgent(model=model).run(public_task)

    assert not result.succeeded
    assert result.answer is None
    assert result.failure_reason == "Agent completed without calling the answer tool."
    assert result.steps[-1].action == "llm_response"


def test_completed_todos_force_structured_answer(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "completed"},
                        {"content": "Submit the result", "status": "completed"},
                    ]
                },
                "completed-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.bound_tool_sets[-1] == {"answer"}
    assert model.bound_tool_choices[-1] == "answer"


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
    assert any("request" in step.action_input for step in subagent_steps)


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
                        "target.write_text('完成', encoding='utf-8')\n"
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
    assert "完成" in output


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
    assert model.bound_tool_sets[0] == {"read_file"}
    assert model.bound_tool_choices[0] == "read_file"
    assert model.bound_tool_sets[1] == {"execute_python", "grep", "read_file"}
    assert model.bound_tool_choices[1] is None
    assert model.bound_tool_sets[2] == {
        "analyze_plan",
        "execute_python",
        "grep",
        "read_file",
    }
    assert model.bound_tool_choices[2] is None
    assert model.bound_tool_sets[3] == {"write_todos"}
    assert model.bound_tool_choices[3] == "write_todos"
    for tool_names in model.bound_tool_sets[4:]:
        assert {"execute", "ls", "write_file", "edit_file"}.isdisjoint(tool_names)
    assert all("execute_python" in tool_names for tool_names in model.bound_tool_sets[4:])


def test_python_output_is_utf8(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {"code": "print('北京 上海 全国')\n"},
                "unicode-output",
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, output_call = _tool_call(result, "unicode-output")
    output = json.dumps(output_call["result"], ensure_ascii=False)
    assert "北京 上海 全国" in output
    assert "�" not in output


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
