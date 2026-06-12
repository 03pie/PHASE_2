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
from data_agent_baseline.agents.middleware import _canonical_knowledge_quote
from data_agent_baseline.agents.middleware import _discovery_state
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.analyze_plan import analyze_plan_tool
from data_agent_baseline.tools.execute_python import create_execute_python_tool


def _plan_args(
    *,
    requirement_quote: str = "Return the observed value.",
    requirement_type: str = "measure",
    knowledge_status: str = "authoritative",
    knowledge_quote: str = "Use the observed source value exactly.",
    context_paths: list[str] | None = None,
    columns: list[tuple[str, list[str]]] | None = None,
    transformations: list[dict[str, Any]] | None = None,
    expected_row_count: int | None = None,
    steps: list[str] | None = None,
    version: int = 1,
    changed_fields: list[str] | None = None,
    evidence_changes: list[str] | None = None,
) -> dict[str, Any]:
    paths = context_paths or ["/context/data.txt"]
    plan_steps = steps or ["Compute and validate", "Submit the result"]
    plan_transformations = transformations or []
    authoritative = knowledge_status == "authoritative"
    return {
        "schema_version": "1.0",
        "intent": {
            "requirements": [
                {
                    "statement": "Return the requested source value.",
                    "requirement_type": requirement_type,
                    "quote": requirement_quote,
                }
            ],
            "unresolved": [],
        },
        "output_spec": {
            "columns": [
                {"name": name, "source_fields": source_fields}
                for name, source_fields in (columns or [("value", ["value"])])
            ],
            "row_grain": "one source record",
            "row_policy": "transform" if plan_transformations else "preserve",
            "transformations": plan_transformations,
            "ordering": "unspecified" if plan_transformations else "source",
            "sort_keys": [],
            "null_policy": "preserve",
            "expected_row_count": expected_row_count,
        },
        "evidence": {
            "knowledge_status": knowledge_status,
            "knowledge_rules": (
                [
                    {
                        "rule_type": "semantic",
                        "quote": knowledge_quote,
                        "source_path": "/context/knowledge.md",
                    }
                ]
                if authoritative
                else []
            ),
            "knowledge_issue": (
                "" if authoritative else "Knowledge conflicts with the observed schema."
            ),
            "context_sources": [
                {
                    "path": path,
                    "observations": [f"{path} contains the inspected source data."],
                }
                for path in paths
            ],
            "cross_validated_inference": (
                "" if authoritative else "Use the field shared by the inspected sources."
            ),
        },
        "revision": {
            "version": version,
            "reason": "Initial evidence-based plan." if version == 1 else "Evidence changed.",
            "evidence_changes": evidence_changes or [],
            "changed_fields": changed_fields or [],
        },
        "steps": plan_steps,
        "delegation_candidates": [],
    }


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]
    auto_discovery_plan: bool = True
    request_quote: str = "Return the observed value."
    knowledge_quote: str = "Use the observed source value exactly."
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
                        "args": _plan_args(
                            requirement_quote=self.request_quote,
                            knowledge_quote=self.knowledge_quote,
                        ),
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
    payload = json.dumps(
        {"columns": columns, "rows": rows},
        ensure_ascii=False,
    )
    return _tool_response(
        "execute_python",
        {
            "code": (
                "import json\n"
                f"payload = json.loads({payload!r})\n"
                "set_answer(payload['columns'], payload['rows'])\n"
            )
        },
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
        "execute_python",
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
    assert plan_call["args"]["evidence"]["knowledge_status"] == "authoritative"
    assert plan_call["args"]["evidence"]["knowledge_rules"] == [
        {
            "rule_type": "semantic",
            "quote": "Use the observed source value exactly.",
            "source_path": "/context/knowledge.md",
        }
    ]
    assert [
        source["path"] for source in plan_call["args"]["evidence"]["context_sources"]
    ] == ["/context/data.txt"]
    assert plan_call["args"]["output_spec"]["row_policy"] == "preserve"
    assert plan_call["args"]["output_spec"]["transformations"] == []


def test_authoritative_knowledge_rejects_inferred_override() -> None:
    arguments = _plan_args()
    arguments["evidence"]["cross_validated_inference"] = (
        "Use a different unit inferred from data.csv."
    )
    result = analyze_plan_tool.func(
        **arguments,
        original_request="Return the observed value.",
        tool_call_id="invalid-authoritative-plan",
    )

    assert not isinstance(result, ToolMessage)
    assert result.update["analysis_plan"]["evidence"]["knowledge_issue"] == ""
    assert (
        result.update["analysis_plan"]["evidence"]["cross_validated_inference"]
        == ""
    )


def test_invalid_knowledge_requires_cross_source_validation() -> None:
    arguments = _plan_args(
        knowledge_status="invalid",
        context_paths=["/context/data.csv"],
    )
    result = analyze_plan_tool.func(
        **arguments,
        original_request="Return the observed value.",
        tool_call_id="single-source-plan",
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "at least two distinct" in str(result.content)


def test_plan_rejects_quote_not_present_in_original_request(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_file",
                {"file_path": "/context/knowledge.md"},
                "quote-knowledge",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/data.txt"},
                "quote-source",
            ),
            _tool_response(
                "analyze_plan",
                _plan_args(requirement_quote="Invented user requirement."),
                "invalid-user-quote",
            ),
            _tool_response(
                "analyze_plan",
                _plan_args(),
                "valid-user-quote",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "quote-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, valid_plan = _tool_call(result, "valid-user-quote")
    assert valid_plan["ok"] is True


def test_markdown_knowledge_quote_is_canonicalized(
    public_task: PublicTask,
) -> None:
    public_task.context_dir.joinpath("knowledge.md").write_text(
        "| `value` | Use the observed source value exactly. |\n",
        encoding="utf-8",
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_file",
                {"file_path": "/context/knowledge.md"},
                "markdown-knowledge",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/data.txt"},
                "markdown-source",
            ),
            _tool_response(
                "analyze_plan",
                _plan_args(
                    knowledge_quote=(
                        "value: Use the observed source value exactly."
                    )
                ),
                "markdown-plan",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "markdown-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    plan_message = next(
        message
        for message in result.steps
        if message.action == "analyze_plan"
    )
    plan_content = json.dumps(plan_message.observation, ensure_ascii=False)
    assert "| `value` | Use the observed source value exactly. |" in plan_content


def test_empty_knowledge_content_does_not_break_quote_canonicalization() -> None:
    assert _canonical_knowledge_quote("missing quote", None) is None


def test_python_knowledge_read_is_available_for_quote_validation() -> None:
    messages: list[BaseMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute_python",
                    "args": {
                        "code": (
                            "print(open('/context/knowledge.md', "
                            "encoding='utf-8').read())"
                        )
                    },
                    "id": "python-knowledge",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="| `value` | Exact knowledge rule. |",
            name="execute_python",
            tool_call_id="python-knowledge",
            status="success",
        ),
    ]

    discovery = _discovery_state(messages)

    assert discovery.knowledge_available
    assert "| `value` | Exact knowledge rule. |" in discovery.knowledge_content


def test_invalid_plan_tool_json_is_retried(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_file",
                {"file_path": "/context/knowledge.md"},
                "retry-knowledge",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/data.txt"},
                "retry-source",
            ),
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "analyze_plan",
                        "args": '{"intent":',
                        "id": "malformed-plan",
                        "error": "invalid JSON",
                        "type": "invalid_tool_call",
                    }
                ],
            ),
            _tool_response("analyze_plan", _plan_args(), "retried-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "retry-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.bound_tool_choices[3] == "analyze_plan"
    _, retried_plan = _tool_call(result, "retried-plan")
    assert retried_plan["ok"] is True


def test_semantic_knowledge_cannot_authorize_aggregation(
    public_task: PublicTask,
) -> None:
    invalid_plan = _plan_args(
        transformations=[
            {
                "operation": "aggregate",
                "description": "Sum all source rows.",
                "authorization": {
                    "source": "knowledge",
                    "quote": "Use the observed source value exactly.",
                },
            }
        ]
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_file",
                {"file_path": "/context/knowledge.md"},
                "semantic-knowledge",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/data.txt"},
                "semantic-source",
            ),
            _tool_response("analyze_plan", invalid_plan, "semantic-aggregate"),
            _tool_response("analyze_plan", _plan_args(), "preserve-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "semantic-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, preserve_plan = _tool_call(result, "preserve-plan")
    assert preserve_plan["ok"] is True


def test_generic_output_requirement_cannot_authorize_aggregation(
    public_task: PublicTask,
) -> None:
    invalid_plan = _plan_args(
        requirement_quote="Return the observed value.",
        transformations=[
            {
                "operation": "aggregate",
                "description": "Sum all source rows.",
                "authorization": {
                    "source": "user",
                    "quote": "Return the observed value.",
                },
            }
        ],
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_file",
                {"file_path": "/context/knowledge.md"},
                "output-knowledge",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/data.txt"},
                "output-source",
            ),
            _tool_response("analyze_plan", invalid_plan, "output-aggregate"),
            _tool_response("analyze_plan", _plan_args(), "output-preserve"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "output-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.bound_tool_sets[3] == {
        "analyze_plan",
        "execute_python",
        "grep",
        "read_file",
    }
    _, preserve_plan = _tool_call(result, "output-preserve")
    assert preserve_plan["ok"] is True


def test_explicit_user_requirement_can_authorize_transformation(
    public_task: PublicTask,
) -> None:
    task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question="Return the maximum observed value.",
        ),
        assets=public_task.assets,
    )
    plan = _plan_args(
        requirement_quote="maximum observed value",
        requirement_type="calculation",
        transformations=[
            {
                "operation": "aggregate",
                "description": "Return the maximum source value.",
                "authorization": {
                    "source": "user",
                    "quote": "maximum observed value",
                },
            }
        ],
        expected_row_count=1,
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_file",
                {"file_path": "/context/knowledge.md"},
                "maximum-knowledge",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/data.txt"},
                "maximum-source",
            ),
            _tool_response("analyze_plan", plan, "maximum-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "maximum-todos",
            ),
            _answer_response(columns=["value"], rows=[["maximum"]]),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded


def test_revision_cannot_rewrite_existing_user_requirement(
    public_task: PublicTask,
) -> None:
    revision = _plan_args(
        version=2,
        changed_fields=["intent"],
    )
    revision["intent"]["requirements"][0]["statement"] = (
        "Replace the original requirement with a new interpretation."
    )
    model = ScriptedChatModel(
        responses=[
            _tool_response("analyze_plan", revision, "invalid-revision"),
            _answer_response(columns=["value"], rows=[["original"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["original"]]


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
                _plan_args(
                    knowledge_status="invalid",
                    context_paths=["/context/data.txt"],
                    steps=["Validate", "Submit"],
                ),
                "invalid-plan",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/sample.csv"},
                "second-source",
            ),
            _tool_response(
                "analyze_plan",
                _plan_args(
                    knowledge_status="invalid",
                    context_paths=[
                        "/context/data.txt",
                        "/context/sample.csv",
                    ],
                    steps=["Validate", "Submit"],
                ),
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
        {"analyze_plan", "execute_python", "grep", "read_file"},
        {"write_todos"},
    ]
    assert model.bound_tool_choices[4] is None
    _, invalid_plan = _tool_call(result, "invalid-plan")
    assert invalid_plan["name"] == "analyze_plan"
    assert model.bound_tool_sets[3] == {"execute_python", "grep", "read_file"}


def test_discovery_remains_model_driven_after_targeted_checks(
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
                _plan_args(
                    context_paths=[
                        "/context/data.txt",
                        "/context/sample.csv",
                        "/context/sample.json",
                    ],
                    steps=["Validate", "Submit"],
                ),
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
    assert model.bound_tool_sets[4] == model.bound_tool_sets[2]
    assert model.bound_tool_choices[4] is None


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
    assert invalid_call["name"] == "execute_python"
    assert invalid_call["ok"] is False
    assert invalid_step.ok is False
    assert "exactly match" in json.dumps(invalid_call["result"])
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


def test_expected_answer_row_count_is_enforced(public_task: PublicTask) -> None:
    plan = _plan_args(expected_row_count=2)
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_file",
                {"file_path": "/context/knowledge.md"},
                "row-count-knowledge",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/data.txt"},
                "row-count-source",
            ),
            _tool_response("analyze_plan", plan, "row-count-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "row-count-todos",
            ),
            _answer_response(
                columns=["value"],
                rows=[["one"]],
                tool_call_id="wrong-row-count",
            ),
            _answer_response(
                columns=["value"],
                rows=[["one"], ["two"]],
                tool_call_id="correct-row-count",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    wrong_step, wrong_call = _tool_call(result, "wrong-row-count")
    assert wrong_step.ok is False
    assert wrong_call["ok"] is False
    assert "expected_row_count=2" in json.dumps(wrong_call["result"])


def test_task_2_preserves_source_rows_without_aggregation() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    task_dir = repository_root / "data" / "public" / "input" / "task_2"
    context_dir = task_dir / "context"
    task_record = json.loads(
        task_dir.joinpath("task.json").read_text(encoding="utf-8")
    )
    source_records = json.loads(
        context_dir.joinpath("json", "ed_grossdomesticproduct.json").read_text(
            encoding="utf-8"
        )
    )["records"]
    source_rows = [[record.get("ThirdIndustryGDP")] for record in source_records]
    steps = [
        "Project ThirdIndustryGDP while preserving source rows",
        "Validate 354 rows and submit",
    ]
    plan = _plan_args(
        requirement_quote="第三产业国内生产总值",
        knowledge_quote="GDP contribution from the tertiary (services) sector",
        context_paths=["/context/json/ed_grossdomesticproduct.json"],
        columns=[("ThirdIndustryGDP", ["ThirdIndustryGDP"])],
        expected_row_count=354,
        steps=steps,
    )
    plan["intent"]["requirements"][0] = {
        "statement": "Return the historical third-industry GDP records.",
        "requirement_type": "measure",
        "quote": "第三产业国内生产总值",
    }
    plan["intent"]["unresolved"] = [
        "No aggregation, sorting, or null replacement was explicitly requested."
    ]
    plan["output_spec"]["row_grain"] = "one source GDP record"
    task = PublicTask(
        record=TaskRecord(
            task_id=task_record["task_id"],
            difficulty=str(task_record.get("difficulty", "")),
            question=task_record["question"],
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_file",
                {"file_path": "/context/knowledge.md"},
                "task-2-knowledge",
            ),
            _tool_response(
                "read_file",
                {"file_path": "/context/json/ed_grossdomesticproduct.json", "limit": 40},
                "task-2-source",
            ),
            _tool_response("analyze_plan", plan, "task-2-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": steps[0], "status": "in_progress"},
                        {"content": steps[1], "status": "pending"},
                    ]
                },
                "task-2-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "import json\n"
                        "with open("
                        "'/context/json/ed_grossdomesticproduct.json', "
                        "encoding='utf-8') as handle:\n"
                        "    records = json.load(handle)['records']\n"
                        "rows = [[record.get('ThirdIndustryGDP')] "
                        "for record in records]\n"
                        "set_answer(['ThirdIndustryGDP'], rows)\n"
                    )
                },
                "task-2-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.columns == ["ThirdIndustryGDP"]
    assert result.answer.rows == source_rows
    assert len(result.answer.rows) == 354
    assert sum(row[0] is None for row in result.answer.rows) == 41


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
    assert result.failure_reason == "Agent completed without preparing an answer."
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
    assert result.answer is not None
    assert result.answer.rows == [["done"]]


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
        "execute_python" in tool_names and "task" in tool_names
        for tool_names in model.bound_tool_sets
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
    assert result.failure_reason == "Agent did not prepare an answer within 2 model calls."


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


def test_python_tool_schema_keeps_answer_rows_out_of_model_input(
    tmp_path: Path,
) -> None:
    (tmp_path / "scratch").mkdir()
    tool = create_execute_python_tool(tmp_path, DeepAgentConfig())

    schema = tool.tool_call_schema.model_json_schema()

    assert schema["required"] == ["code"]
    assert set(schema["properties"]) == {"code"}


def test_set_answer_accepts_object_rows(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "rows = [{'value': 'one'}, {'value': 'two'}]\n"
                        "set_answer(['value'], rows)\n"
                    )
                },
                "object-rows",
            )
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["one"], ["two"]]


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
