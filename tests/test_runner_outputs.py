from __future__ import annotations

import csv
import json
import multiprocessing
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult

from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.run.runner import _run_single_task_with_timeout, run_single_task
from tests.test_deep_agent import ScriptedChatModel, _answer_response


class TraceInspectingModel(ScriptedChatModel):
    trace_path: Path
    initial_trace_seen: bool = False
    incremental_trace: dict[str, Any] | None = None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.call_count == 0:
            initial_trace = json.loads(self.trace_path.read_text(encoding="utf-8"))
            self.initial_trace_seen = (
                initial_trace["status"] == "running" and initial_trace["steps"] == []
            )
        elif self.call_count == 4:
            self.incremental_trace = json.loads(self.trace_path.read_text(encoding="utf-8"))

        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


def test_runner_preserves_prediction_and_trace_contract(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    task_dir = dataset_root / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    task_dir.joinpath("task.json").write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "difficulty": "easy",
                "question": "Return one row.",
            }
        ),
        encoding="utf-8",
    )
    context_dir.joinpath("data.txt").write_text("value\n", encoding="utf-8")
    context_dir.joinpath("knowledge.md").write_text(
        "Use the source value exactly.\n",
        encoding="utf-8",
    )

    model = ScriptedChatModel(
        request_quote="Return one row.",
        knowledge_quote="Use the source value exactly.",
        responses=[
            _answer_response(
                columns=["value"],
                rows=[["one"]],
                tool_call_id="answer-output",
            )
        ]
    )
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(
            model="fake",
            api_base="https://example.invalid/v1",
            api_key="test",
        ),
        run=RunConfig(output_dir=tmp_path / "runs", max_workers=1),
    )
    run_output_dir = tmp_path / "run"
    run_output_dir.mkdir()

    artifact = run_single_task(
        task_id="task_1",
        config=config,
        run_output_dir=run_output_dir,
        model=model,
    )

    assert artifact.succeeded
    assert artifact.prediction_csv_path is not None
    with artifact.prediction_csv_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [["value"], ["one"]]

    trace = json.loads(artifact.trace_path.read_text(encoding="utf-8"))
    assert trace["task_id"] == "task_1"
    assert trace["answer"] == {"columns": ["value"], "rows": [["one"]]}
    assert [step["action"] for step in trace["steps"][:5]] == [
        "system_prompt",
        "user_prompt",
        "read_doc",
        "analyze_plan",
        "write_todos",
    ]
    tool_calls = [
        tool_call
        for step in trace["steps"]
        for tool_call in step["observation"].get("tool_calls", [])
    ]
    assert [tool_call["tool_call_id"] for tool_call in tool_calls] == [
        "schema-call",
        "plan-call",
        "todos-call",
        "answer-output",
    ]
    assert trace["succeeded"] is True
    assert trace["status"] == "completed"
    assert "e2e_elapsed_seconds" in trace


def test_runner_updates_trace_before_next_model_call(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    task_dir = dataset_root / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    task_dir.joinpath("task.json").write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "difficulty": "easy",
                "question": "Inspect and return one row.",
            }
        ),
        encoding="utf-8",
    )
    context_dir.joinpath("data.txt").write_text("value\n", encoding="utf-8")
    context_dir.joinpath("knowledge.md").write_text(
        "Use the source value exactly.\n",
        encoding="utf-8",
    )

    run_output_dir = tmp_path / "run"
    trace_path = run_output_dir / "task_1" / "trace.json"
    model = TraceInspectingModel(
        trace_path=trace_path,
        request_quote="Inspect and return one row.",
        knowledge_quote="Use the source value exactly.",
        responses=[
            AIMessage(
                content="Inspecting.",
                tool_calls=[
                    {
                        "name": "read_doc",
                        "args": {"path": "/context/data.txt"},
                        "id": "read-before-answer",
                        "type": "tool_call",
                    }
                ],
            ),
            _answer_response(
                columns=["value"],
                rows=[["one"]],
                tool_call_id="answer-after-read",
            ),
        ],
    )
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(
            model="fake",
            api_base="https://example.invalid/v1",
            api_key="test",
        ),
        run=RunConfig(output_dir=tmp_path / "runs", max_workers=1),
    )

    artifact = run_single_task(
        task_id="task_1",
        config=config,
        run_output_dir=run_output_dir,
        model=model,
    )

    assert artifact.succeeded
    assert model.initial_trace_seen
    assert model.incremental_trace is not None
    assert model.incremental_trace["status"] == "running"
    incremental_actions = [
        step["action"] for step in model.incremental_trace["steps"]
    ]
    assert incremental_actions[:5] == [
        "system_prompt",
        "user_prompt",
        "read_doc",
        "analyze_plan",
        "write_todos",
    ]
    incremental_tool_calls = [
        tool_call
        for step in model.incremental_trace["steps"]
        for tool_call in step["observation"].get("tool_calls", [])
    ]
    assert [tool_call["name"] for tool_call in incremental_tool_calls] == [
        "read_doc",
        "analyze_plan",
        "write_todos",
        "read_doc",
    ]
    assert incremental_tool_calls[3]["tool_call_id"] == "read-before-answer"


def _send_large_subprocess_result(
    task_id: str,
    config: AppConfig,
    trace_path: Path,
    connection: Any,
) -> None:
    del task_id, config, trace_path
    connection.send(
        {
            "ok": True,
            "run_result": {
                "task_id": "task_1",
                "answer": {"columns": ["value"], "rows": [["one"]]},
                "steps": [{"payload": "x" * 250_000}],
                "failure_reason": None,
                "succeeded": True,
            },
        }
    )
    connection.close()


def test_subprocess_result_is_received_before_join(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(),
        run=RunConfig(output_dir=tmp_path / "runs", task_timeout_seconds=5),
    )
    trace_path = tmp_path / "trace.json"
    monkeypatch.setattr(
        "data_agent_baseline.run.runner._run_single_task_in_subprocess",
        _send_large_subprocess_result,
    )

    result = _run_single_task_with_timeout(
        task_id="task_1",
        config=config,
        trace_path=trace_path,
    )

    assert result["succeeded"] is True
    assert result["answer"] == {"columns": ["value"], "rows": [["one"]]}
    assert len(result["steps"][0]["payload"]) == 250_000
    assert not multiprocessing.active_children()
