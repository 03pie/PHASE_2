from __future__ import annotations

import csv
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import data_agent_baseline.run.runner as runner_module
from langchain_core.messages import AIMessage

from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.run.runner import _run_single_task_with_timeout, run_single_task


class _ScriptedModel:
    def __init__(self, *responses: dict[str, object]) -> None:
        self.responses = list(responses)

    def bind_tools(self, _tools: object, **_kwargs: object) -> "_ScriptedModel":
        return self

    def invoke(self, _messages: object) -> AIMessage:
        if not self.responses:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "blocked",
                        "args": {"reason": "script exhausted"},
                        "id": "call_exhausted",
                    }
                ],
            )
        response = self.responses.pop(0)
        return AIMessage(content="", tool_calls=response["tool_calls"])  # type: ignore[arg-type]


def _call(name: str, args: dict[str, object], call_id: str) -> dict[str, object]:
    return {"tool_calls": [{"name": name, "args": args, "id": call_id}]}


def _scripted_csv_model() -> _ScriptedModel:
    return _ScriptedModel(
        _call("inspect_source", {"path": "/context/records.csv"}, "call_inspect"),
        _call(
            "bind",
            {
                "binding_type": "structured_source",
                "source_ref": "src_0001",
                "evidence_refs": ["ev_0001"],
                "allowed_columns": ["name", "amount"],
                "alignment": "observed CSV",
            },
            "call_bind",
        ),
        _call("inspect_relation", {"binding_ref": "bind_0001"}, "call_relation"),
        _call(
            "run_verified_compute",
            {"binding_refs": ["bind_0001"], "sql": "SELECT amount FROM rel_0001"},
            "call_compute",
        ),
        _call(
            "verify_alignment",
            {
                "decision": "candidate_answer",
                "target_kind": "compute_result",
                "compute_refs": ["comp_0001"],
                "binding_refs": ["bind_0001"],
                "evidence_refs": ["ev_0001"],
                "alignment": "The compute result is the final answer table requested.",
            },
            "call_verify_answer",
        ),
        _call("submit_final", {"compute_ref": "comp_0001", "answer": {"columns": ["amount"]}}, "call_final"),
    )


def _write_task(dataset_root: Path, *, question: str) -> None:
    task_dir = dataset_root / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    task_dir.joinpath("task.json").write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "difficulty": "easy",
                "question": question,
            }
        ),
        encoding="utf-8",
    )
    context_dir.joinpath("records.csv").write_text(
        "name,amount\nalpha,3\nbeta,5\n",
        encoding="utf-8",
    )
    context_dir.joinpath("knowledge.md").write_text(
        "# Records\n\n```sql\nSELECT amount FROM records\n```\n",
        encoding="utf-8",
    )


def test_runner_preserves_prediction_and_trace_contract(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_task(dataset_root, question="records amount")
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(model="unused", api_base="https://example.invalid/v1", api_key="test"),
        run=RunConfig(output_dir=tmp_path / "runs", max_workers=1),
    )
    run_output_dir = tmp_path / "run"
    run_output_dir.mkdir()

    artifact = run_single_task(
        task_id="task_1",
        config=config,
        run_output_dir=run_output_dir,
        model=_scripted_csv_model(),
    )

    assert artifact.succeeded
    assert artifact.prediction_csv_path is not None
    with artifact.prediction_csv_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [["amount"], ["3"], ["5"]]

    trace = json.loads(artifact.trace_path.read_text(encoding="utf-8"))
    assert trace["task_id"] == "task_1"
    assert trace["answer"] == {"columns": ["amount"], "rows": [["3"], ["5"]]}
    actions = [step["action"] for step in trace["steps"]]
    assert actions[:2] == [
        "codex_bootstrap_inventory",
        "codex_bootstrap_knowledge",
    ]
    assert "inspect_source" in actions
    assert "bind" in actions
    assert "inspect_relation" in actions
    assert "run_verified_compute" in actions
    assert "submit_final" in actions
    assert actions[-1] == "codex_final_audit"
    tool_steps = [
        step
        for step in trace["steps"]
        if step["action"] in {"inspect_source", "bind", "inspect_relation", "run_verified_compute"}
    ]
    assert tool_steps
    assert all(step["tool_call_id"] for step in tool_steps)
    loop_steps = [step for step in trace["steps"] if step["action"] == "codex_turn"]
    assert loop_steps
    assert all("tool_calls" in step["observation"] for step in loop_steps)
    assert any(step["observation"]["tool_calls"] for step in loop_steps)
    assert trace["succeeded"] is True
    assert trace["status"] == "completed"
    assert "e2e_elapsed_seconds" in trace


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


def _write_answer_trace_then_sleep(
    task_id: str,
    config: AppConfig,
    trace_path: Path,
    connection: Any,
) -> None:
    del config, connection
    trace_path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "answer": {"columns": ["value"], "rows": [["one"]]},
                "steps": [{"action": "final_audit"}],
                "failure_reason": None,
                "succeeded": False,
                "status": "running",
            }
        ),
        encoding="utf-8",
    )
    time.sleep(5)


def test_timeout_preserves_prepared_answer_from_trace(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(),
        run=RunConfig(output_dir=tmp_path / "runs", task_timeout_seconds=2),
    )
    trace_path = tmp_path / "trace.json"
    monkeypatch.setattr(
        "data_agent_baseline.run.runner._run_single_task_in_subprocess",
        _write_answer_trace_then_sleep,
    )

    result = _run_single_task_with_timeout(
        task_id="task_1",
        config=config,
        trace_path=trace_path,
    )

    assert result["succeeded"] is True
    assert result["failure_reason"] is None
    assert result["answer"] == {"columns": ["value"], "rows": [["one"]]}
    assert not multiprocessing.active_children()


def test_timeout_cleanup_terminates_windows_process_tree(monkeypatch: Any) -> None:
    calls: list[list[str]] = []

    class FakeProcess:
        pid = 12345

        def terminate(self) -> None:
            raise AssertionError("Windows cleanup should use taskkill /T.")

    monkeypatch.setattr(runner_module.os, "name", "nt")
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )

    runner_module._terminate_process_tree(FakeProcess())  # type: ignore[arg-type]

    assert calls == [["taskkill", "/PID", "12345", "/T", "/F"]]


class BrokenPipeConnection:
    closed: bool = False

    def send(self, payload: dict[str, Any]) -> None:
        del payload
        raise BrokenPipeError()

    def close(self) -> None:
        self.closed = True


def test_subprocess_send_ignores_closed_parent_pipe(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config = AppConfig(
        dataset=DatasetConfig(root_path=tmp_path / "dataset"),
        agent=AgentConfig(),
        run=RunConfig(output_dir=tmp_path / "runs"),
    )
    monkeypatch.setattr(
        runner_module,
        "_run_single_task_core",
        lambda **kwargs: {
            "task_id": kwargs["task_id"],
            "answer": {"columns": ["value"], "rows": [["one"]]},
            "steps": [],
            "failure_reason": None,
            "succeeded": True,
        },
    )
    connection = BrokenPipeConnection()

    runner_module._run_single_task_in_subprocess(
        "task_1",
        config,
        tmp_path / "trace.json",
        connection,
    )

    assert connection.closed
