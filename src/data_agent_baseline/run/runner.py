from __future__ import annotations

import csv
import json
import multiprocessing
import os
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from multiprocessing.connection import Connection
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from data_agent_baseline.agents.deep_agent import DeepAgent
from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig

_REPLACE_RETRY_DELAYS_SECONDS = (0.05, 0.1, 0.2, 0.4, 0.8)


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path)
            if self.prediction_csv_path
            else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_chat_model(config: AppConfig) -> ChatOpenAI:
    if not config.agent.api_key:
        raise RuntimeError("Missing model API key in config.agent.api_key.")
    if not config.agent.model:
        raise RuntimeError("Missing model name in config.agent.model.")
    if not config.agent.api_base:
        raise RuntimeError("Missing model API base in config.agent.api_base.")

    return ChatOpenAI(
        model=config.agent.model,
        base_url=config.agent.api_base.rstrip("/"),
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
        timeout=1800.0,
        max_retries=config.agent.max_retries,
        max_tokens=8192,
    )


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")


def _replace_with_retry(temp_path: Path, path: Path) -> None:
    last_error: PermissionError | None = None
    for delay_seconds in (*_REPLACE_RETRY_DELAYS_SECONDS, 0.0):
        try:
            temp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            if delay_seconds > 0:
                time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error


def _write_text_file(path: Path, content: str, *, fallback_direct: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(path)
    temp_path.write_text(
        content,
        encoding="utf-8",
    )
    try:
        _replace_with_retry(temp_path, path)
    except PermissionError:
        if not fallback_direct:
            raise
        path.write_text(content, encoding="utf-8")
        temp_path.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text_file(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        fallback_direct=True,
    )


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not columns:
        raise ValueError("Cannot write prediction CSV without columns.")
    if not rows:
        raise ValueError("Cannot write prediction CSV without rows.")
    if any(len(row) != len(columns) for row in rows):
        raise ValueError("Prediction CSV rows must match the number of columns.")

    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    _write_text_file(path, buffer.getvalue(), fallback_direct=False)


def _is_prepared_answer(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    columns = value.get("columns")
    rows = value.get("rows")
    return (
        isinstance(columns, list)
        and bool(columns)
        and isinstance(rows, list)
        and bool(rows)
    )


def _failure_run_result_payload(
    task_id: str,
    failure_reason: str,
    *,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    partial_payload: dict[str, Any] = {}
    if trace_path is not None and trace_path.exists():
        try:
            loaded_payload = json.loads(trace_path.read_text(encoding="utf-8"))
            if isinstance(loaded_payload, dict):
                partial_payload = loaded_payload
        except (OSError, json.JSONDecodeError):
            partial_payload = {}

    partial_answer = partial_payload.get("answer")
    if _is_prepared_answer(partial_answer):
        return {
            "task_id": task_id,
            "answer": partial_answer,
            "steps": partial_payload.get("steps", []),
            "failure_reason": None,
            "succeeded": True,
        }

    return {
        "task_id": task_id,
        "answer": partial_payload.get("answer"),
        "steps": partial_payload.get("steps", []),
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _trace_payload(
    run_result: AgentRunResult,
    *,
    status: str,
    started_at: float,
) -> dict[str, Any]:
    payload = run_result.to_dict()
    payload["status"] = status
    payload["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return payload


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model: BaseChatModel | None = None,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    agent = DeepAgent(
        model=model or build_chat_model(config),
        config=DeepAgentConfig(
            max_steps=config.agent.max_steps,
            execute_timeout_seconds=config.agent.execute_timeout_seconds,
            max_output_bytes=config.agent.max_output_bytes,
            model_call_interval_seconds=config.agent.model_call_interval_seconds,
            question_structure_enabled=config.agent.question_structure_enabled,
        ),
    )
    started_at = perf_counter()

    def write_trace(run_result: AgentRunResult, status: str) -> None:
        if trace_path is not None:
            _write_json(
                trace_path,
                _trace_payload(
                    run_result,
                    status=status,
                    started_at=started_at,
                ),
            )

    run_result = agent.run(task, trace_callback=write_trace if trace_path is not None else None)
    return run_result.to_dict()


def _run_single_task_in_subprocess(
    task_id: str,
    config: AppConfig,
    trace_path: Path,
    connection: Connection,
) -> None:
    try:
        payload = {
            "ok": True,
            "run_result": _run_single_task_core(
                task_id=task_id,
                config=config,
                trace_path=trace_path,
            ),
        }
    except BaseException as exc:  # noqa: BLE001
        payload = {"ok": False, "error": str(exc)}
    try:
        connection.send(payload)
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()


def _stop_process(process: multiprocessing.Process) -> None:
    process.join(timeout=1.0)
    if process.is_alive():
        _terminate_process_tree(process)
        process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join()


def _terminate_process_tree(process: multiprocessing.Process) -> None:
    if os.name == "nt" and process.pid is not None:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    process.terminate()


def _run_single_task_with_timeout(
    *,
    task_id: str,
    config: AppConfig,
    trace_path: Path,
) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(
            task_id=task_id,
            config=config,
            trace_path=trace_path,
        )

    receive_connection, send_connection = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, trace_path, send_connection),
    )
    process.start()
    send_connection.close()
    deadline = perf_counter() + timeout_seconds
    result: dict[str, Any] | None = None

    try:
        while perf_counter() < deadline:
            remaining_seconds = deadline - perf_counter()
            if receive_connection.poll(min(0.1, max(remaining_seconds, 0.0))):
                try:
                    received = receive_connection.recv()
                except EOFError:
                    break
                if isinstance(received, dict):
                    result = received
                break
            if not process.is_alive():
                break
    finally:
        receive_connection.close()

    if result is None and process.is_alive():
        _stop_process(process)
        return _failure_run_result_payload(
            task_id,
            f"Task timed out after {timeout_seconds} seconds.",
            trace_path=trace_path,
        )

    _stop_process(process)
    if result is None:
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            return _failure_run_result_payload(
                task_id,
                f"Task exited unexpectedly with exit code {exit_code}.",
                trace_path=trace_path,
            )
        return _failure_run_result_payload(
            task_id,
            "Task exited without returning a result.",
            trace_path=trace_path,
        )

    if result.get("ok"):
        return dict(result["run_result"])
    return _failure_run_result_payload(
        task_id,
        f"Task failed with uncaught error: {result['error']}",
        trace_path=trace_path,
    )


def _write_task_outputs(
    task_id: str,
    run_output_dir: Path,
    run_result: dict[str, Any],
    *,
    trace_path: Path,
) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(trace_path, run_result)

    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model: BaseChatModel | None = None,
) -> TaskRunArtifacts:
    started_at = perf_counter()
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(
        trace_path,
        {
            "task_id": task_id,
            "answer": None,
            "steps": [],
            "failure_reason": None,
            "succeeded": False,
            "status": "running",
            "e2e_elapsed_seconds": 0.0,
        },
    )

    if model is None:
        run_result = _run_single_task_with_timeout(
            task_id=task_id,
            config=config,
            trace_path=trace_path,
        )
    else:
        run_result = _run_single_task_core(
            task_id=task_id,
            config=config,
            model=model,
            trace_path=trace_path,
        )
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    run_result["status"] = "completed" if run_result.get("succeeded") else "failed"
    return _write_task_outputs(
        task_id,
        run_output_dir,
        run_result,
        trace_path=trace_path,
    )


def run_benchmark(
    *,
    config: AppConfig,
    model: BaseChatModel | None = None,
    limit: int | None = None,
    task_ids: list[str] | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(
        config.run.output_dir, run_id=config.run.run_id
    )

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks(task_ids=task_ids)
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=model,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
