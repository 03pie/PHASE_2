from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, NotRequired

from deepagents.graph import DeepAgentState

from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.schema import AnswerTable

# 运行过程中的增量 trace 回调，runner 用它持续落盘执行状态。
TraceCallback = Callable[[AgentRunResult, str], None]


def merge_observed_sources_state(
    left: list[dict[str, Any]] | None,
    right: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge concurrently observed source metadata by normalized source path."""

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for sources in (left, right):
        if not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, dict):
                continue
            path = str(source.get("path") or "").replace("\\", "/")
            if not path:
                continue
            if path not in merged:
                order.append(path)
                merged[path] = {"path": path}
            merged[path].update(source)
            merged[path]["path"] = path
    return [merged[path] for path in order if path in merged]


def merge_narrative_extraction_cache_state(
    left: list[dict[str, Any]] | None,
    right: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge cached narrative extraction windows by source/field/anchor key."""

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for caches in (left, right):
        if not isinstance(caches, list):
            continue
        for cache in caches:
            if not isinstance(cache, dict):
                continue
            key = str(cache.get("cache_key") or "").strip()
            if not key:
                continue
            if key not in merged:
                order.append(key)
                merged[key] = {"cache_key": key, "records": []}
            current = merged[key]
            for field in ("path", "columns", "record_anchor"):
                if cache.get(field) not in (None, "", []):
                    current[field] = cache[field]
            record_index = {
                str(record.get("record_key") or ""): record
                for record in current.get("records", [])
                if isinstance(record, dict) and str(record.get("record_key") or "")
            }
            for record in cache.get("records") or []:
                if not isinstance(record, dict):
                    continue
                record_key = str(record.get("record_key") or "").strip()
                if not record_key:
                    continue
                if record_key not in record_index:
                    copied = dict(record)
                    current.setdefault("records", []).append(copied)
                    record_index[record_key] = copied
                    continue
                target = record_index[record_key]
                target_values = target.setdefault("values", {})
                if not isinstance(target_values, dict):
                    target_values = {}
                    target["values"] = target_values
                for field, value in (record.get("values") or {}).items():
                    if target_values.get(field) in (None, "") and value not in (None, ""):
                        target_values[field] = value
                target_matched = set(target.get("matched_fields") or [])
                target_matched.update(record.get("matched_fields") or [])
                if target_matched:
                    target["matched_fields"] = sorted(str(item) for item in target_matched)
                line_number = record.get("line_number")
                if isinstance(line_number, int):
                    existing_line = target.get("line_number")
                    target["line_number"] = (
                        min(existing_line, line_number)
                        if isinstance(existing_line, int)
                        else line_number
                    )
    return [merged[key] for key in order if key in merged]


@dataclass(frozen=True, slots=True)
class DeepAgentConfig:
    """DeepAgent 的运行限制配置。"""

    max_steps: int = 16
    execute_timeout_seconds: int = 30
    max_output_bytes: int = 100_000
    model_call_interval_seconds: float = 0.0
    question_structure_enabled: bool = False

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if self.execute_timeout_seconds < 1:
            raise ValueError("execute_timeout_seconds must be at least 1.")
        if self.max_output_bytes < 1:
            raise ValueError("max_output_bytes must be at least 1.")
        if self.model_call_interval_seconds < 0:
            raise ValueError("model_call_interval_seconds cannot be negative.")


class BenchmarkDeepAgentState(DeepAgentState):
    """在 DeepAgents 默认状态上扩展基准任务专用字段。"""

    original_request: NotRequired[str]
    question_structure: NotRequired[dict[str, Any]]
    question_structure_enforced: NotRequired[bool]
    knowledge_content: NotRequired[str]
    knowledge_content_hash: NotRequired[str]
    answer: NotRequired[AnswerTable | None]
    prepared_answer: NotRequired[AnswerTable | None]
    answer_candidate: NotRequired[dict[str, Any]]
    analysis_plan: NotRequired[dict[str, Any]]
    evidence_contract: NotRequired[dict[str, Any]]
    observed_sources: NotRequired[
        Annotated[list[dict[str, Any]], merge_observed_sources_state]
    ]
    narrative_extraction_cache: NotRequired[
        Annotated[list[dict[str, Any]], merge_narrative_extraction_cache_state]
    ]
    todos: NotRequired[list[dict[str, str]]]
