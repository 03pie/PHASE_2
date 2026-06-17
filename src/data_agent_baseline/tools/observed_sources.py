from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from data_agent_baseline.agents.deep_state import BenchmarkDeepAgentState
from data_agent_baseline.tools._helpers import jsonable


def normalize_source_path(path: str) -> str:
    return str(path or "").replace("\\", "/")


def sample_hash(value: Any) -> str:
    encoded = json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_key(source: Mapping[str, Any]) -> str:
    return normalize_source_path(str(source.get("path") or ""))


def merge_observed_sources(
    existing: Any,
    additions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    if isinstance(existing, list):
        for item in existing:
            if not isinstance(item, Mapping):
                continue
            key = source_key(item)
            if not key:
                continue
            merged[key] = dict(item)
            order.append(key)

    for source in additions:
        key = source_key(source)
        if not key:
            continue
        normalized = {
            key: jsonable(value)
            for key, value in source.items()
            if value is not None and value != []
        }
        normalized["path"] = key
        if key in merged:
            merged[key] = {**merged[key], **normalized}
        else:
            merged[key] = normalized
            order.append(key)
    return [merged[key] for key in order if key in merged]


def observed_sources_command(
    *,
    state: Mapping[str, Any],
    message: ToolMessage,
    sources: list[dict[str, Any]],
) -> Command[BenchmarkDeepAgentState]:
    return Command(
        update={
            "observed_sources": merge_observed_sources(
                state.get("observed_sources"),
                sources,
            ),
            "messages": [message],
        }
    )
