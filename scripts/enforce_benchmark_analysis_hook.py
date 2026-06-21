#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from typing import Any

ALLOWED_COMMAND_MARKERS = (
    "scripts/run_benchmark_with_analysis.py",
    "scripts\\run_benchmark_with_analysis.py",
    "scripts/run_task_with_analysis.py",
    "scripts\\run_task_with_analysis.py",
    "scripts/analyze_benchmark_run.py",
    "scripts\\analyze_benchmark_run.py",
    "scripts/run_and_analyze.sh",
    "scripts\\run_and_analyze.sh",
)

REWRITE_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"(?i)\b(?:uv\s+run\s+python\s+-m\s+data_agent_baseline\.cli|python\s+-m\s+data_agent_baseline\.cli)\s+run-benchmark\b"),
        "uv run python scripts/run_benchmark_with_analysis.py",
        "Benchmark command was rewritten to scripts/run_benchmark_with_analysis.py so the analysis report is generated automatically.",
    ),
    (
        re.compile(r"(?i)\b(?:uv\s+run\s+python\s+-m\s+data_agent_baseline\.cli|python\s+-m\s+data_agent_baseline\.cli)\s+run-task\b"),
        "uv run python scripts/run_task_with_analysis.py",
        "Task run command was rewritten to scripts/run_task_with_analysis.py so the analysis report is generated automatically.",
    ),
    (
        re.compile(r"(?i)\b(?:uv\s+run\s+)?dabench\s+run-benchmark\b"),
        "uv run python scripts/run_benchmark_with_analysis.py",
        "Benchmark command was rewritten to scripts/run_benchmark_with_analysis.py so the analysis report is generated automatically.",
    ),
    (
        re.compile(r"(?i)\b(?:uv\s+run\s+)?dabench\s+run-task\b"),
        "uv run python scripts/run_task_with_analysis.py",
        "Task run command was rewritten to scripts/run_task_with_analysis.py so the analysis report is generated automatically.",
    ),
)


def _load_payload() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    return json.loads(raw)


def _extract_terminal_command(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if payload.get("hookEventName") != "PreToolUse":
        return None, None
    if payload.get("tool_name") != "run_in_terminal":
        return None, None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None, None
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return None, None
    return tool_input, command


def _already_analysis_aware(command: str) -> bool:
    normalized = command.lower()
    return any(marker in normalized for marker in ALLOWED_COMMAND_MARKERS)


def _rewrite_command(command: str) -> tuple[str, str] | None:
    if _already_analysis_aware(command):
        return None
    for pattern, replacement, message in REWRITE_RULES:
        rewritten, replacement_count = pattern.subn(replacement, command, count=1)
        if replacement_count:
            return rewritten, message
    return None


def main() -> None:
    payload = _load_payload()
    tool_input, command = _extract_terminal_command(payload)
    if tool_input is None or command is None:
        return

    rewrite_result = _rewrite_command(command)
    if rewrite_result is None:
        return
    rewritten_command, message = rewrite_result

    response = {
        "systemMessage": message,
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": {
                **tool_input,
                "command": rewritten_command,
            },
            "additionalContext": message,
        },
    }
    json.dump(response, sys.stdout)


if __name__ == "__main__":
    main()