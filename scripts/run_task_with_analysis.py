#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_agent_baseline.config import load_app_config


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run one task first, then automatically analyze the exact run directory "
            "that was produced in this execution."
        )
    )
    parser.add_argument("task_id", help="Task ID to run, for example task_19.")
    parser.add_argument(
        "--config",
        dest="config_flag",
        type=Path,
        default=None,
        help="Benchmark config path.",
    )
    parser.add_argument(
        "--analysis-penalty-lambda",
        type=float,
        default=None,
        help="Optional lambda for the local diagnostic score in the analysis report.",
    )
    parser.add_argument(
        "--analysis-output-md",
        type=Path,
        default=None,
        help="Optional markdown analysis report output path.",
    )
    parser.add_argument(
        "--analysis-output-json",
        type=Path,
        default=None,
        help="Optional JSON analysis report output path.",
    )
    parser.add_argument(
        "--analysis-all-tasks",
        action="store_true",
        help="Analyze all tasks from the metadata source instead of only attempted tasks.",
    )
    return parser.parse_known_args(sys.argv[1:])


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config_flag is not None:
        return args.config_flag.resolve()
    default_path = PROJECT_ROOT / "configs" / "submission.yaml"
    if default_path.exists():
        return default_path.resolve()
    raise FileNotFoundError("Please provide --config PATH.")


def snapshot_run_dirs(runs_root: Path) -> set[Path]:
    if not runs_root.exists():
        return set()
    return {path.resolve() for path in runs_root.iterdir() if path.is_dir()}


def detect_run_dir(runs_root: Path, before: set[Path], started_at: float) -> Path | None:
    after = snapshot_run_dirs(runs_root)
    created = sorted(
        after - before,
        key=lambda path: ((path / "summary.json").exists(), path.stat().st_mtime),
        reverse=True,
    )
    if created:
        return created[0]

    if not after:
        return None

    fresh = [
        path
        for path in after
        if path.stat().st_mtime >= started_at - 1.0
    ]
    if fresh:
        fresh.sort(
            key=lambda path: ((path / "summary.json").exists(), path.stat().st_mtime),
            reverse=True,
        )
        return fresh[0]
    return None


def run_command(command: list[str]) -> int:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


def resolve_dabench_command_prefix() -> list[str]:
    candidate = Path(sys.executable).resolve().with_name("dabench")
    if candidate.exists():
        return [str(candidate)]
    return [sys.executable, "-c", "from data_agent_baseline.cli import main; main()"]


def main() -> None:
    args, task_args = parse_args()
    config_path = resolve_config_path(args)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    app_config = load_app_config(config_path)
    runs_root = app_config.run.output_dir.resolve()
    before = snapshot_run_dirs(runs_root)
    started_at = time.time()

    run_task_command = [
        *resolve_dabench_command_prefix(),
        "run-task",
        args.task_id,
        "--config",
        str(config_path),
        *task_args,
    ]
    print(f"[run_task_with_analysis] benchmark config: {config_path}", flush=True)
    print(f"[run_task_with_analysis] task id: {args.task_id}", flush=True)
    run_task_exit_code = run_command(run_task_command)

    run_dir = detect_run_dir(runs_root, before, started_at)
    if run_dir is None:
        if run_task_exit_code != 0:
            raise SystemExit(run_task_exit_code)
        raise RuntimeError(
            "Task run finished but no new run directory was detected. "
            "Please rerun with a unique run.output_dir/run_id."
        )

    analysis_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "analyze_benchmark_run.py"),
        "--config",
        str(config_path),
        "--run-dir",
        str(run_dir),
    ]
    if args.analysis_penalty_lambda is not None:
        analysis_command.extend(["--penalty-lambda", str(args.analysis_penalty_lambda)])
    if args.analysis_output_md is not None:
        analysis_command.extend(["--output-md", str(args.analysis_output_md.resolve())])
    if args.analysis_output_json is not None:
        analysis_command.extend(["--output-json", str(args.analysis_output_json.resolve())])
    if args.analysis_all_tasks:
        analysis_command.append("--all-tasks")

    print(f"[run_task_with_analysis] analyzing run: {run_dir}", flush=True)
    analysis_exit_code = run_command(analysis_command)

    if run_task_exit_code != 0:
        raise SystemExit(run_task_exit_code)
    if analysis_exit_code != 0:
        raise SystemExit(analysis_exit_code)


if __name__ == "__main__":
    main()