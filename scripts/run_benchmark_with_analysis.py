from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_agent_baseline.config import load_app_config
from data_agent_baseline.run.runner import run_benchmark

from scripts.analyze_benchmark_run import analyze_run, _write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        default=None,
        help="Task id to run; repeat for multiple tasks.",
    )
    parser.add_argument("--gold-root", type=Path, default=Path("data/output"))
    parser.add_argument("--input-root", type=Path, default=Path("data/input"))
    args = parser.parse_args()

    config = load_app_config(args.config)
    run_dir, artifacts = run_benchmark(
        config=config,
        limit=args.limit,
        task_ids=args.task_ids,
    )
    analyses, aggregate = analyze_run(run_dir, args.gold_root, args.input_root)
    _write_report(run_dir=run_dir, analyses=analyses, aggregate=aggregate)
    print(
        json.dumps(
            {
                **aggregate,
                "attempted_tasks": len(artifacts),
                "analysis_report": str(run_dir / "analysis_report.json"),
                "analysis_markdown": str(run_dir / "analysis_report.zh.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
