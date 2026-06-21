#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_agent_baseline.config import load_app_config

UTC = timezone.utc

NULL_TOKENS = {"", "null", "none", "nan", "nat", "<na>"}
NUMBER_RE = re.compile(r"^[+-]?(?:(?:\d+\.\d+)|(?:\d+)|(?:\.\d+))$")
DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
TASK_ROW_RE = re.compile(
    r"^\|\s*(task_\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|$"
)
FIRST_NAME_HEADERS = {"first_name", "firstname", "given_name", "givenname"}
LAST_NAME_HEADERS = {"last_name", "lastname", "surname", "family_name", "familyname"}


@dataclass(frozen=True, slots=True)
class TaskMeta:
    task_id: str
    difficulty: str
    data_formats: str
    question_en: str
    question_zh: str


@dataclass(frozen=True, slots=True)
class RunTaskInfo:
    task_id: str
    succeeded: bool
    failure_reason: str | None
    prediction_csv_path: str | None
    trace_path: str | None


@dataclass(frozen=True, slots=True)
class TableData:
    headers: list[str]
    rows: list[list[str]]
    width: int


@dataclass(frozen=True, slots=True)
class TaskEvaluation:
    task_id: str
    difficulty: str
    data_formats: str
    question_zh: str
    question_type: str
    modalities: list[str]
    strict_binary_score: float
    gold_column_count: int
    prediction_column_count: int
    matched_gold_columns: int
    missing_gold_columns: int
    extra_prediction_columns: int
    recall: float
    redundancy_ratio: float
    penalized_score: float | None
    task_status: str
    failure_class: str
    benchmark_succeeded: bool | None
    benchmark_failure_reason: str | None
    prediction_exists: bool
    prediction_path: str | None
    gold_path: str
    elapsed_seconds: float | None
    predicted_headers: list[str]
    gold_headers: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a benchmark run against public gold files and generate a Chinese analysis report."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to the benchmark config YAML. When provided, the script will infer "
            "the input root, gold root, and run output root from it."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Benchmark run directory, e.g. artifacts/runs/20260424T103119Z.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recently modified run directory under artifacts/runs.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Root directory that contains run directories. If omitted, infer from config or use artifacts/runs.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Task input root. If omitted, infer from config.",
    )
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=None,
        help="Ground-truth root directory. If omitted, infer as the sibling output/ directory of input root.",
    )
    parser.add_argument(
        "--task-meta",
        type=Path,
        default=None,
        help=(
            "Optional task metadata markdown file like task_questions_bilingual.zh.md. "
            "If omitted, metadata will be derived from input/task.json and the markdown will be used only if found."
        ),
    )
    parser.add_argument(
        "--penalty-lambda",
        type=float,
        default=None,
        help=(
            "Optional lambda for the Section 6.3 local diagnostic score. "
            "If omitted, the script only reports strict leaderboard score plus recall/redundancy diagnostics."
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional output markdown report path. Defaults to <run-dir>/analysis_report.zh.md",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON report path. Defaults to <run-dir>/analysis_report.json",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Analyze all tasks from the metadata file instead of only the tasks attempted in this run.",
    )
    return parser.parse_args()


def resolve_run_dir(run_dir: Path | None, runs_root: Path, use_latest: bool) -> Path:
    if use_latest:
        run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
        if not run_dirs:
            raise FileNotFoundError(f"No run directories found under {runs_root}")
        return max(run_dirs, key=lambda path: path.stat().st_mtime)
    if run_dir is None:
        raise ValueError("Please provide --run-dir or use --latest.")
    return run_dir


def parse_task_metadata(markdown_path: Path) -> dict[str, TaskMeta]:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    metas: dict[str, TaskMeta] = {}
    for line in lines:
        match = TASK_ROW_RE.match(line.strip())
        if match is None:
            continue
        task_id, difficulty, data_formats, question_en, question_zh = match.groups()
        if task_id in metas:
            continue
        metas[task_id] = TaskMeta(
            task_id=task_id,
            difficulty=difficulty.strip(),
            data_formats=data_formats.strip(),
            question_en=question_en.strip(),
            question_zh=question_zh.strip(),
        )
    if not metas:
        raise ValueError(f"Failed to parse any task metadata from {markdown_path}")
    return metas


def infer_gold_root_from_input_root(input_root: Path) -> Path:
    if input_root.name == "input":
        return input_root.parent / "output"
    return input_root.parent / "output"


def infer_paths_from_config(config_path: Path) -> tuple[Path, Path, Path]:
    app_config = load_app_config(config_path.resolve())
    input_root = app_config.dataset.root_path.resolve()
    gold_root = infer_gold_root_from_input_root(input_root)
    runs_root = app_config.run.output_dir.resolve()
    return input_root, gold_root, runs_root


def _classify_context_entry(path: Path) -> str | None:
    if path.is_dir():
        return None

    name = path.name.lower()
    suffix = path.suffix.lower()
    if name == "knowledge.md":
        return "knowledge.md"
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return "db"
    if suffix in {".md", ".txt", ".doc", ".docx"}:
        return "doc"
    if suffix:
        return suffix.lstrip(".")
    return name


def detect_context_formats(context_dir: Path) -> str:
    if not context_dir.exists():
        return ""
    priority = {"csv": 0, "db": 1, "json": 2, "doc": 3, "knowledge.md": 4}
    formats = {
        item
        for child in context_dir.rglob("*")
        for item in [_classify_context_entry(child)]
        if item is not None
    }
    ordered = sorted(formats, key=lambda item: (priority.get(item, 99), item))
    return ", ".join(ordered)


def collect_task_metadata_from_input_root(input_root: Path) -> dict[str, TaskMeta]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    metas: dict[str, TaskMeta] = {}
    for task_dir in sorted(path for path in input_root.iterdir() if path.is_dir() and path.name.startswith("task_")):
        task_json_path = task_dir / "task.json"
        if not task_json_path.exists():
            continue
        payload = json.loads(task_json_path.read_text(encoding="utf-8"))
        task_id = str(payload.get("task_id", task_dir.name))
        question = str(payload.get("question", "")).strip()
        metas[task_id] = TaskMeta(
            task_id=task_id,
            difficulty=str(payload.get("difficulty", "")).strip(),
            data_formats=detect_context_formats(task_dir / "context"),
            question_en=question,
            question_zh=question,
        )
    if not metas:
        raise ValueError(f"Failed to infer any tasks from input root: {input_root}")
    return metas


def load_task_metadata(input_root: Path, markdown_path: Path | None) -> dict[str, TaskMeta]:
    inferred = collect_task_metadata_from_input_root(input_root)
    if markdown_path is None or not markdown_path.exists():
        return inferred
    try:
        overlay = parse_task_metadata(markdown_path)
    except ValueError:
        return inferred
    merged = dict(inferred)
    for task_id, item in overlay.items():
        base = merged.get(task_id)
        if base is None:
            merged[task_id] = item
            continue
        merged[task_id] = TaskMeta(
            task_id=task_id,
            difficulty=item.difficulty or base.difficulty,
            data_formats=item.data_formats or base.data_formats,
            question_en=item.question_en or base.question_en,
            question_zh=item.question_zh or base.question_zh or item.question_en or base.question_en,
        )
    return merged


def resolve_analysis_roots(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None]:
    input_root = args.input_root.resolve() if args.input_root else None
    gold_root = args.gold_root.resolve() if args.gold_root else None
    runs_root = args.runs_root.resolve() if args.runs_root else None
    task_meta_path = args.task_meta.resolve() if args.task_meta else None

    if args.config is not None:
        inferred_input_root, inferred_gold_root, inferred_runs_root = infer_paths_from_config(args.config)
        if input_root is None:
            input_root = inferred_input_root
        if gold_root is None:
            gold_root = inferred_gold_root
        if runs_root is None:
            runs_root = inferred_runs_root

    if input_root is None:
        input_root = (PROJECT_ROOT / "data" / "public" / "input").resolve()
    if gold_root is None:
        gold_root = infer_gold_root_from_input_root(input_root).resolve()
    if runs_root is None:
        runs_root = (PROJECT_ROOT / "artifacts" / "runs").resolve()
    if task_meta_path is None:
        default_meta = PROJECT_ROOT / "task_questions_bilingual.zh.md"
        task_meta_path = default_meta.resolve() if default_meta.exists() else None

    return input_root, gold_root, runs_root, task_meta_path


def load_run_summary(run_dir: Path) -> dict[str, RunTaskInfo]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    result: dict[str, RunTaskInfo] = {}
    for item in payload.get("tasks", []):
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id", "")).strip()
        if not task_id:
            continue
        result[task_id] = RunTaskInfo(
            task_id=task_id,
            succeeded=bool(item.get("succeeded")),
            failure_reason=item.get("failure_reason"),
            prediction_csv_path=item.get("prediction_csv_path"),
            trace_path=item.get("trace_path"),
        )
    return result


def infer_attempted_task_ids(run_dir: Path, run_summary: dict[str, RunTaskInfo]) -> list[str]:
    if run_summary:
        return sorted(run_summary)
    task_ids = [
        path.name
        for path in run_dir.iterdir()
        if path.is_dir() and path.name.startswith("task_")
    ]
    return sorted(task_ids)


def read_trace_elapsed_seconds(trace_path: Path | None) -> float | None:
    if trace_path is None or not trace_path.exists():
        return None
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_value = payload.get("e2e_elapsed_seconds")
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def read_csv_table(path: Path) -> TableData | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return TableData(headers=[], rows=[], width=0)

    headers = list(rows[0])
    data_rows = [list(row) for row in rows[1:]]
    width = max([len(headers), *(len(row) for row in data_rows)] or [0])
    if width == 0:
        return TableData(headers=[], rows=[], width=0)
    headers += [""] * (width - len(headers))
    normalized_rows = [row + [""] * (width - len(row)) for row in data_rows]
    return TableData(headers=headers, rows=normalized_rows, width=width)


def normalize_numeric(text: str) -> str | None:
    if NUMBER_RE.match(text) is None:
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    quantized = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{quantized:.2f}"


def normalize_date_or_datetime(text: str) -> str | None:
    date_match = DATE_RE.fullmatch(text)
    if date_match is not None:
        year, month, day = (int(item) for item in date_match.groups())
        try:
            parsed = date(year, month, day)
        except ValueError:
            return None
        return parsed.isoformat()

    datetime_candidate = text.replace("Z", "+00:00")
    try:
        parsed_dt = datetime.fromisoformat(datetime_candidate)
    except ValueError:
        return None

    if parsed_dt.tzinfo is not None:
        utc_dt = parsed_dt.astimezone(UTC)
        return utc_dt.isoformat().replace("+00:00", "Z")
    return parsed_dt.isoformat()


def normalize_cell(value: str) -> str:
    text = str(value).strip(" \t\r\n")
    if text.lower() in NULL_TOKENS:
        return ""

    normalized_datetime = normalize_date_or_datetime(text)
    if normalized_datetime is not None:
        return normalized_datetime

    normalized_number = normalize_numeric(text)
    if normalized_number is not None:
        return normalized_number

    return text


def canonicalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", header.lower()).strip("_")


def normalize_combined_name(first_name: str, last_name: str) -> str:
    parts = [normalize_cell(first_name), normalize_cell(last_name)]
    return " ".join(part for part in parts if part).strip()


def build_column_signatures(table: TableData | None) -> tuple[list[tuple[str, ...]], list[str]]:
    if table is None or table.width == 0:
        return [], []

    signatures: list[tuple[str, ...]] = []
    headers = table.headers[: table.width]
    canonical_headers = [canonicalize_header(header) for header in headers]
    consumed_columns: set[int] = set()

    first_name_indices = [index for index, header in enumerate(canonical_headers) if header in FIRST_NAME_HEADERS]
    last_name_indices = [index for index, header in enumerate(canonical_headers) if header in LAST_NAME_HEADERS]
    for first_index, last_index in zip(first_name_indices, last_name_indices):
        if first_index in consumed_columns or last_index in consumed_columns:
            continue
        combined_values = [
            normalize_combined_name(row[first_index], row[last_index])
            for row in table.rows
        ]
        signatures.append(tuple(sorted(combined_values)))
        headers[first_index] = f"{headers[first_index]}+{headers[last_index]}"
        consumed_columns.add(first_index)
        consumed_columns.add(last_index)

    rendered_headers: list[str] = []
    for column_index in range(table.width):
        if column_index in consumed_columns:
            if canonical_headers[column_index] in LAST_NAME_HEADERS:
                continue
            rendered_headers.append(headers[column_index])
            continue
        values = [normalize_cell(row[column_index]) for row in table.rows]
        signatures.append(tuple(sorted(values)))
        rendered_headers.append(headers[column_index])
    return signatures, rendered_headers


def infer_question_type(question_en: str, question_zh: str) -> str:
    en = question_en.lower()
    zh = question_zh
    if any(keyword in en for keyword in ["how many", "count", "number of"]) or "多少" in zh:
        return "counting"
    if any(keyword in en for keyword in ["percentage", "percent", "ratio"]) or any(
        keyword in zh for keyword in ["百分比", "占比", "多少倍", "比例"]
    ):
        return "ratio_or_percentage"
    if "average" in en or "平均" in zh:
        return "average"
    if any(keyword in en for keyword in ["lowest", "highest", "best", "fastest", "most"]) or any(
        keyword in zh for keyword in ["最低", "最高", "最佳", "最快", "最多"]
    ):
        return "extremum_lookup"
    if any(keyword in en for keyword in ["list", "identify", "provide", "state", "which", "what are"]) or any(
        keyword in zh for keyword in ["列出", "给出", "说明", "识别", "哪些", "名字是什么"]
    ):
        return "lookup_or_listing"
    return "other"


def parse_modalities(data_formats: str) -> list[str]:
    return [item.strip() for item in data_formats.split(",") if item.strip()]


def evaluate_task(
    task_id: str,
    meta: TaskMeta,
    run_dir: Path,
    gold_root: Path,
    run_summary: dict[str, RunTaskInfo],
    penalty_lambda: float | None,
) -> TaskEvaluation:
    prediction_path = run_dir / task_id / "prediction.csv"
    gold_path = gold_root / task_id / "gold.csv"
    prediction_table = read_csv_table(prediction_path)
    gold_table = read_csv_table(gold_path)
    if gold_table is None:
        raise FileNotFoundError(f"Missing gold.csv for {task_id}: {gold_path}")

    predicted_signatures, predicted_headers = build_column_signatures(prediction_table)
    gold_signatures, gold_headers = build_column_signatures(gold_table)
    gold_counter = Counter(gold_signatures)
    prediction_counter = Counter(predicted_signatures)

    matched_gold_columns = sum(min(gold_counter[signature], prediction_counter[signature]) for signature in gold_counter)
    gold_column_count = len(gold_signatures)
    prediction_column_count = len(predicted_signatures)
    missing_gold_columns = max(gold_column_count - matched_gold_columns, 0)
    extra_prediction_columns = max(prediction_column_count - matched_gold_columns, 0)
    recall = (matched_gold_columns / gold_column_count) if gold_column_count else 1.0
    strict_binary_score = 1.0 if missing_gold_columns == 0 else 0.0
    redundancy_ratio = (
        extra_prediction_columns / prediction_column_count if prediction_column_count > 0 else 0.0
    )
    penalized_score: float | None = None
    if penalty_lambda is not None:
        penalized_score = max(0.0, recall - penalty_lambda * redundancy_ratio)

    run_info = run_summary.get(task_id)
    trace_path = Path(run_info.trace_path) if run_info and run_info.trace_path else None
    elapsed_seconds = read_trace_elapsed_seconds(trace_path)

    if prediction_table is None:
        task_status = "missing_prediction"
    elif strict_binary_score >= 1.0 and extra_prediction_columns == 0:
        task_status = "perfect"
    elif strict_binary_score >= 1.0:
        task_status = "perfect_with_extra_columns"
    else:
        task_status = "gold_not_fully_covered"

    benchmark_failure_reason = run_info.failure_reason if run_info else None
    lowered_failure_reason = (benchmark_failure_reason or "").lower()
    if "connection error" in lowered_failure_reason or "uncaught error" in lowered_failure_reason:
        failure_class = "infra_error"
    elif (
        task_status == "missing_prediction"
        or "did not submit an answer" in lowered_failure_reason
        or "max_steps" in lowered_failure_reason
    ):
        failure_class = "no_answer"
    elif strict_binary_score >= 1.0 and extra_prediction_columns > 0:
        failure_class = "column_redundancy"
    elif strict_binary_score < 1.0:
        failure_class = "semantic_drift"
    else:
        failure_class = "none"

    return TaskEvaluation(
        task_id=task_id,
        difficulty=meta.difficulty,
        data_formats=meta.data_formats,
        question_zh=meta.question_zh,
        question_type=infer_question_type(meta.question_en, meta.question_zh),
        modalities=parse_modalities(meta.data_formats),
        strict_binary_score=strict_binary_score,
        gold_column_count=gold_column_count,
        prediction_column_count=prediction_column_count,
        matched_gold_columns=matched_gold_columns,
        missing_gold_columns=missing_gold_columns,
        extra_prediction_columns=extra_prediction_columns,
        recall=recall,
        redundancy_ratio=redundancy_ratio,
        penalized_score=penalized_score,
        task_status=task_status,
        failure_class=failure_class,
        benchmark_succeeded=run_info.succeeded if run_info else None,
        benchmark_failure_reason=benchmark_failure_reason,
        prediction_exists=prediction_table is not None,
        prediction_path=str(prediction_path) if prediction_table is not None else None,
        gold_path=str(gold_path),
        elapsed_seconds=elapsed_seconds,
        predicted_headers=predicted_headers,
        gold_headers=gold_headers,
    )


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_group(task_results: list[TaskEvaluation]) -> dict[str, Any]:
    strict_scores = [item.strict_binary_score for item in task_results]
    recalls = [item.recall for item in task_results]
    redundancy = [item.redundancy_ratio for item in task_results]
    penalized = [item.penalized_score for item in task_results if item.penalized_score is not None]
    return {
        "task_count": len(task_results),
        "perfect_count": sum(1 for item in task_results if item.strict_binary_score >= 1.0),
        "missing_prediction_count": sum(1 for item in task_results if item.task_status == "missing_prediction"),
        "strict_score_avg": mean(strict_scores),
        "recall_avg": mean(recalls),
        "redundancy_ratio_avg": mean(redundancy),
        "penalized_score_avg": mean(penalized) if penalized else None,
    }


def summarize_failure_classes(task_results: list[TaskEvaluation]) -> dict[str, int]:
    counts = Counter(item.failure_class for item in task_results)
    return {
        "infra_error": int(counts.get("infra_error", 0)),
        "no_answer": int(counts.get("no_answer", 0)),
        "semantic_drift": int(counts.get("semantic_drift", 0)),
        "column_redundancy": int(counts.get("column_redundancy", 0)),
        "none": int(counts.get("none", 0)),
    }


def build_recommendations(task_results: list[TaskEvaluation]) -> list[str]:
    imperfect = [item for item in task_results if item.strict_binary_score < 1.0]
    if not imperfect:
        return ["本次运行在严格榜单口径下所有任务都拿到了满分，没有发现需要优先修复的题型。"]

    recommendations: list[str] = []
    by_difficulty: dict[str, list[TaskEvaluation]] = defaultdict(list)
    by_type: dict[str, list[TaskEvaluation]] = defaultdict(list)
    doc_failures = 0
    timeout_failures = 0
    for item in imperfect:
        by_difficulty[item.difficulty].append(item)
        by_type[item.question_type].append(item)
        if "doc" in item.modalities:
            doc_failures += 1
        if item.benchmark_failure_reason and "timed out" in item.benchmark_failure_reason.lower():
            timeout_failures += 1

    worst_difficulty = max(by_difficulty.items(), key=lambda pair: len(pair[1]))
    recommendations.append(
        f"失分最集中的难度是 `{worst_difficulty[0]}`，共有 {len(worst_difficulty[1])} 个任务未拿满分，建议优先复盘这一组题目的 trace。"
    )

    worst_type = max(by_type.items(), key=lambda pair: len(pair[1]))
    recommendations.append(
        f"失分最多的题型是 `{worst_type[0]}`，共有 {len(worst_type[1])} 个任务失分，可以优先针对这类问题补 prompt 或加专门工具。"
    )

    if doc_failures:
        recommendations.append(
            f"未拿满分的任务里有 {doc_failures} 个含 `doc` 文档上下文，说明长文档阅读与证据抽取仍是当前基线的薄弱项。"
        )
    if timeout_failures:
        recommendations.append(
            f"其中有 {timeout_failures} 个任务是超时导致的，建议优先考虑提升文档类任务的检索效率，或适当提高 `task_timeout_seconds`。"
        )

    return recommendations


def render_summary_table(summary: dict[str, dict[str, Any]], include_penalized: bool) -> list[str]:
    header = "| 分组 | 任务数 | 满分数 | 严格榜单分均值 | 覆盖率均值 | 冗余列比例均值 |"
    if include_penalized:
        header += " 诊断分均值 |"
    separator = "| --- | ---: | ---: | ---: | ---: | ---: |"
    if include_penalized:
        separator += " ---: |"
    lines = [header, separator]
    for group_name, item in summary.items():
        line = (
            f"| {group_name} | {item['task_count']} | {item['perfect_count']} | "
            f"{item['strict_score_avg']:.4f} | {item['recall_avg']:.4f} | {item['redundancy_ratio_avg']:.4f} |"
        )
        if include_penalized:
            penalized = item["penalized_score_avg"]
            rendered = f"{penalized:.4f}" if penalized is not None else "N/A"
            line += f" {rendered} |"
        lines.append(line)
    return lines


def format_seconds(seconds: float | None) -> str:
    if seconds is None or math.isnan(seconds):
        return "-"
    return f"{seconds:.1f}s"


def render_markdown_report(
    *,
    run_dir: Path,
    task_results: list[TaskEvaluation],
    penalty_lambda: float | None,
) -> str:
    overall_summary = summarize_group(task_results)
    include_penalized = penalty_lambda is not None

    by_difficulty: dict[str, list[TaskEvaluation]] = defaultdict(list)
    by_question_type: dict[str, list[TaskEvaluation]] = defaultdict(list)
    by_modality: dict[str, list[TaskEvaluation]] = defaultdict(list)
    for item in task_results:
        by_difficulty[item.difficulty].append(item)
        by_question_type[item.question_type].append(item)
        for modality in item.modalities:
            by_modality[modality].append(item)

    difficulty_summary = {
        key: summarize_group(value)
        for key, value in sorted(by_difficulty.items(), key=lambda pair: pair[0])
    }
    question_type_summary = {
        key: summarize_group(value)
        for key, value in sorted(by_question_type.items(), key=lambda pair: pair[0])
    }
    modality_summary = {
        key: summarize_group(value)
        for key, value in sorted(by_modality.items(), key=lambda pair: pair[0])
    }

    imperfect_tasks = [item for item in task_results if item.strict_binary_score < 1.0]
    extra_column_tasks = [
        item for item in task_results if item.strict_binary_score >= 1.0 and item.extra_prediction_columns > 0
    ]

    lines: list[str] = []
    lines.append(f"# Benchmark 结果分析报告：`{run_dir.name}`")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append(f"- 运行目录：`{run_dir}`")
    lines.append(f"- 任务总数：**{overall_summary['task_count']}**")
    lines.append(f"- 严格榜单口径满分任务数：**{overall_summary['perfect_count']}**")
    lines.append(f"- 严格榜单平均分：**{overall_summary['strict_score_avg']:.4f}**")
    lines.append(
        f"- 平均列覆盖率（召回率）：**{overall_summary['recall_avg']:.4f}**；平均冗余列比例：**{overall_summary['redundancy_ratio_avg']:.4f}**"
    )
    if include_penalized:
        lines.append(
            f"- 本地诊断分（lambda={penalty_lambda:g}）均值：**{overall_summary['penalized_score_avg']:.4f}**"
        )
    else:
        lines.append(
            "- 说明：官方公开说明里明确了严格榜单口径是“二元列匹配准确率”。Section 6.3 的惩罚项 `λ` 未在公开说明中给出固定值，因此本报告默认不伪造该分数，只提供 `recall` 和 `冗余列比例` 作为诊断指标。"
        )
    lines.append("")
    lines.append("## 评分口径")
    lines.append("")
    lines.append("- `严格榜单分`：只有当预测结果完整覆盖全部 gold 列签名时记 1，否则记 0。列名不参与匹配，行顺序也会被忽略。")
    lines.append("- `列覆盖率 recall`：`匹配到的 gold 列数 / gold 列总数`。")
    lines.append("- `冗余列比例`：`额外预测列数 / 预测列总数`。")
    if include_penalized:
        lines.append(
            f"- `本地诊断分`：`recall - λ * 冗余列比例`，其中 `λ={penalty_lambda:g}`，最低截断到 0。这个值用于本地分析，不一定等于官方隐藏评测值。"
        )
    lines.append("")
    lines.append("## 失败分类（Phase A 口径）")
    lines.append("")
    failure_class_summary = summarize_failure_classes(task_results)
    lines.append("| 分类 | 数量 |")
    lines.append("| --- | ---: |")
    lines.append(f"| infra_error | {failure_class_summary['infra_error']} |")
    lines.append(f"| no_answer | {failure_class_summary['no_answer']} |")
    lines.append(f"| semantic_drift | {failure_class_summary['semantic_drift']} |")
    lines.append(f"| column_redundancy | {failure_class_summary['column_redundancy']} |")
    lines.append(f"| none | {failure_class_summary['none']} |")
    lines.append("")
    lines.append("## 按难度汇总")
    lines.append("")
    lines.extend(render_summary_table(difficulty_summary, include_penalized))
    lines.append("")
    lines.append("## 按题型汇总")
    lines.append("")
    lines.extend(render_summary_table(question_type_summary, include_penalized))
    lines.append("")
    lines.append("## 按模态汇总")
    lines.append("")
    lines.extend(render_summary_table(modality_summary, include_penalized))
    lines.append("")
    lines.append("## 未拿满分任务")
    lines.append("")
    if not imperfect_tasks:
        lines.append("- 本次没有未拿满分的任务。")
    else:
        lines.append(
            "| task_id | 难度 | 题型 | 失败分类 | 严格分 | 匹配列/Gold列 | 预测列数 | 冗余列数 | Benchmark状态 | 失败原因 | 中文题意 |"
        )
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |")
        for item in imperfect_tasks:
            benchmark_status = "ok" if item.benchmark_succeeded else "fail"
            failure_reason = item.benchmark_failure_reason or "-"
            lines.append(
                f"| {item.task_id} | {item.difficulty} | {item.question_type} | {item.failure_class} | {item.strict_binary_score:.0f} | "
                f"{item.matched_gold_columns}/{item.gold_column_count} | {item.prediction_column_count} | "
                f"{item.extra_prediction_columns} | {benchmark_status} | {failure_reason} | {item.question_zh} |"
            )
    lines.append("")
    lines.append("## 含冗余列但仍满分的任务")
    lines.append("")
    if not extra_column_tasks:
        lines.append("- 本次没有检测到“多预测列但仍覆盖全部 gold 列”的任务。")
    else:
        lines.append("| task_id | 严格分 | Gold列数 | 预测列数 | 冗余列数 | 中文题意 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for item in extra_column_tasks:
            lines.append(
                f"| {item.task_id} | {item.strict_binary_score:.0f} | {item.gold_column_count} | "
                f"{item.prediction_column_count} | {item.extra_prediction_columns} | {item.question_zh} |"
            )
    lines.append("")
    lines.append("## 建议")
    lines.append("")
    for recommendation in build_recommendations(task_results):
        lines.append(f"- {recommendation}")
    lines.append("")
    lines.append("## 全量任务明细")
    lines.append("")
    lines.append(
        "| task_id | 难度 | 模态 | 题型 | 严格分 | recall | 预测列数 | Gold列数 | 冗余列数 | 运行时长 | 状态 | 中文题意 |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for item in task_results:
        lines.append(
            f"| {item.task_id} | {item.difficulty} | {', '.join(item.modalities)} | {item.question_type} | "
            f"{item.strict_binary_score:.0f} | {item.recall:.2f} | {item.prediction_column_count} | {item.gold_column_count} | "
            f"{item.extra_prediction_columns} | {format_seconds(item.elapsed_seconds)} | {item.task_status} | {item.question_zh} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_root, gold_root, runs_root, task_meta_path = resolve_analysis_roots(args)
    run_dir = resolve_run_dir(args.run_dir, runs_root, args.latest).resolve()
    task_meta = load_task_metadata(input_root, task_meta_path)
    run_summary = load_run_summary(run_dir)

    if args.all_tasks:
        task_ids = sorted(task_meta)
    else:
        task_ids = infer_attempted_task_ids(run_dir, run_summary)
    if not task_ids:
        raise ValueError(f"No attempted task directories found in {run_dir}")
    missing_meta = [task_id for task_id in task_ids if task_id not in task_meta]
    if missing_meta:
        raise KeyError(
            "Missing task metadata for attempted tasks: "
            + ", ".join(missing_meta[:10])
            + ("..." if len(missing_meta) > 10 else "")
        )
    task_results = [
        evaluate_task(
            task_id=task_id,
            meta=task_meta[task_id],
            run_dir=run_dir,
            gold_root=gold_root,
            run_summary=run_summary,
            penalty_lambda=args.penalty_lambda,
        )
        for task_id in task_ids
    ]

    overall_summary = summarize_group(task_results)
    report_payload = {
        "run_dir": str(run_dir),
        "generated_at": datetime.now().astimezone().isoformat(),
        "primary_official_metric": "strict_binary_column_match_accuracy",
        "strict_binary_average_score": overall_summary["strict_score_avg"],
        "average_recall": overall_summary["recall_avg"],
        "average_redundancy_ratio": overall_summary["redundancy_ratio_avg"],
        "penalty_lambda": args.penalty_lambda,
        "average_penalized_score": overall_summary["penalized_score_avg"],
        "task_count": overall_summary["task_count"],
        "perfect_count": overall_summary["perfect_count"],
        "missing_prediction_count": overall_summary["missing_prediction_count"],
        "failure_class_summary": summarize_failure_classes(task_results),
        "tasks": [asdict(item) for item in task_results],
    }

    output_md = args.output_md.resolve() if args.output_md else run_dir / "analysis_report.zh.md"
    output_json = args.output_json.resolve() if args.output_json else run_dir / "analysis_report.json"
    output_md.write_text(
        render_markdown_report(run_dir=run_dir, task_results=task_results, penalty_lambda=args.penalty_lambda),
        encoding="utf-8",
    )
    output_json.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Run dir: {run_dir}")
    print(f"Strict binary average score: {overall_summary['strict_score_avg']:.4f}")
    print(f"Perfect tasks: {overall_summary['perfect_count']}/{overall_summary['task_count']}")
    print(f"Average recall: {overall_summary['recall_avg']:.4f}")
    print(f"Average redundancy ratio: {overall_summary['redundancy_ratio_avg']:.4f}")
    if args.penalty_lambda is not None:
        print(f"Average penalized score (lambda={args.penalty_lambda:g}): {overall_summary['penalized_score_avg']:.4f}")
    print(f"Markdown report: {output_md}")
    print(f"JSON report: {output_json}")


if __name__ == "__main__":
    main()
