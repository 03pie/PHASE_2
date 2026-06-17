from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskAnalysis:
    task_id: str
    runner_succeeded: bool
    strict_match: bool
    contains_gold_answer: bool
    score: int
    failure_class: str
    failure_reason: str | None
    prediction_path: str | None
    gold_path: str | None
    prediction_columns: list[str]
    gold_columns: list[str]
    prediction_rows: int
    gold_rows: int
    trace_steps: int
    model_steps: int
    tool_error_count: int
    repeated_error_count: int
    last_actions: list[str]
    gold_evidence_paths: list[dict[str, Any]]
    trace_observed_paths: list[str]
    gold_path_seen_in_trace: bool
    trace_path_gap: str
    key_findings: list[str]


def _read_csv_table(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return [], []
    return [str(item).strip() for item in rows[0]], [
        [str(item).strip() for item in row] for row in rows[1:]
    ]


def _normalize_table(path: Path) -> tuple[list[str], list[list[str]]]:
    columns, rows = _read_csv_table(path)
    normalized_rows = []
    for row in rows:
        padded = [*row, *([""] * max(0, len(columns) - len(row)))]
        normalized_rows.append(padded[: len(columns)])
    return columns, normalized_rows


def _normalize_cell(value: object) -> str:
    text = str(value if value is not None else "").strip()
    if text == "":
        return ""
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return " ".join(text.casefold().split())
    if number.is_integer():
        return str(int(number))
    return f"{number:.12g}"


def _parse_number(value: object) -> float | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _cells_equal(left: object, right: object) -> bool:
    left_number = _parse_number(left)
    right_number = _parse_number(right)
    if left_number is not None and right_number is not None:
        tolerance = max(1e-8, 1e-10 * max(abs(left_number), abs(right_number), 1.0))
        return abs(left_number - right_number) <= tolerance
    return _normalize_cell(left) == _normalize_cell(right)


def _row_contains_gold_values(prediction_row: list[str], gold_row: list[str]) -> bool:
    unused_prediction_indexes = set(range(len(prediction_row)))
    for gold_value in gold_row:
        match_index = next(
            (
                index
                for index in unused_prediction_indexes
                if _cells_equal(prediction_row[index], gold_value)
            ),
            None,
        )
        if match_index is None:
            return False
        unused_prediction_indexes.remove(match_index)
    return True


def _contains_gold_answer(
    prediction_rows: list[list[str]],
    gold_rows: list[list[str]],
) -> bool:
    if not gold_rows or not prediction_rows:
        return False
    unused_prediction_indexes = set(range(len(prediction_rows)))
    for gold_row in gold_rows:
        match_index = next(
            (
                index
                for index in unused_prediction_indexes
                if _row_contains_gold_values(prediction_rows[index], gold_row)
            ),
            None,
        )
        if match_index is None:
            return False
        unused_prediction_indexes.remove(match_index)
    return True


def _gold_value_set(gold_rows: list[list[str]], *, limit: int = 500) -> set[str]:
    values: set[str] = set()
    for row in gold_rows:
        for value in row:
            normalized = _normalize_cell(value)
            if normalized:
                values.add(normalized)
                if len(values) >= limit:
                    return values
    return values


def _row_signature(row: list[object]) -> tuple[str, ...]:
    return tuple(sorted(_normalize_cell(value) for value in row))


def _context_virtual_path(path: Path, task_id: str) -> str:
    marker = Path("data") / "input" / task_id / "context"
    parts = path.parts
    marker_parts = marker.parts
    for index in range(0, len(parts) - len(marker_parts) + 1):
        if tuple(part.casefold() for part in parts[index : index + len(marker_parts)]) == tuple(
            part.casefold() for part in marker_parts
        ):
            rel = Path(*parts[index + len(marker_parts) :]).as_posix()
            return f"/context/{rel}" if rel else "/context"
    return path.as_posix()


def _score_tabular_rows(
    *,
    columns: list[object],
    rows: list[list[object]],
    gold_columns: list[str],
    gold_rows: list[list[str]],
    gold_values: set[str],
) -> dict[str, Any]:
    source_values: set[str] = set()
    source_row_counts = Counter(_row_signature(row) for row in rows)
    normalized_columns = {_normalize_cell(item) for item in columns}
    row_hits = 0
    for gold_row in gold_rows:
        signature = _row_signature(gold_row)
        if source_row_counts[signature] > 0:
            row_hits += 1
            source_row_counts[signature] -= 1
    for row in rows:
        for value in row:
            normalized = _normalize_cell(value)
            if normalized in gold_values:
                source_values.add(normalized)
    header_hits = sum(
        1
        for column in gold_columns
        if _normalize_cell(column) in normalized_columns
    )
    return {
        "header_hits": header_hits,
        "value_hits": len(source_values),
        "row_hits": row_hits,
        "row_count": len(rows),
    }


def _scan_csv_source(
    path: Path,
    *,
    gold_columns: list[str],
    gold_rows: list[list[str]],
    gold_values: set[str],
) -> dict[str, Any]:
    columns, rows = _read_csv_table(path)
    stats = _score_tabular_rows(
        columns=columns,
        rows=rows,
        gold_columns=gold_columns,
        gold_rows=gold_rows,
        gold_values=gold_values,
    )
    stats["source_type"] = "csv"
    return stats


def _flatten_json_scalars(value: Any) -> list[Any]:
    scalars: list[Any] = []
    if isinstance(value, dict):
        for item in value.values():
            scalars.extend(_flatten_json_scalars(item))
    elif isinstance(value, list):
        for item in value:
            scalars.extend(_flatten_json_scalars(item))
    else:
        scalars.append(value)
    return scalars


def _scan_json_source(
    path: Path,
    *,
    gold_columns: list[str],
    gold_values: set[str],
) -> dict[str, Any]:
    value = _load_json(path)
    keys: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            keys.update(str(key) for key in item)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    scalars = _flatten_json_scalars(value)
    source_values = {
        _normalize_cell(item) for item in scalars if _normalize_cell(item) in gold_values
    }
    header_hits = sum(
        1
        for column in gold_columns
        if _normalize_cell(column) in {_normalize_cell(key) for key in keys}
    )
    return {
        "source_type": "json",
        "header_hits": header_hits,
        "value_hits": len(source_values),
        "row_hits": 0,
        "row_count": None,
    }


def _scan_sqlite_source(
    path: Path,
    *,
    gold_columns: list[str],
    gold_rows: list[list[str]],
    gold_values: set[str],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    gold_probe_values = list(gold_values)[:80]
    normalized_gold_columns = {_normalize_cell(column) for column in gold_columns}
    with sqlite3.connect(path) as conn:
        table_names = [
            str(row[0])
            for row in conn.execute(
                "select name from sqlite_master where type='table' order by name"
            ).fetchall()
        ]
        for table in table_names:
            quoted_table = '"' + table.replace('"', '""') + '"'
            columns = [
                str(row[1])
                for row in conn.execute(f"pragma table_info({quoted_table})").fetchall()
            ]
            normalized_columns = {_normalize_cell(column) for column in columns}
            header_hits = len(normalized_gold_columns & normalized_columns)
            row_count = int(
                conn.execute(f"select count(*) from {quoted_table}").fetchone()[0]
            )
            value_hits = 0
            if columns:
                for gold_value in gold_probe_values:
                    found = False
                    for column in columns:
                        quoted_column = '"' + column.replace('"', '""') + '"'
                        query = (
                            f"select 1 from {quoted_table} "
                            f"where cast({quoted_column} as text) = ? limit 1"
                        )
                        if conn.execute(query, (gold_value,)).fetchone():
                            value_hits += 1
                            found = True
                            break
                    if found:
                        continue
            stats = {
                "header_hits": header_hits,
                "value_hits": value_hits,
                "row_hits": 0,
                "row_count": row_count,
            }
            if stats["value_hits"] or stats["row_hits"] or stats["header_hits"]:
                stats["source_type"] = "sqlite"
                stats["table"] = table
                evidence.append(stats)
    return evidence


def _scan_text_source(
    path: Path,
    *,
    gold_columns: list[str],
    gold_values: set[str],
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    normalized_text = " ".join(text.casefold().split())
    value_hits = sum(1 for value in gold_values if value and value in normalized_text)
    header_hits = sum(
        1 for column in gold_columns if _normalize_cell(column) in normalized_text
    )
    return {
        "source_type": "text",
        "header_hits": header_hits,
        "value_hits": value_hits,
        "row_hits": 0,
        "row_count": None,
    }


def _find_gold_evidence_paths(
    *,
    task_id: str,
    gold_columns: list[str],
    gold_rows: list[list[str]],
    input_root: Path,
    limit: int = 5,
) -> list[dict[str, Any]]:
    context_dir = input_root / task_id / "context"
    if not context_dir.is_dir() or not gold_rows:
        return []
    gold_values = _gold_value_set(gold_rows)
    if not gold_values:
        return []
    evidence: list[dict[str, Any]] = []
    for path in sorted(item for item in context_dir.rglob("*") if item.is_file()):
        suffix = path.suffix.casefold()
        try:
            if suffix == ".csv":
                stats_list = [
                    _scan_csv_source(
                        path,
                        gold_columns=gold_columns,
                        gold_rows=gold_rows,
                        gold_values=gold_values,
                    )
                ]
            elif suffix == ".json":
                stats_list = [
                    _scan_json_source(
                        path,
                        gold_columns=gold_columns,
                        gold_values=gold_values,
                    )
                ]
            elif suffix in {".sqlite", ".db"}:
                stats_list = _scan_sqlite_source(
                    path,
                    gold_columns=gold_columns,
                    gold_rows=gold_rows,
                    gold_values=gold_values,
                )
            elif suffix in {".md", ".txt"}:
                stats_list = [
                    _scan_text_source(
                        path,
                        gold_columns=gold_columns,
                        gold_values=gold_values,
                    )
                ]
            else:
                continue
        except Exception as exc:  # pragma: no cover - best-effort diagnostics
            stats_list = [
                {
                    "source_type": suffix.lstrip(".") or "unknown",
                    "header_hits": 0,
                    "value_hits": 0,
                    "row_hits": 0,
                    "row_count": None,
                    "scan_error": str(exc),
                }
            ]
        for stats in stats_list:
            if not (stats.get("value_hits") or stats.get("row_hits") or stats.get("header_hits")):
                continue
            score = (
                int(stats.get("row_hits") or 0) * 1000
                + int(stats.get("value_hits") or 0) * 10
                + int(stats.get("header_hits") or 0)
            )
            evidence.append(
                {
                    "path": str(path),
                    "virtual_path": _context_virtual_path(path, task_id),
                    "score": score,
                    "gold_value_count": len(gold_values),
                    **stats,
                }
            )
    evidence.sort(
        key=lambda item: (
            item["score"],
            item.get("row_hits") or 0,
            item.get("value_hits") or 0,
        ),
        reverse=True,
    )
    return evidence[:limit]


_CONTEXT_PATH_RE = re.compile(r"/context/[^\s\"'<>),;\]]+")


def _trace_observed_paths(trace: dict[str, Any]) -> list[str]:
    text = json.dumps(trace, ensure_ascii=False, default=str)
    paths = set(_CONTEXT_PATH_RE.findall(text))
    return sorted(paths)


def _gold_path_seen(gold_evidence: list[dict[str, Any]], observed_paths: list[str]) -> bool:
    observed = set(observed_paths)
    for item in gold_evidence[:2]:
        virtual = item.get("virtual_path")
        if isinstance(virtual, str) and virtual in observed:
            return True
    return False


def _trace_path_gap(
    *,
    contains_gold_answer: bool,
    runner_succeeded: bool,
    prediction_rows: list[list[str]],
    gold_evidence: list[dict[str, Any]],
    observed_paths: list[str],
) -> str:
    if contains_gold_answer:
        return "answer_contained"
    if not gold_evidence:
        return "gold_source_not_found_by_scanner"
    if not _gold_path_seen(gold_evidence, observed_paths):
        return "gold_source_not_observed"
    if not prediction_rows and not runner_succeeded:
        return "gold_source_observed_but_no_final_answer"
    return "gold_source_observed_but_wrong_extraction_or_shape"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def _load_summary_or_partial(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        return _load_json(summary_path)

    tasks: list[dict[str, Any]] = []
    for task_dir in sorted(run_dir.glob("task_*"), key=_task_sort_key):
        if not task_dir.is_dir():
            continue
        trace_path = task_dir / "trace.json"
        if not trace_path.is_file():
            continue
        try:
            trace = _load_json(trace_path)
        except (OSError, json.JSONDecodeError):
            trace = {}
        prediction_path = task_dir / "prediction.csv"
        tasks.append(
            {
                "task_id": task_dir.name,
                "task_output_dir": str(task_dir),
                "prediction_csv_path": (
                    str(prediction_path) if prediction_path.is_file() else None
                ),
                "trace_path": str(trace_path),
                "succeeded": bool(trace.get("succeeded")),
                "failure_reason": trace.get("failure_reason"),
            }
        )

    return {
        "run_id": run_dir.name,
        "task_count": len(tasks),
        "succeeded_task_count": sum(1 for item in tasks if item["succeeded"]),
        "partial": True,
        "tasks": tasks,
    }


def _tool_calls(trace: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for step in trace.get("steps") or []:
        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue
        for call in observation.get("tool_calls") or []:
            if isinstance(call, dict):
                calls.append(call)
    return calls


def _compact_error(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    text = " ".join(text.split())
    return text[:500]


def _last_actions(trace: dict[str, Any], limit: int = 8) -> list[str]:
    actions: list[str] = []
    for step in trace.get("steps") or []:
        action = str(step.get("action") or "")
        ok = step.get("ok")
        if action:
            actions.append(f"{action}:{'ok' if ok else 'err'}")
    return actions[-limit:]


def _classify(
    *,
    runner_succeeded: bool,
    strict_match: bool,
    contains_gold_answer: bool,
    failure_reason: str | None,
    pred_columns: list[str],
    gold_columns: list[str],
    pred_rows: list[list[str]],
    gold_rows: list[list[str]],
    tool_errors: list[str],
) -> tuple[str, list[str]]:
    findings: list[str] = []
    if strict_match:
        return "full_score", findings
    if contains_gold_answer:
        return "answer_contained", findings
    if failure_reason:
        findings.append(failure_reason)
        if "uncaught error" in failure_reason:
            return "infra_error", findings
        if "did not prepare an answer" in failure_reason:
            return "no_answer_budget", findings
        return "run_failure", findings
    if not runner_succeeded:
        return "run_failure", findings
    if not pred_columns:
        findings.append("No prediction CSV was produced.")
        return "no_prediction", findings
    missing_columns = [column for column in gold_columns if column not in pred_columns]
    extra_columns = [column for column in pred_columns if column not in gold_columns]
    if missing_columns:
        findings.append(f"Missing gold columns: {missing_columns[:8]}")
    if extra_columns:
        findings.append(f"Extra prediction columns: {extra_columns[:8]}")
    if len(pred_rows) != len(gold_rows):
        findings.append(f"Row count differs: predicted {len(pred_rows)} vs gold {len(gold_rows)}.")
    if pred_columns != gold_columns:
        if set(gold_columns).issubset(set(pred_columns)):
            return "column_redundancy_or_order", findings
        return "wrong_columns", findings
    if len(pred_rows) != len(gold_rows):
        return "wrong_row_count", findings
    return "wrong_values", findings or tool_errors[:3]


def analyze_run(
    run_dir: Path,
    gold_root: Path,
    input_root: Path = Path("data/input"),
) -> tuple[list[TaskAnalysis], dict[str, Any]]:
    summary = _load_summary_or_partial(run_dir)
    analyses: list[TaskAnalysis] = []
    for task in summary.get("tasks") or []:
        task_id = str(task.get("task_id") or "")
        trace_path = Path(task["trace_path"])
        trace = _load_json(trace_path)
        gold_path = gold_root / task_id / "gold.csv"
        prediction_raw = task.get("prediction_csv_path")
        prediction_path = Path(prediction_raw) if prediction_raw else None

        gold_columns: list[str] = []
        gold_rows: list[list[str]] = []
        if gold_path.is_file():
            gold_columns, gold_rows = _normalize_table(gold_path)

        pred_columns: list[str] = []
        pred_rows: list[list[str]] = []
        if prediction_path is not None and prediction_path.is_file():
            pred_columns, pred_rows = _normalize_table(prediction_path)

        strict_match = (
            bool(gold_columns)
            and pred_columns == gold_columns
            and pred_rows == gold_rows
        )
        contains_gold_answer = _contains_gold_answer(pred_rows, gold_rows)
        calls = _tool_calls(trace)
        tool_errors = [
            _compact_error(call.get("result"))
            for call in calls
            if call.get("ok") is False or call.get("status") == "error"
        ]
        repeated_error_count = sum(
            count - 1 for count in Counter(tool_errors).values() if count > 1
        )
        failure_class, findings = _classify(
            runner_succeeded=bool(task.get("succeeded")),
            strict_match=strict_match,
            contains_gold_answer=contains_gold_answer,
            failure_reason=task.get("failure_reason"),
            pred_columns=pred_columns,
            gold_columns=gold_columns,
            pred_rows=pred_rows,
            gold_rows=gold_rows,
            tool_errors=tool_errors,
        )
        if tool_errors:
            findings.append(f"Tool errors observed: {len(tool_errors)}.")
        if repeated_error_count:
            findings.append(f"Repeated equivalent tool errors: {repeated_error_count}.")

        steps = trace.get("steps") or []
        gold_evidence = (
            []
            if contains_gold_answer
            else _find_gold_evidence_paths(
                task_id=task_id,
                gold_columns=gold_columns,
                gold_rows=gold_rows,
                input_root=input_root,
            )
        )
        observed_paths = [] if contains_gold_answer else _trace_observed_paths(trace)
        gold_seen = _gold_path_seen(gold_evidence, observed_paths)
        path_gap = _trace_path_gap(
            contains_gold_answer=contains_gold_answer,
            runner_succeeded=bool(task.get("succeeded")),
            prediction_rows=pred_rows,
            gold_evidence=gold_evidence,
            observed_paths=observed_paths,
        )
        if not contains_gold_answer:
            top_evidence = gold_evidence[0] if gold_evidence else None
            if top_evidence:
                source_label = top_evidence["virtual_path"]
                if top_evidence.get("table"):
                    source_label += f"::{top_evidence['table']}"
                findings.append(f"Top gold evidence: {source_label}.")
            findings.append(f"Trace path gap: {path_gap}.")
        analyses.append(
            TaskAnalysis(
                task_id=task_id,
                runner_succeeded=bool(task.get("succeeded")),
                strict_match=strict_match,
                contains_gold_answer=contains_gold_answer,
                score=1 if contains_gold_answer else 0,
                failure_class=failure_class,
                failure_reason=task.get("failure_reason"),
                prediction_path=str(prediction_path) if prediction_path else None,
                gold_path=str(gold_path) if gold_path.is_file() else None,
                prediction_columns=pred_columns,
                gold_columns=gold_columns,
                prediction_rows=len(pred_rows),
                gold_rows=len(gold_rows),
                trace_steps=len(steps),
                model_steps=sum(1 for step in steps if step.get("action") == "model"),
                tool_error_count=len(tool_errors),
                repeated_error_count=repeated_error_count,
                last_actions=_last_actions(trace),
                gold_evidence_paths=gold_evidence,
                trace_observed_paths=observed_paths,
                gold_path_seen_in_trace=gold_seen,
                trace_path_gap=path_gap,
                key_findings=findings[:8],
            )
        )

    aggregate = {
        "run_dir": str(run_dir),
        "task_count": len(analyses),
        "runner_succeeded_count": sum(item.runner_succeeded for item in analyses),
        "strict_full_score_count": sum(item.strict_match for item in analyses),
        "answer_contained_count": sum(item.contains_gold_answer for item in analyses),
        "answer_contained_average": (
            sum(item.score for item in analyses) / len(analyses) if analyses else 0.0
        ),
        "strict_binary_average": (
            sum(1 for item in analyses if item.strict_match) / len(analyses)
            if analyses
            else 0.0
        ),
        "failure_classes": dict(Counter(item.failure_class for item in analyses)),
        "trace_path_gaps": dict(Counter(item.trace_path_gap for item in analyses)),
    }
    return analyses, aggregate


def _write_report(
    *,
    run_dir: Path,
    analyses: list[TaskAnalysis],
    aggregate: dict[str, Any],
) -> None:
    json_path = run_dir / "analysis_report.json"
    json_path.write_text(
        json.dumps(
            {
                "aggregate": aggregate,
                "tasks": [asdict(item) for item in analyses],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"# Benchmark Analysis - {run_dir.name}",
        "",
        "## Summary",
        "",
        f"- Tasks: {aggregate['task_count']}",
        f"- Runner succeeded: {aggregate['runner_succeeded_count']}",
        f"- Strict full-score tasks: {aggregate['strict_full_score_count']}",
        f"- Answer-contained tasks: {aggregate['answer_contained_count']}",
        f"- Answer-contained average: {aggregate['answer_contained_average']:.4f}",
        f"- Strict binary average: {aggregate['strict_binary_average']:.4f}",
        f"- Failure classes: {aggregate['failure_classes']}",
        f"- Trace path gaps: {aggregate['trace_path_gaps']}",
        "",
        "## Not Answer-Contained Tasks",
        "",
        "| Task | Class | Runner OK | Pred rows/cols | Gold rows/cols | Top gold evidence | Trace saw it | Gap | Findings |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for item in analyses:
        if item.contains_gold_answer:
            continue
        finding = "<br>".join(item.key_findings) if item.key_findings else ""
        top = item.gold_evidence_paths[0] if item.gold_evidence_paths else {}
        top_label = str(top.get("virtual_path") or "")
        if top.get("table"):
            top_label += f"::{top['table']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    item.task_id,
                    item.failure_class,
                    str(item.runner_succeeded),
                    f"{item.prediction_rows}/{len(item.prediction_columns)}",
                    f"{item.gold_rows}/{len(item.gold_columns)}",
                    top_label.replace("|", "\\|"),
                    str(item.gold_path_seen_in_trace),
                    item.trace_path_gap,
                    finding.replace("|", "\\|"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Answer-Contained Tasks",
            "",
            ", ".join(item.task_id for item in analyses if item.contains_gold_answer)
            or "(none)",
            "",
            "## Strict Full-Score Tasks",
            "",
            ", ".join(item.task_id for item in analyses if item.strict_match)
            or "(none)",
            "",
        ]
    )
    (run_dir / "analysis_report.zh.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--gold-root", type=Path, default=Path("data/output"))
    parser.add_argument("--input-root", type=Path, default=Path("data/input"))
    args = parser.parse_args()
    analyses, aggregate = analyze_run(args.run_dir, args.gold_root, args.input_root)
    _write_report(run_dir=args.run_dir, analyses=analyses, aggregate=aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
