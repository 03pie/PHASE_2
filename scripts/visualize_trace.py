from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a trace.json file into a standalone HTML report.",
    )
    parser.add_argument("trace_path", type=Path, help="Path to trace.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path. Defaults to trace_report.html next to trace.json.",
    )
    return parser.parse_args()


def load_trace(trace_path: Path) -> dict[str, Any]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("trace.json root must be a JSON object.")
    return payload


def dump_json(value: Any) -> str:
    return escape(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False))


def short_text(value: Any, *, limit: int = 160) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=False)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def summarize_observation(observation: dict[str, Any] | None) -> str:
    if not isinstance(observation, dict):
        return "No observation"
    tool_name = observation.get("tool", "unknown")
    ok = observation.get("ok")
    parts = [f"tool={tool_name}"]
    if ok is not None:
        parts.append(f"ok={ok}")
    content = observation.get("content")
    if isinstance(content, dict):
        if "path" in content:
            parts.append(f"path={content['path']}")
        elif "root" in content:
            parts.append(f"root={content['root']}")
        preview = content.get("preview")
        output = content.get("output")
        if isinstance(preview, str) and preview.strip():
            parts.append(short_text(preview, limit=90))
        elif isinstance(output, str) and output.strip():
            parts.append(short_text(output, limit=90))
    return " | ".join(parts)


def render_answer_table(answer: Any) -> str:
    if not isinstance(answer, dict):
        return '<div class="empty-state">No answer payload in trace.</div>'

    columns = answer.get("columns")
    rows = answer.get("rows")
    if not isinstance(columns, list) or not columns:
        return '<div class="empty-state">Answer payload exists, but no columns were found.</div>'

    header_html = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body_rows: list[str] = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, list):
                cells = "".join(f"<td>{escape(str(cell))}</td>" for cell in row)
                body_rows.append(f"<tr>{cells}</tr>")
    if not body_rows:
        body_rows.append(
            f'<tr><td colspan="{len(columns)}" class="empty-cell">No answer rows</td></tr>'
        )

    return (
        '<div class="table-shell">'
        '<table>'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def render_action_bars(steps: list[dict[str, Any]]) -> str:
    counts = Counter(str(step.get("action", "unknown")) for step in steps)
    if not counts:
        return '<div class="empty-state">No steps available.</div>'
    max_count = max(counts.values())
    items: list[str] = []
    for action, count in counts.most_common():
        width = 12 if max_count == 0 else max(12, int((count / max_count) * 100))
        items.append(
            "<div class=\"bar-row\">"
            f"<div class=\"bar-label\">{escape(action)}</div>"
            "<div class=\"bar-track\">"
            f"<div class=\"bar-fill\" style=\"width:{width}%\"></div>"
            "</div>"
            f"<div class=\"bar-value\">{count}</div>"
            "</div>"
        )
    return "".join(items)


def step_status_text(step_ok: Any) -> str:
    if step_ok is True:
        return "success"
    if step_ok is False:
        return "failed"
    return "unknown"


def should_open_step(step_index: Any, step_ok: Any) -> bool:
    if step_ok is False:
        return True
    try:
        return int(step_index) <= 2
    except (TypeError, ValueError):
        return False


def render_subnode(*, label: str, hint: str | None, body_html: str, extra_class: str = "") -> str:
    classes = ["subnode"]
    if extra_class:
        classes.append(extra_class)
    hint_html = f'<span class="subnode-hint">{escape(hint)}</span>' if hint else ""
    return (
        f'<section class="{" ".join(classes)}">'
        '<div class="subnode-card">'
        '<div class="subnode-top">'
        f'<span class="subnode-label">{escape(label)}</span>'
        f"{hint_html}"
        "</div>"
        f"{body_html}"
        "</div>"
        "</section>"
    )


def render_payload_subnode(*, label: str, hint: str | None, value: Any, summary_label: str) -> str:
    return render_subnode(
        label=label,
        hint=hint,
        extra_class="payload-node",
        body_html=(
            '<details class="payload-block">'
            f'<summary>{escape(summary_label)}</summary>'
            f"<pre>{dump_json(value)}</pre>"
            "</details>"
        ),
    )


def extract_json_payload(raw_value: Any) -> Any | None:
    if isinstance(raw_value, dict | list):
        return raw_value
    if not isinstance(raw_value, str):
        return None

    trimmed = raw_value.strip()
    if trimmed.startswith("```") and trimmed.endswith("```"):
        lines = trimmed.splitlines()
        if len(lines) >= 2:
            trimmed = "\n".join(lines[1:-1]).strip()
    if not trimmed:
        return None

    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        return None


def render_text_payload_subnode(*, label: str, hint: str | None, text: str, summary_label: str) -> str:
    return render_subnode(
        label=label,
        hint=hint,
        extra_class="response-node",
        body_html=(
            '<details class="payload-block">'
            f'<summary>{escape(summary_label)}</summary>'
            f"<pre>{escape(text)}</pre>"
            "</details>"
        ),
    )


def render_raw_response_subnode(raw_response: Any) -> str:
    parsed_payload = extract_json_payload(raw_response)
    if parsed_payload is not None:
        return render_payload_subnode(
            label="Raw Response",
            hint="assistant raw message",
            value=parsed_payload,
            summary_label="查看格式化 JSON",
        )

    return render_text_payload_subnode(
        label="Raw Response",
        hint=short_text(raw_response, limit=120),
        text=str(raw_response),
        summary_label="查看原始响应",
    )


def render_step_node(step: dict[str, Any]) -> str:
    step_index = step.get("step_index", "?")
    action = escape(str(step.get("action", "unknown")))
    raw_thought = step.get("thought")
    thought_text = str(raw_thought).strip() if raw_thought is not None else ""
    if not thought_text:
        thought_text = "No thought recorded."
    raw_response = step.get("raw_response")
    action_input = step.get("action_input")
    observation = step.get("observation")
    step_ok = step.get("ok")
    status_class = "ok" if step_ok is True else "warn" if step_ok is False else "neutral"
    status_text = step_status_text(step_ok)
    observation_summary = summarize_observation(observation if isinstance(observation, dict) else None)
    headline = escape(short_text(thought_text, limit=220))

    subnodes = [
        render_subnode(
            label="Thought",
            hint="assistant reasoning",
            body_html=f'<div class="subnode-text">{escape(thought_text)}</div>',
            extra_class="thought-node",
        )
    ]
    if action_input is not None:
        subnodes.append(
            render_payload_subnode(
                label="Action Input",
                hint=short_text(action_input, limit=120) or "Structured request",
                value=action_input,
                summary_label="查看 JSON",
            )
        )
    if isinstance(observation, dict):
        subnodes.append(
            render_payload_subnode(
                label="Observation",
                hint=short_text(observation_summary, limit=160),
                value=observation,
                summary_label="查看 observation",
            )
        )
    if raw_response is not None:
      subnodes.append(render_raw_response_subnode(raw_response))

    search_text = escape(
        " ".join(
            [
                str(step.get("step_index", "")),
                str(step.get("thought", "")),
                str(step.get("action", "")),
                short_text(step.get("action_input"), limit=240),
                short_text(observation, limit=240),
            ]
        ).lower()
    )

    open_attr = " open" if should_open_step(step_index, step_ok) else ""

    return (
        f'<details class="step-node {status_class}" data-status="{status_class}" data-search="{search_text}"{open_attr}>'
        "<summary class=\"step-summary-row\">"
        "<div class=\"step-shell\">"
        "<div class=\"step-meta-top\">"
        f"<span class=\"step-number\">Step {escape(str(step_index))}</span>"
        "<span class=\"step-kind\">Agent Node</span>"
        "</div>"
        "<div class=\"step-title-row\">"
        f"<span class=\"step-action\">{action}</span>"
        f"<span class=\"badge {status_class}\">{escape(status_text)}</span>"
        "</div>"
        f"<div class=\"step-headline\">{headline}</div>"
        "</div>"
        "</summary>"
        "<div class=\"step-body\">"
        f"<div class=\"step-subnodes\">{''.join(subnodes)}</div>"
        "</div>"
        "</details>"
    )


def build_html(trace_path: Path, trace_payload: dict[str, Any]) -> str:
    steps = trace_payload.get("steps")
    if not isinstance(steps, list):
        steps = []
    normalized_steps = [step for step in steps if isinstance(step, dict)]

    answer = trace_payload.get("answer")
    task_id = escape(str(trace_payload.get("task_id", trace_path.parent.name)))
    total_steps = len(normalized_steps)
    ok_steps = sum(1 for step in normalized_steps if step.get("ok") is True)
    failed_steps = sum(1 for step in normalized_steps if step.get("ok") is False)
    answer_rows = 0
    answer_columns = 0
    if isinstance(answer, dict):
        rows = answer.get("rows")
        columns = answer.get("columns")
        if isinstance(rows, list):
            answer_rows = len(rows)
        if isinstance(columns, list):
            answer_columns = len(columns)

    summary_cards = "".join(
        [
            f'<section class="metric-card"><div class="metric-label">Task</div><div class="metric-value">{task_id}</div></section>',
            f'<section class="metric-card"><div class="metric-label">Steps</div><div class="metric-value">{total_steps}</div></section>',
            f'<section class="metric-card"><div class="metric-label">Successful</div><div class="metric-value">{ok_steps}</div></section>',
            f'<section class="metric-card"><div class="metric-label">Failed</div><div class="metric-value">{failed_steps}</div></section>',
            f'<section class="metric-card"><div class="metric-label">Answer Shape</div><div class="metric-value">{answer_rows} x {answer_columns}</div></section>',
        ]
    )

    steps_html = "".join(render_step_node(step) for step in normalized_steps)
    generated_at = escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    source_path = escape(str(trace_path))
    payload_preview = dump_json(
        {
            "task_id": trace_payload.get("task_id"),
            "answer": trace_payload.get("answer"),
            "step_count": total_steps,
        }
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trace Report · {task_id}</title>
  <style>
    :root {{
      --paper: #f7f2e8;
      --ink: #1d1b19;
      --muted: #6d665d;
      --panel: rgba(255, 252, 247, 0.82);
      --border: rgba(40, 33, 25, 0.12);
      --accent: #ba5b3d;
      --accent-2: #2f6c64;
      --shadow: 0 18px 48px rgba(39, 27, 15, 0.12);
      --ok: #2d6a4f;
      --warn: #9f3a2e;
      --neutral: #876f48;
      --mono: Consolas, "SFMono-Regular", "Liberation Mono", Menlo, monospace;
      --sans: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      --serif: Georgia, "Noto Serif SC", "Songti SC", serif;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(186, 91, 61, 0.16), transparent 26%),
        radial-gradient(circle at top right, rgba(47, 108, 100, 0.14), transparent 22%),
        linear-gradient(180deg, #efe3cc 0%, var(--paper) 22%, #f4efe8 100%);
      min-height: 100vh;
    }}

    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(29, 27, 25, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(29, 27, 25, 0.03) 1px, transparent 1px);
      background-size: 32px 32px;
      mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.8), transparent 92%);
    }}

    .page {{
      width: min(1400px, calc(100vw - 40px));
      margin: 24px auto 48px;
      position: relative;
      z-index: 1;
    }}

    .hero {{
      display: grid;
      gap: 18px;
      grid-template-columns: 1.3fr 0.7fr;
      align-items: stretch;
      margin-bottom: 20px;
    }}

    .panel {{
      background: var(--panel);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}

    .hero-main {{
      padding: 28px;
      overflow: hidden;
      position: relative;
    }}

    .hero-main::after {{
      content: "TRACE";
      position: absolute;
      right: -22px;
      top: -8px;
      font-family: var(--serif);
      font-size: clamp(68px, 12vw, 150px);
      color: rgba(186, 91, 61, 0.08);
      letter-spacing: 0.06em;
      line-height: 1;
    }}

    .eyebrow {{
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 12px;
      margin-bottom: 12px;
      font-weight: 700;
    }}

    h1 {{
      margin: 0;
      font-family: var(--serif);
      font-size: clamp(32px, 5vw, 56px);
      line-height: 0.95;
      max-width: 12ch;
    }}

    .subtitle {{
      margin: 14px 0 0;
      color: var(--muted);
      max-width: 70ch;
      line-height: 1.6;
      font-size: 15px;
    }}

    .hero-meta {{
      margin-top: 22px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .chip, .meta-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(40, 33, 25, 0.08);
      color: var(--ink);
      font-size: 13px;
    }}

    .hero-side {{
      padding: 24px;
      display: grid;
      gap: 16px;
    }}

    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}

    .metric-card {{
      padding: 16px 18px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.75);
      border: 1px solid rgba(40, 33, 25, 0.08);
      min-height: 92px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}

    .metric-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}

    .metric-value {{
      font-size: 28px;
      font-weight: 700;
      line-height: 1.05;
      word-break: break-word;
    }}

    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.75fr);
      gap: 20px;
      align-items: start;
    }}

    .section {{
      padding: 22px;
      margin-bottom: 20px;
    }}

    .section-title {{
      margin: 0 0 14px;
      font-size: 18px;
      letter-spacing: 0.02em;
    }}

    .section-note {{
      margin: -4px 0 14px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }}

    .table-shell {{
      overflow: auto;
      border-radius: 18px;
      border: 1px solid rgba(40, 33, 25, 0.08);
      background: rgba(255, 255, 255, 0.72);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 420px;
    }}

    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(40, 33, 25, 0.08);
      text-align: left;
      vertical-align: top;
    }}

    th {{
      background: rgba(47, 108, 100, 0.08);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }}

    tr:last-child td {{
      border-bottom: 0;
    }}

    .empty-state, .empty-cell {{
      color: var(--muted);
      padding: 18px;
    }}

    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
      align-items: center;
    }}

    .toolbar input {{
      flex: 1 1 240px;
      min-height: 42px;
      padding: 10px 14px;
      border-radius: 14px;
      border: 1px solid rgba(40, 33, 25, 0.12);
      background: rgba(255, 255, 255, 0.9);
      font: inherit;
    }}

    .toolbar label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      color: var(--muted);
    }}

    .steps-grid {{
      display: grid;
      gap: 12px;
    }}

    .step-node {{
      position: relative;
      padding-left: 42px;
    }}

    .step-node::before {{
      content: "";
      position: absolute;
      left: 17px;
      top: 18px;
      bottom: -14px;
      width: 2px;
      background: linear-gradient(180deg, rgba(47, 108, 100, 0.26), rgba(40, 33, 25, 0.08));
    }}

    .step-node:last-child::before {{
      display: none;
    }}

    .step-node > summary {{
      list-style: none;
      cursor: pointer;
      position: relative;
      padding-right: 42px;
    }}

    .step-node > summary::-webkit-details-marker {{
      display: none;
    }}

    .step-node > summary::before {{
      content: "";
      position: absolute;
      left: -33px;
      top: 24px;
      width: 14px;
      height: 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 0 0 6px rgba(255, 252, 247, 0.95);
      border: 3px solid rgba(135, 111, 72, 0.78);
    }}

    .step-node.ok > summary::before {{
      border-color: var(--ok);
    }}

    .step-node.warn > summary::before {{
      border-color: var(--warn);
    }}

    .step-node.neutral > summary::before {{
      border-color: var(--neutral);
    }}

    .step-node > summary::after {{
      content: "▸";
      position: absolute;
      right: 14px;
      top: 18px;
      color: var(--muted);
      font-size: 18px;
      transition: transform 120ms ease;
    }}

    .step-node[open] > summary::after {{
      transform: rotate(90deg);
    }}

    .step-shell {{
      padding: 16px 18px;
      border-radius: 22px;
      border: 1px solid rgba(40, 33, 25, 0.1);
      background: rgba(255, 255, 255, 0.78);
      box-shadow: 0 8px 24px rgba(39, 27, 15, 0.08);
      transition: border-color 120ms ease, transform 120ms ease, box-shadow 120ms ease;
    }}

    .step-node[open] .step-shell {{
      border-color: rgba(47, 108, 100, 0.22);
      box-shadow: 0 14px 30px rgba(39, 27, 15, 0.12);
    }}

    .step-summary-row:hover .step-shell {{
      transform: translateY(-1px);
    }}

    .step-meta-top {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
    }}

    .step-number {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.15em;
      color: var(--muted);
    }}

    .step-kind {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--accent-2);
      font-weight: 700;
    }}

    .step-title-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }}

    .step-action {{
      font-weight: 700;
      font-size: 18px;
      word-break: break-word;
    }}

    .badge {{
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
      color: white;
    }}

    .badge.ok {{ background: var(--ok); }}
    .badge.warn {{ background: var(--warn); }}
    .badge.neutral {{ background: var(--neutral); }}

    .step-headline {{
      margin: 0 0 12px;
      font-size: 14px;
      line-height: 1.65;
    }}

    .step-facts {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}

    .node-pill {{
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(247, 242, 232, 0.96);
      border: 1px solid rgba(40, 33, 25, 0.08);
      color: var(--muted);
      font-size: 12px;
    }}

    .step-body {{
      margin: 8px 0 4px 14px;
      padding-left: 18px;
      border-left: 2px solid rgba(40, 33, 25, 0.08);
    }}

    .step-subnodes {{
      display: grid;
      gap: 10px;
    }}

    .subnode {{
      position: relative;
    }}

    .subnode::before {{
      content: "";
      position: absolute;
      left: -25px;
      top: 18px;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: rgba(47, 108, 100, 0.85);
      box-shadow: 0 0 0 4px rgba(255, 252, 247, 0.95);
    }}

    .subnode.payload-node::before {{
      background: rgba(186, 91, 61, 0.78);
    }}

    .subnode.response-node::before {{
      background: rgba(135, 111, 72, 0.82);
    }}

    .subnode-card {{
      border-radius: 18px;
      border: 1px solid rgba(40, 33, 25, 0.08);
      background: rgba(255, 255, 255, 0.66);
      overflow: hidden;
    }}

    .subnode-top {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
      padding: 12px 14px 10px;
      border-bottom: 1px solid rgba(40, 33, 25, 0.06);
      background: rgba(247, 242, 232, 0.7);
    }}

    .subnode-label {{
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent-2);
    }}

    .subnode-hint {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
    }}

    .subnode-text {{
      padding: 14px;
      font-size: 14px;
      line-height: 1.7;
    }}

    .payload-block {{
      border-top: 1px solid rgba(40, 33, 25, 0.06);
    }}

    .payload-block > summary {{
      list-style: none;
      cursor: pointer;
      padding: 10px 14px;
      font-weight: 700;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
      background: rgba(255, 255, 255, 0.72);
    }}

    .payload-block > summary::-webkit-details-marker {{
      display: none;
    }}

    pre {{
      margin: 0;
      padding: 14px;
      overflow: auto;
      font-size: 12px;
      line-height: 1.55;
      font-family: var(--mono);
      white-space: pre-wrap;
      word-break: break-word;
    }}

    .bar-row + .bar-row {{
      margin-top: 12px;
    }}

    .bar-row {{
      display: grid;
      grid-template-columns: minmax(110px, 150px) minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
    }}

    .bar-label {{
      font-size: 13px;
      font-weight: 600;
      word-break: break-word;
    }}

    .bar-track {{
      height: 13px;
      border-radius: 999px;
      background: rgba(40, 33, 25, 0.08);
      overflow: hidden;
    }}

    .bar-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), #d18d4b 60%, #ecc86f 100%);
    }}

    .bar-value {{
      font-weight: 700;
      min-width: 24px;
      text-align: right;
    }}

    .footer-note {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }}

    @media (max-width: 1100px) {{
      .hero, .layout {{
        grid-template-columns: 1fr;
      }}
    }}

    @media (max-width: 720px) {{
      .page {{
        width: min(100vw - 18px, 100%);
        margin: 10px auto 28px;
      }}

      .hero-main, .hero-side, .section {{
        padding: 16px;
        border-radius: 18px;
      }}

      .metric-grid {{
        grid-template-columns: 1fr 1fr;
      }}

      .step-node {{
        padding-left: 34px;
      }}

      .step-node::before {{
        left: 13px;
      }}

      .step-node > summary::before {{
        left: -29px;
        top: 22px;
      }}

      .step-title-row {{
        flex-direction: column;
        align-items: flex-start;
      }}

      .step-body {{
        margin-left: 10px;
        padding-left: 14px;
      }}

      .bar-row {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="panel hero-main">
        <div class="eyebrow">Trace Visual Report</div>
        <h1>{task_id}</h1>
        <p class="subtitle">把 trace.json 中的答案结果、推理步骤、工具调用和原始观测整理成一页 HTML，便于排查任务过程、比较步骤质量，以及快速定位异常观测。</p>
        <div class="hero-meta">
          <span class="chip">source: {source_path}</span>
          <span class="chip">generated: {generated_at}</span>
        </div>
      </div>
      <aside class="panel hero-side">
        <h2 class="section-title">Payload Preview</h2>
        <pre>{payload_preview}</pre>
      </aside>
    </section>

    <div class="metric-grid">
      {summary_cards}
    </div>

    <div class="layout">
      <main>
        <section class="panel section">
          <h2 class="section-title">Final Answer</h2>
          <p class="section-note">如果 trace 中包含 answer.columns 和 answer.rows，这里会以表格形式展示最终输出。</p>
          {render_answer_table(answer)}
        </section>

        <section class="panel section">
          <h2 class="section-title">Step Timeline</h2>
          <p class="section-note">按执行顺序展示每一步的 thought、action、action_input、observation 和 raw_response。可以用搜索框快速筛选。</p>
          <div class="toolbar">
            <input id="searchBox" type="search" placeholder="搜索 thought / action / observation ...">
            <label><input id="hideOk" type="checkbox"> 隐藏全部成功步骤</label>
          </div>
          <div id="stepsGrid" class="steps-grid">
            {steps_html or '<div class="empty-state">No steps found.</div>'}
          </div>
        </section>
      </main>

      <aside>
        <section class="panel section">
          <h2 class="section-title">Action Breakdown</h2>
          <p class="section-note">统计每种 action 在当前 trace 中出现的次数，方便判断代理主要把时间花在什么操作上。</p>
          {render_action_bars(normalized_steps)}
        </section>

        <section class="panel section">
          <h2 class="section-title">Raw JSON</h2>
          <p class="section-note">用于和可视化结果对照，避免隐藏字段被漏看。</p>
          <pre>{dump_json(trace_payload)}</pre>
        </section>

        <section class="panel section">
          <h2 class="section-title">Usage</h2>
          <pre>python scripts/visualize_trace.py path/to/trace.json
python scripts/visualize_trace.py path/to/trace.json -o report.html</pre>
          <p class="footer-note">脚本仅依赖 Python 标准库，默认会在 trace.json 同目录生成 trace_report.html。</p>
        </section>
      </aside>
    </div>
  </div>

  <script>
    const searchBox = document.getElementById('searchBox');
    const hideOk = document.getElementById('hideOk');
    const cards = Array.from(document.querySelectorAll('.step-node'));

    function applyFilters() {{
      const query = (searchBox.value || '').trim().toLowerCase();
      const hideSuccessful = hideOk.checked;
      for (const card of cards) {{
        const haystack = card.dataset.search || '';
        const matchesQuery = !query || haystack.includes(query);
        const isSuccessful = card.dataset.status === 'ok';
        const visible = matchesQuery && !(hideSuccessful && isSuccessful);
        card.style.display = visible ? '' : 'none';
        if (visible && query) {{
          card.open = true;
        }}
      }}
    }}

    searchBox.addEventListener('input', applyFilters);
    hideOk.addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    trace_path = args.trace_path.resolve()
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    output_path = args.output.resolve() if args.output else trace_path.with_name("trace_report.html")
    trace_payload = load_trace(trace_path)
    html = build_html(trace_path, trace_payload)
    output_path.write_text(html, encoding="utf-8")
    print(f"HTML report written to: {output_path}")


if __name__ == "__main__":
    main()