from __future__ import annotations

import json
import re
from pathlib import Path

REMOVE_PREFIXES = (
    "Database ID:",
    "Domain:",
    "Classification:",
    "Version:",
    "Effective Date:",
    "Author:",
    "Created:",
    "Updated:",
    "Modified:",
    "Source:",
    "Tags:",
    "Page:",
    "页码:",
    "页眉:",
    "页脚:",
    "Header:",
    "Footer:",
    "Confidential",
    "Copyright",
    "©",
)

REMOVE_PATTERNS = (
    r"^\s*Page\s+\d+(\s*of\s*\d+)?\s*$",
    r"^\s*第\s*\d+\s*页(\s*共\s*\d+\s*页)?\s*$",
    r"^\s*[-—_]{3,}\s*$",
    r"^\s*\d+\s*/\s*\d+\s*$",
)

INVALID_KNOWLEDGE_MARKERS = (
    "Step3 skipped",
    "evidence 为空",
    "未生成 knowledge guide",
    "no evidence",
)


def _is_removed_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines and all(any(line.startswith(prefix) for prefix in REMOVE_PREFIXES) for line in lines):
        return True
    return any(re.match(pattern, stripped, flags=re.IGNORECASE) for pattern in REMOVE_PATTERNS)


def _clean_markdown(text: str) -> str:
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [_clean_markdown(cell.strip()) for cell in stripped.split("|")]


def _is_table_separator(line: str) -> bool:
    cells = _split_table_row(line)
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells
    )


def _is_table_line(line: str) -> bool:
    return "|" in line and len(_split_table_row(line)) > 1


def _current_path(headings: list[tuple[int, str]]) -> str:
    titles = [title for _, title in headings if not title.startswith("Knowledge Guide:")]
    return " > ".join(titles)


def _append_entry(
    entries: list[dict[str, str]],
    *,
    path: str,
    entry_type: str,
    text: str,
) -> None:
    text = _clean_markdown(text)
    if not text or _is_removed_text(text):
        return
    entries.append(
        {
            "id": f"{len(entries) + 1:04d}",
            "path": path,
            "type": entry_type,
            "text": text,
        }
    )


def markdown_knowledge_to_schema(markdown_text: str) -> list[dict[str, str]]:
    """Convert a task knowledge guide into compact schema entries."""
    if not markdown_text.strip():
        return []
    if any(marker.lower() in markdown_text.lower() for marker in INVALID_KNOWLEDGE_MARKERS):
        return []

    entries: list[dict[str, str]] = []
    headings: list[tuple[int, str]] = []
    paragraph_lines: list[str] = []
    lines = markdown_text.splitlines()
    index = 0

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = " ".join(paragraph_lines)
        paragraph_lines.clear()
        _append_entry(
            entries,
            path=_current_path(headings),
            entry_type="content",
            text=text,
        )

    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip()

        if not line:
            flush_paragraph()
            index += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            title = _clean_markdown(heading_match.group(2))
            if not _is_removed_text(title):
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, title))
            index += 1
            continue

        if (
            _is_table_line(line)
            and index + 1 < len(lines)
            and _is_table_separator(lines[index + 1])
        ):
            flush_paragraph()
            headers = _split_table_row(line)
            index += 2
            while index < len(lines) and _is_table_line(lines[index].strip()):
                if _is_table_separator(lines[index]):
                    index += 1
                    continue
                row = _split_table_row(lines[index])
                pairs = [
                    f"{header}: {row[column_index] if column_index < len(row) else ''}"
                    for column_index, header in enumerate(headers)
                    if header
                ]
                path = _current_path(headings)
                title = headings[-1][1] if headings else path
                prefix = f"In {title}, " if title else ""
                _append_entry(
                    entries,
                    path=path,
                    entry_type="table_row",
                    text=prefix + ", ".join(pairs),
                )
                index += 1
            continue

        list_match = re.match(r"^\s*(?:[-*+]|\d+[.)])\s+(.+)$", raw_line)
        if list_match:
            flush_paragraph()
            _append_entry(
                entries,
                path=_current_path(headings),
                entry_type="list_item",
                text=list_match.group(1),
            )
            index += 1
            continue

        if _is_removed_text(line):
            flush_paragraph()
        else:
            paragraph_lines.append(line)
        index += 1

    flush_paragraph()
    return entries


def load_knowledge_schema(context_dir: Path) -> list[dict[str, str]]:
    knowledge_path = context_dir / "knowledge.md"
    if not knowledge_path.exists() or not knowledge_path.is_file():
        return []
    try:
        markdown_text = knowledge_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        markdown_text = knowledge_path.read_text(encoding="utf-8", errors="replace")
    return markdown_knowledge_to_schema(markdown_text)


def format_knowledge_schema_prompt(context_dir: Path) -> str:
    knowledge_path = context_dir / "knowledge.md"
    schema = load_knowledge_schema(context_dir)
    if not schema:
        if knowledge_path.exists():
            status = (
                "A /context/knowledge.md file is present, but no usable schema could be "
                "extracted from it."
            )
        else:
            status = "No /context/knowledge.md file is present."
        return (
            "Knowledge schema:\n"
            f"{status} Treat the knowledge guide as unavailable for this task. Resolve "
            "definitions, units, constraints, and ambiguity by inspecting the actual "
            "/context/ data files, and keep every conclusion consistent with observed data."
        )

    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        "Knowledge schema extracted from /context/knowledge.md:\n"
        "Importance: when usable, this schema is the authoritative reference for data "
        "definitions, units, constraints, table/field semantics, and ambiguity resolution. "
        "Consult it before interpreting raw columns or entity names.\n"
        "Consistency guard: some tasks may contain empty, stale, or invalid knowledge. If "
        "the schema conflicts with observed files or cannot explain the data, prefer "
        "consistency with /context/ data and validate the mapping before answering.\n"
        "Schema JSON:\n"
        f"```json\n{schema_json}\n```"
    )
