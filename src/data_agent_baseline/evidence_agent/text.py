from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_IDENTIFIER_RE = re.compile(
    r"[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]*"
    r"(?:\.[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]*)*"
)
_CODE_SPAN_RE = re.compile(r"`([^`\n]{1,180})`")
_SQL_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_QUESTION_VALUE_RE = re.compile(
    r"\b\d{3}-\d{3,}\b|\b\d{6}\b|\b\d{4,}\b|[\u4e00-\u9fff]{2,12}"
)


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").casefold())


def normalize_identifier(value: Any) -> str:
    return re.sub(r"[^a-z0-9_\u4e00-\u9fff]+", "", str(value or "").casefold())


def code_mentions(text: str) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for group in _CODE_SPAN_RE.findall(text or ""):
        for match in _IDENTIFIER_RE.findall(group):
            key = match.casefold()
            if key not in seen:
                seen.add(key)
                values.append(match)
    for match in _IDENTIFIER_RE.findall(text or ""):
        key = match.casefold()
        if key not in seen and (
            "." in match or "_" in match or any(char.isupper() for char in match[1:])
        ):
            seen.add(key)
            values.append(match)
    return values


def sql_blocks(text: str) -> list[str]:
    blocks = [block.strip() for block in _SQL_BLOCK_RE.findall(text or "") if block.strip()]
    if blocks:
        return blocks
    if re.search(r"\bSELECT\b.+\bFROM\b", text or "", flags=re.IGNORECASE | re.DOTALL):
        return [text.strip()]
    return []


def question_values(question: str) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for match in _QUESTION_VALUE_RE.findall(question or ""):
        value = match.strip()
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    return values


def read_document_text(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        try:
            import fitz  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - dependency is environment-bound
            raise RuntimeError(f"PyMuPDF is required to read PDF files: {exc}") from exc
        document = fitz.open(path)
        try:
            return "\n".join(page.get_text("text") for page in document)
        finally:
            document.close()
    return path.read_text(encoding="utf-8", errors="replace")


def compact_text(text: str, *, limit: int = 500) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."
