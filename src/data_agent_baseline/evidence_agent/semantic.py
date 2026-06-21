from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.evidence_agent.text import normalize_key

_EN_WORD_RE = re.compile(r"[a-z][a-z0-9]{1,}")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _identifier_parts(value: str) -> list[str]:
    spaced = _CAMEL_BOUNDARY_RE.sub(" ", value)
    return [part for part in re.split(r"[^A-Za-z0-9]+", spaced) if len(part) > 1]


def _cjk_terms(value: str) -> set[str]:
    terms: set[str] = set()
    for phrase in _CJK_RE.findall(value):
        if len(phrase) < 2:
            continue
        terms.add(phrase)
        # Generic character n-grams keep Chinese matching lexical without
        # adding a fixed business lexicon.
        for width in (2, 3, 4):
            if len(phrase) < width:
                continue
            for index in range(0, len(phrase) - width + 1):
                terms.add(phrase[index : index + width])
    return terms


def semantic_terms(value: Any) -> set[str]:
    text = str(value or "")
    terms: set[str] = set()
    for word in _EN_WORD_RE.findall(text.casefold()):
        if word not in _STOP_WORDS:
            terms.add(word)
    for part in _identifier_parts(text):
        lowered = part.casefold()
        if lowered not in _STOP_WORDS:
            terms.add(lowered)
    terms.update(_cjk_terms(text))
    compact = normalize_key(text)
    if compact and compact not in _STOP_WORDS:
        terms.add(compact)
    return {term for term in terms if term}


def semantic_text(value: Any) -> str:
    return " ".join(sorted(semantic_terms(value)))


def semantic_score(query: Any, target: Any) -> float:
    query_terms = semantic_terms(query)
    if not query_terms:
        return 0.0
    target_text = str(target or "")
    target_key = normalize_key(target_text)
    target_terms = semantic_terms(target_text)
    overlap = query_terms & target_terms
    score = float(len(overlap) * 2)
    for term in query_terms - overlap:
        if len(term) >= 3 and target_key and term in target_key:
            score += 0.75
    query_key = normalize_key(query)
    if query_key and query_key in target_key:
        score += 6.0
    return score


def column_question_score(question: str, column: str, context_text: str = "") -> float:
    column_key = normalize_key(column)
    if not column_key:
        return 0.0
    score = semantic_score(question, f"{column} {context_text}")
    question_key = normalize_key(question)
    if column_key and column_key in question_key:
        score += 5.0
    return score


def source_question_score(question: str, source_name: str, context_text: str = "") -> float:
    source_key = normalize_key(source_name)
    if not source_key:
        return 0.0
    score = semantic_score(question, f"{source_name} {context_text}")
    question_key = normalize_key(question)
    if source_key and source_key in question_key:
        score += 5.0
    return score
