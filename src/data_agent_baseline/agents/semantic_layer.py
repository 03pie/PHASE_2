from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

KnowledgeFactKind = Literal[
    "logical_table",
    "field",
    "unit",
    "join",
    "calculation",
    "filter_rule",
    "ordering_rule",
    "output_rule",
    "example_query",
]
KnowledgeFactStatus = Literal[
    "usable",
    "binding_unresolved",
    "schema_conflict",
    "narrative_only",
]
SourceType = Literal["sqlite", "csv", "json", "doc"]
BindingConfidence = Literal[
    "exact",
    "case_variant",
    "basename_match",
    "narrative_match",
]

DB_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
DOC_SUFFIXES = {".log", ".md", ".pdf", ".txt"}


@dataclass(frozen=True)
class KnowledgeFact:
    fact_id: str
    kind: KnowledgeFactKind
    logical_table: str | None
    logical_field: str | None
    operation: str | None
    quote: str
    source_path: str
    status: KnowledgeFactStatus = "usable"


@dataclass(frozen=True)
class PhysicalBinding:
    logical_name: str
    source_path: str
    source_type: SourceType
    table_or_path: str
    fields: tuple[str, ...]
    confidence: BindingConfidence
    evidence: str


@dataclass(frozen=True)
class SemanticContext:
    facts: tuple[KnowledgeFact, ...]
    bindings: tuple[PhysicalBinding, ...]
    issues: tuple[str, ...]


def _virtual_path(path: Path, context_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(context_root.resolve())
        return "/context/" + rel.as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_name(value: str | None) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value or "").casefold())


def _split_markdown_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells or all(not cell for cell in cells):
        return None
    if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
        return None
    return [cell.strip().strip("`").strip() for cell in cells]


def _infer_heading_table(line: str) -> str | None:
    stripped = line.strip().lstrip("#").strip()
    backticked = re.findall(r"`([^`]+)`", stripped)
    for value in reversed(backticked):
        candidate = value.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", candidate):
            continue
        if "_" in candidate or candidate.isupper():
            return candidate
    matches = [
        match
        for match in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{2,}\b", stripped)
        if "_" in match
    ]
    return matches[-1] if matches else None


def _operation_from_sql(sql: str) -> str | None:
    upper = sql.upper()
    operations = []
    if " WHERE " in f" {upper} ":
        operations.append("filter")
    if re.search(r"\b(GROUP\s+BY|COUNT|SUM|AVG|MIN|MAX)\s*\(?", upper):
        operations.append("aggregate")
    if " ORDER BY " in f" {upper} ":
        operations.append("sort")
    if " LIMIT " in f" {upper} ":
        operations.append("limit")
    return ",".join(operations) if operations else None


def _table_from_sql(sql: str) -> str | None:
    match = re.search(r"\bFROM\s+([`\"\[]?)([a-zA-Z0-9_./-]+)\1", sql, re.IGNORECASE)
    return match.group(2) if match else None


def _fields_from_text(text: str) -> list[str]:
    fields: list[str] = []
    for value in re.findall(r"`([^`]+)`", text):
        if value not in fields:
            fields.append(value)
    return fields


def _status_for_fact(
    *,
    fact: KnowledgeFact,
    bindings: Iterable[PhysicalBinding],
) -> KnowledgeFactStatus:
    table = _normalize_name(fact.logical_table)
    field = _normalize_name(fact.logical_field)
    if not table and not field:
        return fact.status
    table_bindings = [
        binding
        for binding in bindings
        if table and _normalize_name(binding.logical_name) == table
    ]
    if table and not table_bindings:
        return "binding_unresolved"
    if field and table_bindings:
        for binding in table_bindings:
            fields = {_normalize_name(item) for item in binding.fields}
            if field in fields:
                return fact.status
            if any(item.casefold() == str(fact.logical_field or "").casefold() for item in binding.fields):
                return "usable"
        if any(binding.source_type == "doc" for binding in table_bindings):
            return "narrative_only"
        return "schema_conflict"
    if table_bindings and any(binding.source_type == "doc" for binding in table_bindings):
        return "narrative_only"
    return fact.status


def parse_knowledge_content(
    content: str,
    *,
    source_path: str = "/context/knowledge.md",
) -> tuple[KnowledgeFact, ...]:
    facts: list[KnowledgeFact] = []
    current_table: str | None = None
    markdown_header: list[str] | None = None
    in_code = False
    code_lines: list[str] = []

    def append_fact(
        kind: KnowledgeFactKind,
        quote: str,
        *,
        logical_table: str | None = None,
        logical_field: str | None = None,
        operation: str | None = None,
        status: KnowledgeFactStatus = "usable",
    ) -> None:
        text = quote.strip()
        if not text:
            return
        facts.append(
            KnowledgeFact(
                fact_id=f"kf_{len(facts) + 1}",
                kind=kind,
                logical_table=logical_table or current_table,
                logical_field=logical_field,
                operation=operation,
                quote=text,
                source_path=source_path,
                status=status,
            )
        )

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                sql = "\n".join(code_lines).strip()
                operation = _operation_from_sql(sql)
                append_fact(
                    "example_query",
                    sql,
                    logical_table=_table_from_sql(sql) or current_table,
                    operation=operation,
                )
                code_lines = []
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        if stripped.startswith("#"):
            current_table = _infer_heading_table(stripped)
            continue

        row = _split_markdown_row(stripped)
        if row is not None and len(row) >= 2:
            headerish = {cell.casefold() for cell in row}
            if {"column", "field", "字段"} & headerish or any(
                cell.casefold() in {"table", "表名", "单位", "unit"}
                for cell in row
            ):
                markdown_header = row
                continue

            logical_table = current_table
            logical_field = row[0].strip()
            unit_value = ""
            if markdown_header:
                header_cells = [cell.casefold() for cell in markdown_header]

                def find_header(*needles: str) -> int | None:
                    for index, header in enumerate(header_cells):
                        if any(needle in header for needle in needles):
                            return index
                    return None

                table_index = find_header("表名", "table")
                field_index = find_header("字段", "column", "field")
                unit_index = find_header("单位", "unit")
                if table_index is not None and table_index < len(row):
                    logical_table = row[table_index].strip() or logical_table
                if field_index is not None and field_index < len(row):
                    logical_field = row[field_index].strip()
                if unit_index is not None and unit_index < len(row):
                    unit_value = row[unit_index].strip()
            if logical_field:
                append_fact(
                    "field",
                    stripped,
                    logical_table=logical_table,
                    logical_field=logical_field,
                )
            unit_cells = [
                cell
                for cell in ([unit_value] if unit_value else row[2:])
                if cell and re.search(r"(unit|亿元|元|%|percent|date|time|股|年)", cell, re.IGNORECASE)
            ]
            if logical_field and unit_cells:
                append_fact(
                    "unit",
                    stripped,
                    logical_table=logical_table,
                    logical_field=logical_field,
                )
            continue

        lower = stripped.casefold()
        if "join" in lower or "关联" in stripped or "连接" in stripped or "↔" in stripped:
            append_fact("join", stripped, operation="join")
            continue
        if re.search(r"\b(formula|calculate|calculation)\b|公式|计算", stripped, re.IGNORECASE):
            fields = _fields_from_text(stripped)
            append_fact(
                "calculation",
                stripped,
                logical_field=fields[0] if fields else None,
                operation="derive",
            )
            continue
        if re.search(r"\b(where|filter)\b|筛选|过滤|条件", stripped, re.IGNORECASE):
            fields = _fields_from_text(stripped)
            append_fact(
                "filter_rule",
                stripped,
                logical_field=fields[0] if fields else None,
                operation="filter",
            )
            continue
        if re.search(r"\b(order|sort|top|limit)\b|排序|最高|最低|前\d+", stripped, re.IGNORECASE):
            fields = _fields_from_text(stripped)
            append_fact(
                "ordering_rule",
                stripped,
                logical_field=fields[0] if fields else None,
                operation="sort",
            )
            continue

    if in_code and code_lines:
        sql = "\n".join(code_lines).strip()
        append_fact(
            "example_query",
            sql,
            logical_table=_table_from_sql(sql) or current_table,
            operation=_operation_from_sql(sql),
        )
    return tuple(facts)


def _iter_json_keys(data: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(data, Mapping):
        for key, value in data.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            yield path
            yield from _iter_json_keys(value, path)
    elif isinstance(data, list):
        for item in data[:5]:
            yield from _iter_json_keys(item, prefix)


def _sample_doc_evidence(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return f"narrative source {path.name}"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return f"narrative source {path.name}"
    for line in lines:
        text = line.strip()
        if text:
            return text[:240]
    return f"narrative source {path.name}"


def index_context_sources(context_root: Path) -> tuple[PhysicalBinding, ...]:
    bindings: list[PhysicalBinding] = []

    for path in sorted(context_root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        virtual = _virtual_path(path, context_root)
        stem = path.stem
        if suffix in DB_SUFFIXES:
            try:
                with sqlite3.connect(str(path)) as connection:
                    cursor = connection.cursor()
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    )
                    for (table_name,) in cursor.fetchall():
                        table = str(table_name)
                        cursor.execute(
                            'PRAGMA table_info("' + table.replace('"', '""') + '")'
                        )
                        fields = tuple(str(row[1]) for row in cursor.fetchall())
                        bindings.append(
                            PhysicalBinding(
                                logical_name=table,
                                source_path=virtual,
                                source_type="sqlite",
                                table_or_path=table,
                                fields=fields,
                                confidence="exact",
                                evidence=f"SQLite table {table} with {len(fields)} columns.",
                            )
                        )
            except sqlite3.Error:
                continue
        elif suffix == ".csv":
            try:
                with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                    fields = tuple(str(item) for item in next(csv.reader(handle), []))
            except (OSError, csv.Error):
                continue
            bindings.append(
                PhysicalBinding(
                    logical_name=stem,
                    source_path=virtual,
                    source_type="csv",
                    table_or_path=virtual,
                    fields=fields,
                    confidence="basename_match",
                    evidence=f"CSV {path.name} with {len(fields)} header fields.",
                )
            )
        elif suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            keys = tuple(dict.fromkeys(_iter_json_keys(data)))
            bindings.append(
                PhysicalBinding(
                    logical_name=stem,
                    source_path=virtual,
                    source_type="json",
                    table_or_path=virtual,
                    fields=keys,
                    confidence="basename_match",
                    evidence=f"JSON {path.name} with {len(keys)} observed keys.",
                )
            )
        elif suffix in DOC_SUFFIXES:
            bindings.append(
                PhysicalBinding(
                    logical_name=stem,
                    source_path=virtual,
                    source_type="doc",
                    table_or_path=virtual,
                    fields=(),
                    confidence="narrative_match",
                    evidence=_sample_doc_evidence(path),
                )
            )
    return tuple(bindings)


def build_semantic_context(context_root: Path) -> SemanticContext:
    knowledge_path = context_root / "knowledge.md"
    raw_facts: tuple[KnowledgeFact, ...] = ()
    if knowledge_path.exists():
        try:
            raw_facts = parse_knowledge_content(
                knowledge_path.read_text(encoding="utf-8", errors="replace")
            )
        except OSError:
            raw_facts = ()
    bindings = index_context_sources(context_root)
    facts = tuple(
        fact
        if (status := _status_for_fact(fact=fact, bindings=bindings)) == fact.status
        else KnowledgeFact(
            fact_id=fact.fact_id,
            kind=fact.kind,
            logical_table=fact.logical_table,
            logical_field=fact.logical_field,
            operation=fact.operation,
            quote=fact.quote,
            source_path=fact.source_path,
            status=status,
        )
        for fact in raw_facts
    )

    issues: list[str] = []
    binding_names = {_normalize_name(binding.logical_name) for binding in bindings}
    for fact in facts:
        table = _normalize_name(fact.logical_table)
        if table and table not in binding_names:
            issues.append(
                f"Knowledge logical table {fact.logical_table!r} has no exact physical binding."
            )
    return SemanticContext(facts=facts, bindings=bindings, issues=tuple(dict.fromkeys(issues)))


def _matches_query(binding: PhysicalBinding, query_norm: str) -> bool:
    if not query_norm:
        return False
    names = [
        binding.logical_name,
        Path(binding.table_or_path).stem,
        binding.table_or_path,
        *binding.fields,
    ]
    for name in names:
        normalized = _normalize_name(name)
        if not normalized:
            continue
        if query_norm in normalized:
            return True
        if len(normalized) >= 5 and normalized in query_norm:
            return True
    return False


def _binding_score(binding: PhysicalBinding, query_norm: str) -> tuple[int, int, str]:
    logical_norm = _normalize_name(binding.logical_name)
    field_norms = [_normalize_name(item) for item in binding.fields]
    type_rank = {"sqlite": 0, "csv": 1, "json": 2, "doc": 3}.get(binding.source_type, 9)
    if logical_norm == query_norm:
        return (0, type_rank, binding.source_path)
    if logical_norm and (
        logical_norm in query_norm or query_norm in logical_norm
    ):
        return (1, type_rank, binding.source_path)
    if query_norm in field_norms:
        return (2, type_rank, binding.source_path)
    if any(
        query_norm in field or (len(field) >= 5 and field in query_norm)
        for field in field_norms
        if field
    ):
        return (3, type_rank, binding.source_path)
    return (4, type_rank, binding.source_path)


def _local_path_from_source(source_path: str, context_root: Path) -> Path | None:
    if not source_path.startswith("/context/"):
        return None
    return context_root / source_path.removeprefix("/context/")


def _line_evidence(
    binding: PhysicalBinding,
    context_root: Path,
    terms: Iterable[str],
) -> list[dict[str, Any]]:
    if binding.source_type != "doc":
        return []
    path = _local_path_from_source(binding.source_path, context_root)
    if path is None or path.suffix.lower() == ".pdf":
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    normalized_terms = [
        _normalize_name(term)
        for term in terms
        if _normalize_name(term)
    ]
    if not normalized_terms:
        return []
    evidence: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        normalized_line = _normalize_name(line)
        if not any(term in normalized_line for term in normalized_terms):
            continue
        evidence.append(
            {
                "line_number": index + 1,
                "content": line[:300],
            }
        )
        if len(evidence) >= 5:
            break
    return evidence


def _binding_payload(
    binding: PhysicalBinding,
    context_root: Path,
    terms: Iterable[str],
) -> dict[str, Any]:
    payload = asdict(binding)
    evidence = _line_evidence(binding, context_root, terms)
    if evidence:
        payload["line_evidence"] = evidence
    return payload


def query_semantic_context(
    context_root: Path,
    query: str,
    *,
    max_matches: int = 25,
) -> dict[str, Any]:
    semantic = build_semantic_context(context_root)
    query_norm = _normalize_name(query)
    candidates = [
        binding
        for binding in semantic.bindings
        if _matches_query(binding, query_norm)
    ]
    candidate_keys = {
        (binding.source_path, binding.table_or_path)
        for binding in candidates
    }
    for fact in semantic.facts:
        fact_field = _normalize_name(fact.logical_field)
        fact_table = _normalize_name(fact.logical_table)
        if not query_norm:
            continue
        if not (
            (fact_field and (query_norm in fact_field or fact_field in query_norm))
            or (fact_table and (query_norm in fact_table or fact_table in query_norm))
        ):
            continue
        if not fact_table:
            continue
        for binding in semantic.bindings:
            if _normalize_name(binding.logical_name) != fact_table:
                continue
            key = (binding.source_path, binding.table_or_path)
            if key in candidate_keys:
                continue
            candidates.append(binding)
            candidate_keys.add(key)
    candidates.sort(key=lambda binding: _binding_score(binding, query_norm))
    candidates = candidates[:max_matches]

    logical_bindings: dict[str, list[dict[str, Any]]] = {}
    for fact in semantic.facts:
        logical = fact.logical_table or fact.logical_field
        if not logical:
            continue
        logical_norm = _normalize_name(logical)
        matches = [
            binding
            for binding in semantic.bindings
            if _normalize_name(binding.logical_name) == logical_norm
            or any(_normalize_name(field) == _normalize_name(fact.logical_field) for field in binding.fields)
        ][:5]
        if not matches:
            continue
        logical_bindings.setdefault(logical, [])
        for binding in matches:
            logical_bindings[logical].append(asdict(binding))

    def fact_related(fact: KnowledgeFact) -> bool:
        if not query_norm:
            return True
        values = [
            fact.logical_table or "",
            fact.logical_field or "",
            fact.quote,
        ]
        return any(query_norm in _normalize_name(value) for value in values)

    binding_issues = [
        issue
        for issue in semantic.issues
        if query_norm and query_norm in _normalize_name(issue)
    ]
    for fact in semantic.facts:
        if not fact_related(fact):
            continue
        if fact.status in {"binding_unresolved", "schema_conflict", "narrative_only"}:
            binding_issues.append(
                (
                    f"{fact.fact_id} {fact.kind} {fact.logical_table or ''}."
                    f"{fact.logical_field or ''} status={fact.status}"
                ).strip()
            )

    return {
        "source_candidates": [
            _binding_payload(binding, context_root, [query])
            for binding in candidates
        ],
        "logical_bindings": logical_bindings,
        "binding_issues": list(dict.fromkeys(binding_issues))[:max_matches],
        "knowledge_facts": [asdict(fact) for fact in semantic.facts[:max_matches]],
    }
