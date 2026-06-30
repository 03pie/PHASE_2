from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Literal

DataForm = Literal[
    "sqlite_database",
    "csv_records",
    "json_records",
    "pdf_document",
    "markdown_document",
    "video",
    "unknown_file",
]

ActionKind = Literal["tool_call", "bind", "compute", "final", "blocked"]
BindingType = Literal[
    "structured_source",
    "structured_field",
    "document_window",
    "document_record_set",
    "value",
    "operation",
    "answer_candidate",
]

RequirementStatus = Literal[
    "pending",
    "satisfied",
    "not_applicable",
    "conflict",
    "blocked",
]

COMPUTABLE_BINDING_TYPES = {
    "structured_source",
    "structured_field",
    "document_record_set",
}


@dataclass(frozen=True, slots=True)
class KnowledgeSection:
    id: str
    heading_path: str
    line_start: int | None
    line_end: int | None
    text: str
    mentions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class KnowledgeLookupEntry:
    token: str
    section_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    status: str = "document_mention_only"
    must_verify: bool = True


@dataclass(frozen=True, slots=True)
class KnowledgeSemanticCard:
    id: str
    kind: str
    semantic_scope: str
    semantic_slot: str | None
    name: str
    definition: str
    aliases: tuple[str, ...] = ()
    unit: str | None = None
    record_grain: str | None = None
    join_keys: tuple[str, ...] = ()
    formula: str | None = None
    section_id: str | None = None
    heading_path: str = ""
    line_start: int | None = None
    line_end: int | None = None

    @property
    def semantic_id(self) -> str:
        return self.id

    @property
    def canonical_table(self) -> str:
        """Backward-compatible alias for the semantic scope, not a physical table."""

        return self.semantic_scope

    @property
    def canonical_field(self) -> str | None:
        """Backward-compatible alias for the semantic slot, not a physical field."""

        return self.semantic_slot

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KnowledgeSourceMapping:
    card_id: str
    source_id: str | None
    source_path: str | None
    data_form: DataForm | str | None
    status: str
    semantic_scope: str | None = None
    semantic_slot: str | None = None
    physical_table: str | None = None
    physical_field: str | None = None
    match_reason: str = ""
    warnings: tuple[str, ...] = ()

    @property
    def matched_table(self) -> str | None:
        """Backward-compatible alias for the candidate physical table."""

        return self.physical_table

    @property
    def matched_field(self) -> str | None:
        """Backward-compatible alias for the candidate physical field."""

        return self.physical_field

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceRef:
    id: str
    path: Path
    virtual_path: str
    basename: str
    stem: str
    suffix: str
    data_form: DataForm
    size_bytes: int
    tables: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = self.path.as_posix()
        return payload


@dataclass(frozen=True, slots=True)
class CandidateRef:
    id: str
    kind: str
    source_id: str | None
    data_form: DataForm | str
    match_reason: str
    path: str | None = None
    table: str | None = None
    field: str | None = None
    value: Any = None
    evidence_id: str | None = None
    requires_verification: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    tool_name: str
    ok: bool
    summary: str
    payload: dict[str, Any] = dataclass_field(default_factory=dict)
    source_id: str | None = None
    candidate_id: str | None = None
    data_form: DataForm | str | None = None
    negative_scope: dict[str, Any] | None = None
    allowed_next_tools: tuple[str, ...] = ()
    recommended_next_actions: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Binding:
    id: str
    binding_type: BindingType | str
    evidence_refs: tuple[str, ...]
    source_id: str | None = None
    relation_name: str | None = None
    table: str | None = None
    field: str | None = None
    allowed_columns: tuple[str, ...] = ()
    alignment: str = ""
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Requirement:
    id: str
    text: str
    status: RequirementStatus | str = "pending"
    source_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    binding_refs: tuple[str, ...] = ()
    compute_refs: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnswerContract:
    intent_summary: str = ""
    answer_grain: str = ""
    final_outputs: tuple[str, ...] = ()
    requested_outputs: tuple[str, ...] = ()
    constraints: tuple[dict[str, Any], ...] = ()
    operations: dict[str, Any] = dataclass_field(default_factory=dict)
    helper_fields: dict[str, tuple[str, ...]] = dataclass_field(default_factory=dict)
    field_roles: tuple[dict[str, Any], ...] = ()
    row_shape: str = "preserve_rows"
    row_limit: int | None = None
    null_policy: str = "preserve"
    transform_intent: str = ""
    document_policy: dict[str, Any] = dataclass_field(default_factory=dict)
    unresolved_terms: tuple[str, ...] = ()
    not_physical_schema: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticSourceCandidate:
    card_id: str
    semantic_scope: str | None = None
    semantic_slot: str | None = None
    source_id: str | None = None
    source_path: str | None = None
    data_form: DataForm | str | None = None
    physical_table: str | None = None
    physical_field: str | None = None
    candidate_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticSelection:
    card_ids: tuple[str, ...] = ()
    selected_cards: tuple[dict[str, Any], ...] = ()
    rationale: str = ""
    unmapped_intents: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ComputeResult:
    id: str
    sql: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    binding_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelAction:
    kind: ActionKind | str
    reason: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] = dataclass_field(default_factory=dict)
    binding_type: BindingType | str | None = None
    evidence_refs: tuple[str, ...] = ()
    source_ref: str | None = None
    binding_refs: tuple[str, ...] = ()
    sql: str | None = None
    compute_ref: str | None = None
    answer: dict[str, Any] | None = None
    raw: dict[str, Any] = dataclass_field(default_factory=dict)

    def signature(self) -> str:
        if self.kind == "tool_call":
            return f"tool_call:{self.tool_name}:{self.arguments}"
        if self.kind == "bind":
            return f"bind:{self.binding_type}:{self.evidence_refs}:{self.source_ref}:{self.arguments}"
        if self.kind == "compute":
            return f"compute:{self.sql}:{self.binding_refs}"
        if self.kind == "final":
            return f"final:{self.compute_ref}:{self.binding_refs}:{self.evidence_refs}:{self.answer}"
        return f"{self.kind}:{self.reason}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "binding_type": self.binding_type,
            "evidence_refs": list(self.evidence_refs),
            "source_ref": self.source_ref,
            "binding_refs": list(self.binding_refs),
            "sql": self.sql,
            "compute_ref": self.compute_ref,
            "answer": self.answer,
        }


@dataclass(frozen=True, slots=True)
class GuardDecision:
    allowed: bool
    reason: str
    allowed_next_tools: tuple[str, ...] = ()
    recommended_next_actions: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    max_output_chars: int = 8_000
    observes_environment: bool = True
    candidate_only_allowed: bool = True
    data_forms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    tool_name: str
    call_id: str
    arguments: dict[str, Any]
    action: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ToolOutputEnvelope:
    ok: bool
    evidence_id: str
    tool_name: str
    summary: str
    payload: dict[str, Any]
    guard: dict[str, Any]
    source_id: str | None = None
    candidate_id: str | None = None
    data_form: DataForm | str | None = None
    negative_scope: dict[str, Any] | None = None
    allowed_next_tools: tuple[str, ...] = ()
    recommended_next_actions: tuple[dict[str, Any], ...] = ()
    progress: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_evidence(
        cls,
        evidence: Evidence,
        *,
        guard: GuardDecision,
        progress: dict[str, Any] | None = None,
    ) -> "ToolOutputEnvelope":
        return cls(
            ok=evidence.ok,
            evidence_id=evidence.id,
            tool_name=evidence.tool_name,
            summary=evidence.summary,
            payload=evidence.payload,
            guard=guard.to_dict(),
            source_id=evidence.source_id,
            candidate_id=evidence.candidate_id,
            data_form=evidence.data_form,
            negative_scope=evidence.negative_scope,
            allowed_next_tools=evidence.allowed_next_tools,
            recommended_next_actions=evidence.recommended_next_actions,
            progress=progress or {},
        )


@dataclass(frozen=True, slots=True)
class RecoveryHint:
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    source: str
    priority: int = 0
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TranscriptWindow:
    max_groups: int = 8
    groups: list[list[Any]] = dataclass_field(default_factory=list)

    def add_model_response(self, response: Any) -> None:
        self.groups.append([response])
        self._trim()

    def add_tool_output(self, message: Any) -> None:
        if not self.groups:
            self.groups.append([message])
        else:
            self.groups[-1].append(message)
        self._trim()

    def messages(self) -> list[Any]:
        output: list[Any] = []
        for group in self.groups[-self.max_groups :]:
            output.extend(group)
        return output

    def _trim(self) -> None:
        if len(self.groups) > self.max_groups:
            self.groups = self.groups[-self.max_groups :]


@dataclass(slots=True)
class LoopState:
    question: str
    context_dir: Path
    sources: dict[str, SourceRef] = dataclass_field(default_factory=dict)
    source_by_path: dict[str, str] = dataclass_field(default_factory=dict)
    knowledge_sections: list[KnowledgeSection] = dataclass_field(default_factory=list)
    knowledge_lookup: dict[str, KnowledgeLookupEntry] = dataclass_field(default_factory=dict)
    matched_sections: list[KnowledgeSection] = dataclass_field(default_factory=list)
    semantic_cards: list[KnowledgeSemanticCard] = dataclass_field(default_factory=list)
    matched_semantic_cards: list[KnowledgeSemanticCard] = dataclass_field(default_factory=list)
    source_mappings: list[KnowledgeSourceMapping] = dataclass_field(default_factory=list)
    semantic_selection: SemanticSelection | None = None
    selected_source_mappings: list[KnowledgeSourceMapping] = dataclass_field(default_factory=list)
    semantic_selection_errors: list[str] = dataclass_field(default_factory=list)
    candidates: dict[str, CandidateRef] = dataclass_field(default_factory=dict)
    evidence: dict[str, Evidence] = dataclass_field(default_factory=dict)
    bindings: dict[str, Binding] = dataclass_field(default_factory=dict)
    requirements: dict[str, Requirement] = dataclass_field(default_factory=dict)
    answer_contract: AnswerContract | None = None
    compute_results: dict[str, ComputeResult] = dataclass_field(default_factory=dict)
    document_record_indexes: dict[str, Any] = dataclass_field(default_factory=dict)
    document_coverage: dict[str, Any] = dataclass_field(default_factory=dict)
    document_agent_packages: list[dict[str, Any]] = dataclass_field(default_factory=list)
    guard_feedback: list[dict[str, Any]] = dataclass_field(default_factory=list)
    negative_scopes: list[dict[str, Any]] = dataclass_field(default_factory=list)
    _negative_scope_keys: set[str] = dataclass_field(default_factory=set)
    final_answer: dict[str, Any] | None = None
    blocked_reason: str | None = None
    step_index: int = 0
    last_progress_key: str | None = None
    repeated_no_progress: int = 0
    _candidate_seq: int = 0
    _evidence_seq: int = 0
    _binding_seq: int = 0
    _requirement_seq: int = 0
    _compute_seq: int = 0

    def add_candidate(
        self,
        *,
        kind: str,
        source_id: str | None,
        data_form: DataForm | str,
        match_reason: str,
        path: str | None = None,
        table: str | None = None,
        field: str | None = None,
        value: Any = None,
        evidence_id: str | None = None,
    ) -> CandidateRef:
        self._candidate_seq += 1
        candidate = CandidateRef(
            id=f"cand_{self._candidate_seq:04d}",
            kind=kind,
            source_id=source_id,
            data_form=data_form,
            match_reason=match_reason,
            path=path,
            table=table,
            field=field,
            value=value,
            evidence_id=evidence_id,
        )
        self.candidates[candidate.id] = candidate
        return candidate

    def add_evidence(
        self,
        *,
        tool_name: str,
        ok: bool,
        summary: str,
        payload: dict[str, Any] | None = None,
        source_id: str | None = None,
        candidate_id: str | None = None,
        data_form: DataForm | str | None = None,
        negative_scope: dict[str, Any] | None = None,
        allowed_next_tools: tuple[str, ...] = (),
        recommended_next_actions: tuple[dict[str, Any], ...] = (),
    ) -> Evidence:
        if negative_scope is not None:
            self.add_negative_scope(negative_scope)
        self._evidence_seq += 1
        evidence = Evidence(
            id=f"ev_{self._evidence_seq:04d}",
            tool_name=tool_name,
            ok=ok,
            summary=summary,
            payload=payload or {},
            source_id=source_id,
            candidate_id=candidate_id,
            data_form=data_form,
            negative_scope=negative_scope,
            allowed_next_tools=allowed_next_tools,
            recommended_next_actions=recommended_next_actions,
        )
        self.evidence[evidence.id] = evidence
        return evidence

    def add_negative_scope(self, scope: dict[str, Any]) -> bool:
        key = repr(sorted((str(k), str(v)) for k, v in scope.items()))
        if key in self._negative_scope_keys:
            return False
        self._negative_scope_keys.add(key)
        self.negative_scopes.append(dict(scope))
        return True

    def add_binding(
        self,
        *,
        binding_type: BindingType | str,
        evidence_refs: tuple[str, ...],
        source_id: str | None = None,
        relation_name: str | None = None,
        table: str | None = None,
        field: str | None = None,
        allowed_columns: tuple[str, ...] = (),
        alignment: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Binding:
        self._binding_seq += 1
        default_relation = (
            f"rel_{self._binding_seq:04d}"
            if str(binding_type) in COMPUTABLE_BINDING_TYPES
            else None
        )
        binding = Binding(
            id=f"bind_{self._binding_seq:04d}",
            binding_type=binding_type,
            evidence_refs=evidence_refs,
            source_id=source_id,
            relation_name=relation_name or default_relation,
            table=table,
            field=field,
            allowed_columns=allowed_columns,
            alignment=alignment,
            metadata=metadata or {},
        )
        self.bindings[binding.id] = binding
        return binding

    def upsert_requirement(
        self,
        *,
        requirement_id: str | None = None,
        text: str,
        status: RequirementStatus | str = "pending",
        source_refs: tuple[str, ...] = (),
        evidence_refs: tuple[str, ...] = (),
        binding_refs: tuple[str, ...] = (),
        compute_refs: tuple[str, ...] = (),
        note: str = "",
    ) -> Requirement:
        if requirement_id and requirement_id in self.requirements:
            req_id = requirement_id
        else:
            self._requirement_seq += 1
            req_id = requirement_id or f"req_{self._requirement_seq:04d}"
        requirement = Requirement(
            id=req_id,
            text=text,
            status=status,
            source_refs=source_refs,
            evidence_refs=evidence_refs,
            binding_refs=binding_refs,
            compute_refs=compute_refs,
            note=note,
        )
        self.requirements[req_id] = requirement
        return requirement

    def add_compute_result(
        self,
        *,
        sql: str,
        columns: tuple[str, ...],
        rows: tuple[tuple[Any, ...], ...],
        binding_refs: tuple[str, ...],
        evidence_refs: tuple[str, ...],
        ok: bool = True,
        error: str | None = None,
    ) -> ComputeResult:
        self._compute_seq += 1
        result = ComputeResult(
            id=f"comp_{self._compute_seq:04d}",
            sql=sql,
            columns=columns,
            rows=rows,
            binding_refs=binding_refs,
            evidence_refs=evidence_refs,
            ok=ok,
            error=error,
        )
        self.compute_results[result.id] = result
        return result
