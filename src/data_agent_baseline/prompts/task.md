Structured question reference from isolated question-only node:
<question_structure>
{question_structure}
</question_structure>

Use the structured question reference above as the primary intent source for
discovery, planning, and execution. The `original_question` field inside it is
kept for exact quote/provenance checks; do not reread the original wording as a
separate invitation to infer extra calculations, filters, ordering, output
columns, row grain, or scope conversions.

Complete recursive context inventory:
{context_inventory}

Injected `/context/knowledge.md` content:
<context_knowledge>
{knowledge_context}
</context_knowledge>

The injected knowledge block is fixed task context. Treat valid knowledge as the
strict standard; do not spend a tool call rereading `knowledge.md` unless you need
to investigate an apparent inconsistency. Inspect the most relevant candidate
identified by knowledge, the structured question reference, and the inventory. If knowledge is unusable
or does not cover the question, cross-check at least two independent sources. Stop
exploring once source, semantics, calculation, and output shape are clear; inspect
another source only for a specific unresolved question. Preserve every explicit
detail captured in the structured question reference. It is a conservative parse
of the original question only; use it to avoid inventing unstated targets or
conditions. If the structure appears to have missed an exact wording detail,
treat that as an ambiguity to resolve conservatively, not as authorization to add
a transformation.

When building `analyze_plan`, treat the question structure as planning guardrails:
- if `conditions.calculations` is empty, do not include aggregate or derive
  transformations and do not reinterpret geography/scope words as calculation
  requests;
- if `conditions.orderings` is empty, use `ordering="source"` and `sort_keys=[]`;
- if `conditions.output_columns` is empty, do not add date, geography, identity,
  or helper columns unless they are the requested target itself;
- if a value is listed only under `target_constraints`, it is a scope or
  interpretation clue, not authorization for a transformation.

The same guardrails apply during discovery tool calls. Do not use exploration to
test an unstated aggregate, ordering, filter, limit, or helper dimension merely
because it seems useful; those calls may be rejected before execution.

Do not infer a filter, aggregation, derivation, ordering, limit, deduplication, or
reshape from context alone. Without an explicit user or knowledge authorization,
project the requested fields while preserving source rows, order, and nulls. Then
call `analyze_plan`, execute and revise it as evidence changes, validate the result, and submit the table by calling
`set_answer(columns, rows)` inside the final `execute_python` call. Do not list
directories unless the inventory is ambiguous. SQLite table names are already
listed under each database file in the inventory; use those table names to decide
whether a database is relevant before calling `inspect_sqlite`. Prefer the specialized
structured-data tools exposed in the current tool schema over raw file reads. Do
not directly read images, audio, or video files unless an explicit text-returning
tool path is exposed.
