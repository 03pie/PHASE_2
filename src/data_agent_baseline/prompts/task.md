Question: {question}

Complete recursive context inventory:
{context_inventory}

Read `/context/knowledge.md` when present, then inspect the most relevant candidate
identified by knowledge, the question, and the inventory. Valid knowledge is the
strict standard. If it is unusable or does not cover the question, cross-check at
least two independent sources. Stop exploring once source, semantics, calculation,
and output shape are clear; inspect another source only for a specific unresolved
question. Preserve every explicit detail in the original question. Do not infer a
filter, aggregation, derivation, ordering, limit, deduplication, or reshape from
context alone. Without an explicit user or knowledge authorization, project the
requested fields while preserving source rows, order, and nulls. Then call
`analyze_plan`, execute and revise it as evidence changes, validate the result, and
submit the table by calling `set_answer(columns, rows)` inside the final
`execute_python` call. Do not list directories.
