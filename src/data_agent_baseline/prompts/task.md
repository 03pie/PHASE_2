Question: {question}

Complete recursive context inventory:
{context_inventory}

Read `/context/knowledge.md` when present, then inspect the most relevant candidate
identified by knowledge, the question, and the inventory. Valid knowledge is the
strict standard. If it is unusable or does not cover the question, cross-check at
least two independent sources. Stop exploring once source, semantics, calculation,
and output shape are clear; inspect another source only for a specific unresolved
question. Then call `analyze_plan`, execute and revise it as evidence changes,
validate the result, and submit the table with `answer`. Do not list directories.
