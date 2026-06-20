Create or revise a traceable analysis plan after inspecting task data.
`output_spec.columns` is the final answer schema only. Put selector, filter,
join, and context fields that are needed for execution but not returned in the
optional `execution_spec.supporting_fields`, and put executable filter/sort/
limit/aggregate steps in `execution_spec.operations`.

Knowledge is semantic guidance: it defines meanings, units, metrics, and
calculation logic. Its markdown tables, section keys, field keys, source hints,
and fact ids do not prove physical tables or columns exist. Use observed data
shape to choose actual sources and fields, and keep exploration-only fields out
of final output columns.
