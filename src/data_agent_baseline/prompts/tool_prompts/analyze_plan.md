Create or revise a traceable analysis plan after inspecting task data.
`output_spec.columns` is the final answer schema only. Put selector, filter,
join, and context fields that are needed for execution but not returned in the
optional `execution_spec.supporting_fields`, and put executable filter/sort/
limit/aggregate steps in `execution_spec.operations` with exact user quotes,
knowledge quotes, or KnowledgeFact ids as authorization.
