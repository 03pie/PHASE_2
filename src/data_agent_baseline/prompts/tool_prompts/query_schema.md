Find matching fields and source candidates across CSV, JSON, SQLite, and narrative
context sources. Use the returned `matches` for direct field hits, and inspect
`source_candidates`, `logical_bindings`, `binding_issues`, and `knowledge_facts`
when a logical table from knowledge is missing from SQLite or may exist as
same-basename CSV/JSON/doc evidence. `knowledge_facts[].fact_id` can be cited in
`execution_spec.operations[].authorization_fact_ids` when it authorizes the
operation.
