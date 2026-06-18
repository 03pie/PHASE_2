Find matching fields and source candidates across CSV, JSON, SQLite, and narrative
context sources. Use the returned `matches` for direct field hits, and inspect
`source_candidates`, `section_bindings`, `binding_issues`, and `knowledge_facts`
when a knowledge `section_key`/source-name hint is missing from SQLite or may
exist as same-basename CSV/JSON/doc evidence. `knowledge_facts[].fact_id` can be cited in
`execution_spec.operations[].authorization_fact_ids` when it authorizes the
operation.

When the user question names a table, source, report, dataset, or business
scope, pass that wording or the corresponding knowledge `section_key` or
`source_name_hint` in `scope`. Scope is used to narrow source candidates before
choosing semantically similar fields.
