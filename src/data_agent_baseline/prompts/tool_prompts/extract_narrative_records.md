Extract a source-bound metric column from narrative text records.

Use this when the active analysis_plan binds an output source_field to a
document source through execution_spec.source_bindings, especially when the
logical table is narrative-only. The tool reads the bound document, preserves
source record order, returns one value column, and submits through the same
answer validator as set_answer. Missing or unavailable metric values are kept
as empty strings.

Pass the exact bound source_path and source_field from the plan. Use start_line
and end_line only when the relevant narrative section has already been read and
the section boundary is known.
