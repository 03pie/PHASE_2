Extract a source-bound metric column from narrative text records.

Use this when the active analysis_plan binds an output source_field to a
document source through execution_spec.source_bindings, especially when the
logical table is narrative-only. The tool reads the bound document, preserves
source record order, returns one value column, and submits through the same
answer validator as set_answer. Missing or unavailable metric values are kept
as empty strings.

Pass the exact bound source_path and source_field from the plan. Prefer bounded
extraction: first use grep_file to locate candidate lines, then read_doc with a
small start_line/max_lines slice. If a record or section continues beyond the
slice, read the next adjacent slice before extracting. Pass start_line/end_line
for the confirmed bounded section. Use whole-document extraction only when the
whole source is already known to contain only the target record sequence.
