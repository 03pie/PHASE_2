Extract source-bound metric columns from narrative text records.

Use this when the active analysis_plan needs values from a narrative document,
especially when the knowledge section/source hint is narrative-only. A requested
source_field/source_fields value is an extraction target, not proof that the
document has a physical column with that name. The tool reads the document,
preserves source record order, returns one or more value columns, and reports
line/field evidence through the same answer validator as set_answer. Missing or
unavailable metric values are kept as empty strings.

Pass the exact bound source_path and either source_field for one column or
source_fields for multiple columns. Optional field_aliases may provide literal
phrases to look near each field; record_anchor limits extraction to matching
record text when a repeated record identifier is present. Prefer bounded
extraction: first use grep_file to locate candidate lines, then read_doc with a
small start_line/max_lines slice. If a record or section continues beyond the
slice, read the next adjacent slice before extracting. Pass start_line/end_line
for the confirmed bounded section. Use whole-document extraction only when the
whole source is already known to contain only the target record sequence.

This is a mechanical extractor. It does not decide which fields the task needs;
that must already be expressed in analysis_plan and the tool arguments.
