任务问题：
{question}

## 结构化问题参考

<question_structure>
{question_structure}
</question_structure>

使用该结构作为意图主入口；`original_question` 只用于精确引用校验。结构中未捕获的计算、筛选、排序、限制、输出列或行粒度转换，一律按未授权处理。

## 上下文清单

{context_inventory}

清单是完整递归视图。除非清单本身存在具体歧义，不要列目录或重新发现文件。

## 注入的 knowledge schema

<knowledge_schema>
{knowledge_schema}
</knowledge_schema>

`knowledge_schema` 是 `/context/knowledge.md` 的结构化视图。`analyze_plan.evidence.knowledge_status` 必须使用 `knowledge_status_for_plan` 的取值；不要把 `availability` 或 fact 的 `binding_status` 填到 `knowledge_status`。优先使用其中的 `fact_id`、`logical_table`、`logical_field`、`binding_status`、`bindings` 和原始 `quote` 建立执行契约；不要从未结构化全文自由扩展语义。如果只有物理源绑定缺失，不要把整篇 knowledge 判 invalid。不可用、不足或冲突的部分用 `knowledge_issue` 与 `cross_validated_inference` 说明证据。

## 本任务执行约束

- 先读取由 question_structure、knowledge 和清单共同指向的最相关来源；一旦字段、粒度、覆盖范围和输出形态足够清楚，就调用 `analyze_plan`。
- 每轮工具决策都先遵守系统注入的 `<evidence_boundary>`：已观察来源证明数据形状，用户/knowledge quote 或 fact_id 才能授权转换；口径歧义写入 evidence 或 unresolved，不要无证据推理成操作。
- `output_spec.columns` 只写最终答案列；排序、筛选、join、selector 和上下文字段放入 `execution_spec.supporting_fields`，不要作为最终输出列提交。
- `evidence.context_sources` 和 `execution_spec.sources` 只能引用已经成功读取的 observed source；SQLite 表级来源可引用为 `/context/db.sqlite::table`。
- 如果 knowledge 的逻辑表不在 SQLite 中，先用 `query_schema` 查看 `source_candidates`、`logical_bindings` 和 `binding_issues`，并检查同 basename 的 CSV/JSON/doc 来源；不要直接声明 knowledge invalid，也不要退到语义相邻但字段不同的来源。
- 如果计划中的 `execution_spec.source_bindings` 将最终 `source_field` 绑定到 doc/Markdown/PDF 叙述来源，先用 `grep_file` 定位候选行，再用 `read_doc(start_line, max_lines)` 分批读取小切片；执行阶段必须调用 `extract_narrative_records(source_path, source_field, start_line, end_line)` 从确认后的绑定切片生成答案候选；不要用语义相邻 JSON/CSV 字段替代已绑定的叙述来源。
- 若 `question_structure.conditions.calculations` 为空且没有 `intent_operators` 授权 `aggregate/derive`，不要把地域、范围或“记录”解释为聚合/派生请求。
- `cross_validated_inference` 和 `intent.unresolved` 只能说明已观察事实、口径不匹配或待确认事项；不要把未授权操作写成“如果没有 X 就计算/筛选/排序/限制/重塑 Y”。可执行操作必须在 `output_spec.transformations` 或 `execution_spec.operations` 中声明并引用用户、knowledge 或 KnowledgeFact 授权。
- 若无显式转换授权，保留源行、源顺序和空值；只投影用户要求返回的目标字段。
- preserve/source-row 计划会由 observed source 的 row_count 约束行数；不要为了通过校验而改写 expected row count。
- 字典表要匹配事件表领域：`D_LABITEMS` 对应 `LABEVENTS`；`D_ITEMS.LINKSTO` 指向它关联的事件表或文件。
- 最终在一次 `execute_python` 调用中验证并调用 `set_answer`。直接源投影可用 `set_answer(columns, rows)`；经过筛选、排序、limit、聚合、派生、join、去重或重塑时传入 audit，至少包含 `source_paths` 和 `operations`，`output_row_count` 与 `output_hash` 由工具按提交表自动写入。
