任务问题：
{question}

## 结构化问题参考

<question_structure>
{question_structure}
</question_structure>

使用该结构作为意图参考；`original_question` 是最终语义来源。结构中未捕获但原问题明确表达的计算、筛选、排序、限制或输出列仍可执行，但必须由数据观察验证可行性。

## 上下文清单

{context_inventory}

清单是完整递归视图。除非清单本身存在具体歧义，不要列目录或重新发现文件。

## 注入的 knowledge schema

<knowledge_schema>
{knowledge_schema}
</knowledge_schema>

`knowledge_schema` 是 `/context/knowledge.md` 的压缩语义索引。它定义术语、字段含义、单位、口径和计算逻辑；其中的章节、表格、列标题、`section_key`、`field_key`、`source_name_hint` 和 `fact_id` 都只是文档组织或引用线索，不证明真实数据存在同名表、同名列或固定行粒度。真实格式、定义出现位置和可执行字段必须通过工具观察发现。如果同名物理源缺失，继续查找 CSV/JSON/SQLite/doc/PDF 中的实际表达形式，不要直接判定 knowledge invalid。

## 本任务执行约束

- 先读取由 question_structure、knowledge 和清单共同指向的最相关来源；一旦字段、粒度、覆盖范围和输出形态足够清楚，就调用 `analyze_plan`。
- 每轮工具决策都参考系统注入的 `<evidence_boundary>`：已观察来源证明数据形状，knowledge 提供语义口径，用户问题提供任务目标；口径歧义写入 evidence 或 unresolved，不要无证据推理成操作。
- `output_spec.columns` 只写最终答案列；排序、筛选、join、selector 和上下文字段放入 `execution_spec.supporting_fields`，不要作为最终输出列提交。
- `evidence.context_sources` 和 `execution_spec.sources` 只能引用已经成功读取的 observed source；SQLite 表级来源可引用为 `/context/db.sqlite::table`。
- 如果 knowledge 提到的名称没有直接 SQLite 表，检查同 basename 的 CSV/JSON/doc/PDF，也用 `query_schema` 按字段和值搜索真实来源；不要直接声明 knowledge invalid，也不要退到语义相邻但字段不同的来源。
- 如果真实来源是 doc/Markdown/PDF 叙述，先用 `grep_file` 定位候选行，再用 `read_doc(start_line, max_lines)` 分批读取小切片；执行阶段必须调用 `extract_narrative_records(source_path, source_field=..., start_line=..., end_line=...)` 或 `extract_narrative_records(source_path, source_fields=[...], start_line=..., end_line=...)` 让工具报告抽取字段和行证据；不要用 knowledge 说明表格推断物理列。
- 若 `question_structure.conditions.calculations` 为空且没有 `intent_operators` 授权 `aggregate/derive`，不要把地域、范围或“记录”解释为聚合/派生请求。
- `cross_validated_inference` 和 `intent.unresolved` 只能说明已观察事实、口径不匹配或待确认事项；不要把未证实的数据格式写成执行前提。可执行操作必须在 `output_spec.transformations` 或 `execution_spec.operations` 中声明。
- 若无显式转换授权，保留源行、源顺序和空值；只投影用户要求返回的目标字段。
- preserve/source-row 计划会由 observed source 的 row_count 约束行数；不要为了通过校验而改写 expected row count。
- 字典表、维表或映射表只能在 observed schema、共享键、外键、link 字段或样例行证明可连接时使用；不要用相似 label 替代已命中的数据域。
- 最终在一次 `execute_python` 调用中验证并调用 `set_answer`。直接源投影可用 `set_answer(columns, rows)`；经过筛选、排序、limit、聚合、派生、join、去重或重塑时传入 audit，至少包含 `source_paths` 和 `operations`，`output_row_count` 与 `output_hash` 由工具按提交表自动写入。
