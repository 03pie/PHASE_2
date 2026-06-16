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

## 注入的 knowledge

<context_knowledge>
{knowledge_context}
</context_knowledge>

如果 knowledge 中某条事实有效且覆盖当前概念，将该 rule/fact 作为术语、单位、计算、筛选和输出规则依据；如果只有物理源绑定缺失，不要把整篇 knowledge 判 invalid。不可用、不足或冲突的部分用 `knowledge_issue` 与 `cross_validated_inference` 说明证据。

## 本任务执行约束

- 先读取由 question_structure、knowledge 和清单共同指向的最相关来源；一旦字段、粒度、覆盖范围和输出形态足够清楚，就调用 `analyze_plan`。
- `output_spec.columns` 只写最终答案列；排序、筛选、join、selector 和上下文字段放入 `execution_spec.supporting_fields`，不要作为最终输出列提交。
- 如果 knowledge 的逻辑表不在 SQLite 中，先用 `query_schema` 查看 `source_candidates`、`logical_bindings` 和 `binding_issues`，并检查同 basename 的 CSV/JSON/doc 来源；不要直接声明 knowledge invalid，也不要退到语义相邻但字段不同的来源。
- 若 `question_structure.conditions.calculations` 为空且没有 `intent_operators` 授权 `aggregate/derive`，不要把地域、范围或“记录”解释为聚合/派生请求。
- `cross_validated_inference` 和 `intent.unresolved` 只能说明已观察事实、口径不匹配或待确认事项；不要把未授权操作写成“如果没有 X 就计算/筛选/排序/限制/重塑 Y”。可执行操作必须在 `output_spec.transformations` 或 `execution_spec.operations` 中声明并引用用户、knowledge 或 KnowledgeFact 授权。
- 若无显式转换授权，保留源行、源顺序和空值；只投影用户要求返回的目标字段。
- 字典表要匹配事件表领域：`D_LABITEMS` 对应 `LABEVENTS`；`D_ITEMS.LINKSTO` 指向它关联的事件表或文件。
- 最终在一次 `execute_python` 调用中验证并调用 `set_answer(columns, rows)`。
