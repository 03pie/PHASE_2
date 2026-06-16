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

如果 knowledge 有效且覆盖当前概念，将其作为术语、单位、计算、筛选和输出规则的权威标准；如果不可用、不足或与实际 schema 冲突，在 `analyze_plan` 中设置 `knowledge_rules=[]`，并用 `knowledge_issue` 与 `cross_validated_inference` 说明证据。

## 本任务执行约束

- 先读取由 question_structure、knowledge 和清单共同指向的最相关来源；一旦字段、粒度、覆盖范围和输出形态足够清楚，就调用 `analyze_plan`。
- 若 `question_structure.conditions.calculations` 为空，不要把地域、范围或“记录”解释为聚合/派生请求。
- `cross_validated_inference` 和 `intent.unresolved` 只能说明已观察事实、口径不匹配或待确认事项；不要把未授权操作写成“如果没有 X 就计算/筛选/排序/限制/重塑 Y”。可执行操作必须在 `output_spec.transformations` 中声明并引用用户或 knowledge 授权。
- 若无显式转换授权，保留源行、源顺序和空值；只投影目标字段以及必要的源记录上下文键。
- 最终在一次 `execute_python` 调用中验证并调用 `set_answer(columns, rows)`。
