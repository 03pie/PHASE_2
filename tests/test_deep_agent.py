from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from data_agent_baseline.agents.deep_agent import DeepAgent
from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.agents.filesystem import Utf8FilesystemBackend
from data_agent_baseline.agents.middleware import _canonical_knowledge_quote
from data_agent_baseline.agents.middleware import _discovery_state
from data_agent_baseline.agents.question_structure import _normalize_structure
from data_agent_baseline.agents.semantic_layer import (
    parse_knowledge_content,
    query_semantic_context,
)
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools.agent_tools.analyze_plan import analyze_plan_tool
from data_agent_baseline.tools.agent_tools.execute_python import (
    create_execute_python_tool,
)
from data_agent_baseline.tools.agent_tools.extract_narrative_records import (
    create_extract_narrative_records_tool,
)
from data_agent_baseline.tools.agent_tools.inspect_sqlite import (
    create_inspect_sqlite_tool,
)
from data_agent_baseline.tools.agent_tools.grep_file import create_grep_file_tool
from data_agent_baseline.tools.agent_tools.query_schema import create_query_schema_tool
from data_agent_baseline.tools.agent_tools.read_csv import create_read_csv_tool
from data_agent_baseline.tools.agent_tools.read_doc import create_read_doc_tool
from data_agent_baseline.tools.agent_tools.read_json import create_read_json_tool
from data_agent_baseline.tools.answer import validate_prepared_answer

DISCOVERY_TOOLS = {
    "execute_sql",
    "execute_python",
    "grep_file",
    "inspect_sqlite",
    "query_schema",
    "read_csv",
    "read_doc",
    "read_json",
}
PLAN_TOOLS = DISCOVERY_TOOLS | {"analyze_plan"}
PLAN_ONLY_TOOLS = {"analyze_plan"}


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _plan_args(
    *,
    requirement_quote: str = "Return the observed value.",
    requirement_type: str = "measure",
    knowledge_status: str = "authoritative",
    knowledge_quote: str = "Use the observed source value exactly.",
    context_paths: list[str] | None = None,
    columns: list[tuple[str, list[str]]] | None = None,
    transformations: list[dict[str, Any]] | None = None,
    expected_row_count: int | None = None,
    steps: list[str] | None = None,
    version: int = 1,
    changed_fields: list[str] | None = None,
    evidence_changes: list[str] | None = None,
) -> dict[str, Any]:
    paths = context_paths or ["/context/data.txt"]
    plan_steps = steps or ["Compute and validate", "Submit the result"]
    plan_transformations = transformations or []
    authoritative = knowledge_status == "authoritative"
    return {
        "schema_version": "1.0",
        "intent": {
            "requirements": [
                {
                    "statement": "Return the requested source value.",
                    "requirement_type": requirement_type,
                    "quote": requirement_quote,
                }
            ],
            "unresolved": [],
        },
        "output_spec": {
            "columns": [
                {"name": name, "source_fields": source_fields}
                for name, source_fields in (columns or [("value", ["value"])])
            ],
            "row_grain": "one source record",
            "row_policy": "transform" if plan_transformations else "preserve",
            "transformations": plan_transformations,
            "ordering": "unspecified" if plan_transformations else "source",
            "sort_keys": [],
            "null_policy": "preserve",
            "expected_row_count": expected_row_count,
        },
        "evidence": {
            "knowledge_status": knowledge_status,
            "knowledge_rules": (
                [
                    {
                        "rule_type": "semantic",
                        "quote": knowledge_quote,
                        "source_path": "/context/knowledge.md",
                    }
                ]
                if authoritative
                else []
            ),
            "knowledge_issue": (
                "" if authoritative else "Knowledge conflicts with the observed schema."
            ),
            "context_sources": [
                {
                    "path": path,
                    "observations": [f"{path} contains the inspected source data."],
                }
                for path in paths
            ],
            "cross_validated_inference": (
                "" if authoritative else "Use the field shared by the inspected sources."
            ),
        },
        "revision": {
            "version": version,
            "reason": "Initial evidence-based plan." if version == 1 else "Evidence changed.",
            "evidence_changes": evidence_changes or [],
            "changed_fields": changed_fields or [],
        },
        "steps": plan_steps,
        "delegation_candidates": [],
    }


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]
    auto_discovery_plan: bool = True
    request_quote: str = "Return the observed value."
    knowledge_quote: str = "Use the observed source value exactly."
    call_count: int = 0
    bound_tool_sets: list[set[str]] = Field(default_factory=list)
    bound_tool_descriptions: list[dict[str, str]] = Field(default_factory=list)
    bound_tool_choices: list[str | None] = Field(default_factory=list)
    system_texts: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling-model"

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if not self.auto_discovery_plan:
            return
        discovery_and_plan = [
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "schema-call",
                content="Inspect a relevant candidate data source.",
            ),
            AIMessage(
                content="Create an evidence-based execution plan.",
                tool_calls=[
                    {
                        "name": "analyze_plan",
                        "args": _plan_args(
                            requirement_quote=self.request_quote,
                            knowledge_quote=self.knowledge_quote,
                        ),
                        "id": "plan-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="Convert the evidence-based plan into actionable todos.",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {
                            "todos": [
                                {"content": "Compute and validate", "status": "in_progress"},
                                {"content": "Submit the result", "status": "pending"},
                            ]
                        },
                        "id": "todos-call",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
        self.responses[0:0] = discovery_and_plan

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ScriptedChatModel:
        del kwargs
        names = {
            str(getattr(tool, "name", tool.get("name") if isinstance(tool, dict) else ""))
            for tool in tools
        }
        descriptions = {}
        for tool in tools:
            if isinstance(tool, dict):
                name = str(tool.get("name") or tool.get("function", {}).get("name") or "")
                function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
                description = str(tool.get("description") or function.get("description") or "")
            else:
                name = str(getattr(tool, "name", ""))
                description = str(getattr(tool, "description", "") or "")
            if name:
                descriptions[name] = description
        self.bound_tool_sets.append(names)
        self.bound_tool_descriptions.append(descriptions)
        self.bound_tool_choices.append(tool_choice)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        system_text = "\n\n".join(
            _message_text(message)
            for message in messages
            if isinstance(message, SystemMessage)
        )
        self.system_texts.append(system_text)
        if self.call_count >= len(self.responses):
            raise RuntimeError("No scripted model responses remaining.")
        message = self.responses[self.call_count]
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


class FailingChatModel(ScriptedChatModel):
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        raise RuntimeError("boom")


def _tool_response(
    name: str,
    args: dict[str, Any],
    tool_call_id: str,
    *,
    content: str = "",
) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": tool_call_id,
                "type": "tool_call",
            }
        ],
    )


def _answer_response(
    *,
    columns: list[str],
    rows: list[list[Any]],
    tool_call_id: str = "answer-call",
) -> AIMessage:
    payload = json.dumps(
        {"columns": columns, "rows": rows},
        ensure_ascii=False,
    )
    return _tool_response(
        "execute_python",
        {
            "code": (
                "import json\n"
                f"payload = json.loads({payload!r})\n"
                "columns = payload['columns']\n"
                "rows = payload['rows']\n"
                "audit = {\n"
                "    'source_paths': ['/context/data.txt'],\n"
                "    'operations': ['scripted_result'],\n"
                "    'output_row_count': len(rows),\n"
                "    'output_hash': answer_hash(columns, rows),\n"
                "}\n"
                "set_answer(columns, rows, audit=audit)\n"
            )
        },
        tool_call_id,
        content="The result table is ready.",
    )


def _question_structure_response(
    *,
    target_quote: str = "Return the observed value.",
    target_constraints: list[dict[str, Any]] | None = None,
) -> AIMessage:
    return AIMessage(
        content=json.dumps(
            {
                "schema_version": "1.0",
                "original_question": target_quote,
                "targets": [
                    {
                        "quote": target_quote,
                        "name": "observed value",
                        "target_type": "measure",
                        "description": "Return the observed value.",
                    }
                ],
                "target_constraints": target_constraints or [],
                "conditions": {
                    "filters": [],
                    "time_ranges": [],
                    "groupings": [],
                    "orderings": [],
                    "limits": [],
                    "calculations": [],
                    "output_columns": [],
                },
                "output": {
                    "row_grain_hint": "source_records",
                    "requested_columns": [],
                    "preserve_source_rows": "true",
                },
                "ambiguities": [],
            },
            ensure_ascii=False,
        )
    )


def _llm_steps(result: Any) -> list[Any]:
    return [
        step
        for step in result.steps
        if "message" in step.raw_response
    ]


def _tool_calls(result: Any) -> list[tuple[Any, dict[str, Any]]]:
    return [
        (step, tool_call)
        for step in _llm_steps(result)
        for tool_call in step.observation.get("tool_calls", [])
    ]


def _tool_call(result: Any, tool_call_id: str) -> tuple[Any, dict[str, Any]]:
    return next(
        (step, tool_call)
        for step, tool_call in _tool_calls(result)
        if tool_call.get("tool_call_id") == tool_call_id
    )


def _command_tool_message(result: Any) -> ToolMessage:
    if isinstance(result, ToolMessage):
        return result
    update = getattr(result, "update", None)
    if isinstance(update, dict):
        messages = update.get("messages")
        if isinstance(messages, list) and messages and isinstance(messages[0], ToolMessage):
            return messages[0]
    raise AssertionError(f"Expected ToolMessage or Command with ToolMessage, got {result!r}")


@pytest.fixture
def public_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "data.txt").write_text("hello from context\n", encoding="utf-8")
    (context_dir / "knowledge.md").write_text(
        "Use the observed source value exactly.\n",
        encoding="utf-8",
    )
    (context_dir / "sample.csv").write_text("name,value\nalpha,1\n", encoding="utf-8")
    (context_dir / "sample.json").write_text('{"value": 1}\n', encoding="utf-8")
    with closing(sqlite3.connect(context_dir / "sample.sqlite")) as connection:
        connection.execute("CREATE TABLE metrics (name TEXT, value REAL)")
        connection.execute("CREATE TABLE observations (observed_at TEXT)")
        connection.commit()
    (context_dir / "sample.pdf").write_bytes(b"%PDF-1.4\n")
    (context_dir / "sample.mp4").write_bytes(b"video")
    return PublicTask(
        record=TaskRecord(
            task_id="task_1",
            difficulty="easy",
            question="Return the observed value.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_reads_context_and_submits_answer(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "read-call",
            ),
            _answer_response(columns=["value"], rows=[["hello from context"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.to_dict() == {
        "columns": ["value"],
        "rows": [["hello from context"]],
    }
    assert [step.action for step in result.steps[:4]] == [
        "system_prompt",
        "user_prompt",
        "read_doc",
        "analyze_plan",
    ]
    assert [
        step.action
        for step in result.steps
    ][:5] == [
        "system_prompt",
        "user_prompt",
        "read_doc",
        "analyze_plan",
        "write_todos",
    ]
    tool_calls = _tool_calls(result)
    assert [tool_call["name"] for _, tool_call in tool_calls] == [
        "read_doc",
        "analyze_plan",
        "write_todos",
        "read_doc",
        "execute_python",
    ]
    assert [tool_call["tool_call_id"] for _, tool_call in tool_calls] == [
        "schema-call",
        "plan-call",
        "todos-call",
        "read-call",
        "answer-call",
    ]
    system_prompt_steps = [
        step
        for step in result.steps
        if step.action == "system_prompt"
        and step.action_input.get("scope") == "main"
    ]
    assert len(system_prompt_steps) == 1
    tool_steps = [
        step
        for step in _llm_steps(result)
        if step.observation.get("tool_calls")
    ]
    assert all(
        set(step.raw_response) == {"llm_input", "message"}
        and "request" not in step.action_input
        and "tools" not in step.action_input
        for step in _llm_steps(result)
    )
    assert all("message" not in step.action_input for step in tool_steps)
    assert tool_steps[0].action_input["tool_calls"] == [
        {
            "name": "read_doc",
            "tool_call_id": "schema-call",
            "args": {"path": "/context/data.txt"},
        }
    ]
    first_llm_input = tool_steps[0].raw_response["llm_input"]
    serialized_llm_input = json.dumps(
        first_llm_input,
        ensure_ascii=False,
    )
    assert first_llm_input["scope"] == "main"
    assert "messages" in first_llm_input
    assert "tools" in first_llm_input
    assert "任务问题" in serialized_llm_input
    assert "SQLite tables: metrics, observations" in serialized_llm_input
    assert "content_preview" not in serialized_llm_input
    user_prompt = next(step for step in result.steps if step.action == "user_prompt")
    prompt_content = user_prompt.action_input["message"]["content"]
    assert "sample.sqlite" in prompt_content
    assert "SQLite tables: metrics, observations" in prompt_content


def test_question_structure_node_is_isolated_and_injected(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _question_structure_response(target_quote="Return the observed value."),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "structured-source",
            ),
            _tool_response("analyze_plan", _plan_args(), "structured-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "structured-todos",
            ),
            _answer_response(
                columns=["value"],
                rows=[["hello from context"]],
                tool_call_id="structured-answer",
            ),
        ],
    )

    result = DeepAgent(
        model=model,
        config=DeepAgentConfig(question_structure_enabled=True),
    ).run(public_task)

    assert result.succeeded
    question_step = result.steps[0]
    assert question_step.action == "question_structure"
    assert question_step.action_input == {"question": "Return the observed value."}
    assert "context" not in json.dumps(question_step.action_input)
    assert question_step.observation["structure"]["targets"][0]["quote"] == (
        "Return the observed value."
    )
    user_prompt = next(step for step in result.steps if step.action == "user_prompt")
    prompt_text = user_prompt.action_input["message"]["content"]
    assert "<question_structure>" in prompt_text
    assert '"target_type": "measure"' in prompt_text

    assert not prompt_text.startswith("Question:")


def test_question_structure_normalizes_unrequested_aggregate_grain() -> None:
    normalized = _normalize_structure(
        {
            "schema_version": "1.0",
            "original_question": "显示这些记录",
            "targets": [
                {
                    "quote": "记录",
                    "name": "records",
                    "target_type": "record_set",
                    "description": "Show records.",
                }
            ],
            "target_constraints": [
                {
                    "quote": "记录",
                    "constraint_type": "output_shape",
                    "value": "records",
                    "explicitness": "explicit",
                }
            ],
            "conditions": {
                "filters": [],
                "time_ranges": [],
                "groupings": [],
                "orderings": [],
                "limits": [],
                "calculations": [],
                "output_columns": [],
            },
            "output": {
                "row_grain_hint": "aggregated_records",
                "requested_columns": [],
                "preserve_source_rows": "unknown",
            },
            "ambiguities": [],
        },
        "显示这些记录",
    )

    assert normalized["output"]["row_grain_hint"] == "source_records"
    assert normalized["output"]["preserve_source_rows"] == "true"


def test_question_structure_keeps_unquoted_condition_strings_as_hints() -> None:
    normalized = _normalize_structure(
        {
            "schema_version": "1.0",
            "original_question": "Show the education distribution.",
            "targets": [
                {
                    "quote": "education distribution",
                    "name": "education distribution",
                    "target_type": "record_set",
                    "description": "Return the distribution.",
                }
            ],
            "conditions": {
                "filters": ["fund size > 100 billion"],
                "calculations": ["group by education"],
            },
            "output": {},
        },
        "Show the education distribution.",
    )

    assert normalized["conditions"]["filters"] == [
        {
            "quote": None,
            "value": "fund size > 100 billion",
            "condition_type": "filter",
            "explicitness": "unquoted_hint",
        }
    ]
    assert normalized["conditions"]["calculations"][0]["quote"] is None
    assert {
        (item["quote"], item["operation"], item["operator_type"])
        for item in normalized["intent_operators"]
    } >= {("distribution", "aggregate", "distribution")}


def test_semantic_layer_extracts_facts_and_same_basename_sources(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True)
    context_dir.joinpath("knowledge.md").write_text(
        "\n".join(
            [
                "# Table `mf_missing`",
                "| Column | Semantic Definition | Unit |",
                "|---|---|---|",
                "| `RRInTenYear` | return rate in ten years | % |",
                "Join `fundcode` to `mf_fundarchives`.",
                "Formula: `RRInTenYear` is already provided by the source.",
                "```sql",
                "SELECT Education, COUNT(*) FROM mf_personalinfo WHERE ExperienceTime > 10 GROUP BY Education",
                "```",
            ]
        ),
        encoding="utf-8",
    )
    doc_dir.joinpath("mf_missing.md").write_text(
        "Narrative records contain RRInTenYear values.",
        encoding="utf-8",
    )

    facts = parse_knowledge_content(
        context_dir.joinpath("knowledge.md").read_text(encoding="utf-8")
    )
    assert {"field", "unit", "join", "calculation", "example_query"} <= {
        fact.kind for fact in facts
    }
    assert any(fact.operation == "filter,aggregate" for fact in facts)

    semantic = query_semantic_context(context_dir, "mf_missing", max_matches=10)
    assert any(
        item["source_type"] == "doc"
        and item["source_path"] == "/context/doc/mf_missing.md"
        for item in semantic["source_candidates"]
    )
    assert any("status=narrative_only" in issue for issue in semantic["binding_issues"])


def test_semantic_layer_uses_unicode_terms_for_narrative_evidence(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True)
    context_dir.joinpath("knowledge.md").write_text(
        "\n".join(
            [
                "# Table `mf_netvalueperformancehis`",
                "| Column | Semantic Definition | Unit |",
                "|---|---|---|",
                "| `RRInTenYear` | 近十年累计回报率 | % |",
            ]
        ),
        encoding="utf-8",
    )
    doc_dir.joinpath("mf_netvalueperformancehis.md").write_text(
        "\n".join(
            [
                "短期表现摘要。",
                "档案 266 的十年回报率为 166.097944%。",
            ]
        ),
        encoding="utf-8",
    )

    semantic = query_semantic_context(context_dir, "十年", max_matches=10)

    candidate = next(
        item
        for item in semantic["source_candidates"]
        if item["source_path"] == "/context/doc/mf_netvalueperformancehis.md"
    )
    assert any(
        "十年回报率" in line["content"]
        for line in candidate.get("line_evidence", [])
    )


def test_query_schema_observes_narrative_line_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    context_dir = workspace / "context"
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True)
    context_dir.joinpath("knowledge.md").write_text(
        "\n".join(
            [
                "# Table `mf_netvalueperformancehis`",
                "| Column | Semantic Definition | Unit |",
                "|---|---|---|",
                "| `RRInTenYear` | 近十年累计回报率 | % |",
            ]
        ),
        encoding="utf-8",
    )
    doc_dir.joinpath("mf_netvalueperformancehis.md").write_text(
        "档案 266 的十年回报率为 166.097944%。",
        encoding="utf-8",
    )
    tool = create_query_schema_tool(
        workspace,
        DeepAgentConfig(max_output_bytes=100_000),
    )

    command = tool.invoke(
        {
            "type": "tool_call",
            "name": "query_schema",
            "id": "schema-narrative",
            "args": {"field": "十年", "state": {}},
        }
    )

    message = _command_tool_message(command)
    assert message.status == "success"
    payload = json.loads(message.content)
    assert any(
        candidate["source_path"] == "/context/doc/mf_netvalueperformancehis.md"
        for candidate in payload["source_candidates"]
    )
    observed = getattr(command, "update", {})["observed_sources"]
    assert observed[0]["path"] == "/context/doc/mf_netvalueperformancehis.md"
    assert observed[0]["observed_by"] == "query_schema"
    assert "十年回报率" in observed[0]["matched_lines"][0]["content"]


def test_grep_file_observes_matched_sources(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    context_dir = workspace / "context" / "doc"
    context_dir.mkdir(parents=True)
    context_dir.joinpath("performance.md").write_text(
        "档案 266 的十年回报率为 166.097944%。",
        encoding="utf-8",
    )
    tool = create_grep_file_tool(
        workspace,
        DeepAgentConfig(max_output_bytes=100_000),
    )

    command = tool.invoke(
        {
            "type": "tool_call",
            "name": "grep_file",
            "id": "grep-narrative",
            "args": {
                "pattern": "十年回报率",
                "path": "/context/doc/performance.md",
                "output_mode": "files_with_matches",
                "state": {},
            },
        }
    )

    message = _command_tool_message(command)
    assert message.status == "success"
    observed = getattr(command, "update", {})["observed_sources"]
    assert observed[0]["path"] == "/context/doc/performance.md"
    assert observed[0]["match_count"] == 1
    assert observed[0]["observed_by"] == "grep_file"
    assert "166.097944" in observed[0]["matched_lines"][0]["content"]


def test_grep_file_searches_pdf_and_suggests_read_doc_slice(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    workspace = tmp_path / "workspace"
    context_dir = workspace / "context" / "doc"
    context_dir.mkdir(parents=True)
    pdf_path = context_dir / "report.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "\n".join(
            [
                "intro line",
                "target metric appears here",
                "following line",
            ]
        ),
    )
    document.save(pdf_path)
    document.close()
    tool = create_grep_file_tool(
        workspace,
        DeepAgentConfig(max_output_bytes=100_000),
    )

    command = tool.invoke(
        {
            "type": "tool_call",
            "name": "grep_file",
            "id": "grep-pdf",
            "args": {
                "pattern": "target metric",
                "path": "/context/doc/report.pdf",
                "output_mode": "content",
                "state": {},
            },
        }
    )

    message = _command_tool_message(command)
    payload = json.loads(message.content)
    assert message.status == "success"
    assert payload["filenames"] == ["/context/doc/report.pdf"]
    assert payload["read_doc_slices"][0]["path"] == "/context/doc/report.pdf"
    assert payload["read_doc_slices"][0]["start_line"] == 0
    assert payload["paging_unit"] == "matched_lines"
    assert payload["total_matches"] == 1
    assert payload["has_more"] is False
    observed = getattr(command, "update", {})["observed_sources"]
    assert observed[0]["source_type"] == "doc"
    assert "target metric" in observed[0]["matched_lines"][0]["content"]


def test_query_schema_narrative_observation_can_drive_plan(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_narrative"
    context_dir = task_dir / "context"
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True)
    context_dir.joinpath("knowledge.md").write_text(
        "\n".join(
            [
                "Use the observed source value exactly.",
                "# Table `mf_netvalueperformancehis`",
                "| Column | Semantic Definition | Unit |",
                "|---|---|---|",
                "| `RRInTenYear` | 近十年累计回报率 | % |",
            ]
        ),
        encoding="utf-8",
    )
    doc_dir.joinpath("mf_netvalueperformancehis.md").write_text(
        "档案 266 的十年回报率为 166.097944%。",
        encoding="utf-8",
    )
    question = "我能看一下基金的十年回报率数据吗"
    plan = _plan_args(
        requirement_quote=question,
        knowledge_quote="| `RRInTenYear` | 近十年累计回报率 | % |",
        context_paths=["/context/doc/mf_netvalueperformancehis.md"],
        columns=[("RRInTenYear", ["RRInTenYear"])],
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "query_schema",
                {"field": "十年"},
                "schema-narrative-plan",
            ),
            _tool_response("analyze_plan", plan, "narrative-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "narrative-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "set_answer(\n"
                        "    ['RRInTenYear'],\n"
                        "    [[166.097944]],\n"
                        "    audit={\n"
                        "        'source_paths': "
                        "['/context/doc/mf_netvalueperformancehis.md'],\n"
                        "        'operations': ['source_bound_projection'],\n"
                        "    },\n"
                        ")\n"
                    )
                },
                "narrative-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(
        PublicTask(
            record=TaskRecord(
                task_id="task_narrative",
                difficulty="easy",
                question=question,
            ),
            assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
        )
    )

    assert result.succeeded
    _, plan_call = _tool_call(result, "narrative-plan")
    assert plan_call["ok"] is True
    _, schema_call = _tool_call(result, "schema-narrative-plan")
    observed = schema_call["result"]["update"]["observed_sources"]
    assert observed[0]["path"] == "/context/doc/mf_netvalueperformancehis.md"


def test_observed_narrative_source_blocks_unavailable_plan(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_narrative_guard"
    context_dir = task_dir / "context"
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True)
    context_dir.joinpath("knowledge.md").write_text(
        "\n".join(
            [
                "Use the observed source value exactly.",
                "# Table `mf_netvalueperformancehis`",
                "| Column | Semantic Definition | Unit |",
                "|---|---|---|",
                "| `RRInTenYear` | 近十年累计回报率 | % |",
            ]
        ),
        encoding="utf-8",
    )
    doc_dir.joinpath("mf_netvalueperformancehis.md").write_text(
        "档案 266 的十年回报率为 166.097944%。",
        encoding="utf-8",
    )
    question = "我能看一下基金的十年回报率数据吗"
    invalid_plan = _plan_args(
        requirement_quote=question,
        knowledge_status="unavailable",
        context_paths=["/context/doc/mf_netvalueperformancehis.md"],
        columns=[("message", [])],
    )
    invalid_plan["evidence"]["knowledge_issue"] = "rrintenyear is narrative_only."
    invalid_plan["evidence"]["cross_validated_inference"] = (
        "Claim the requested data is unavailable."
    )
    valid_plan = _plan_args(
        requirement_quote=question,
        knowledge_quote="| `RRInTenYear` | 近十年累计回报率 | % |",
        context_paths=["/context/doc/mf_netvalueperformancehis.md"],
        columns=[("RRInTenYear", ["RRInTenYear"])],
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "query_schema",
                {"field": "十年"},
                "schema-narrative-guard",
            ),
            _tool_response("analyze_plan", invalid_plan, "unavailable-plan"),
            _tool_response("analyze_plan", valid_plan, "available-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "narrative-guard-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "set_answer(\n"
                        "    ['RRInTenYear'],\n"
                        "    [[166.097944]],\n"
                        "    audit={\n"
                        "        'source_paths': "
                        "['/context/doc/mf_netvalueperformancehis.md'],\n"
                        "        'operations': ['source_bound_projection'],\n"
                        "    },\n"
                        ")\n"
                    )
                },
                "narrative-guard-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(
        PublicTask(
            record=TaskRecord(
                task_id="task_narrative_guard",
                difficulty="easy",
                question=question,
            ),
            assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
        )
    )

    assert result.succeeded
    _, invalid_call = _tool_call(result, "unavailable-plan")
    assert invalid_call["status"] == "error"
    assert "observed narrative sources" in invalid_call["result"]["content"]
    _, valid_call = _tool_call(result, "available-plan")
    assert valid_call["ok"] is True


def test_read_doc_narrative_binding_blocks_unavailable_plan(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_read_doc_narrative_guard"
    context_dir = task_dir / "context"
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True)
    context_dir.joinpath("knowledge.md").write_text(
        "\n".join(
            [
                "Use the observed source value exactly.",
                "# Table `mf_netvalueperformancehis`",
                "| Column | Semantic Definition | Unit |",
                "|---|---|---|",
                "| `RRInTenYear` | ten-year cumulative return rate | % |",
            ]
        ),
        encoding="utf-8",
    )
    doc_dir.joinpath("mf_netvalueperformancehis.md").write_text(
        "\n".join(
            [
                "The opening line is background material.",
                "Archive 266 has a ten-year return rate of 166.097944%.",
            ]
        ),
        encoding="utf-8",
    )
    question = "Show me the fund ten-year return rate data."
    invalid_plan = _plan_args(
        requirement_quote=question,
        knowledge_status="unavailable",
        context_paths=["/context/doc/mf_netvalueperformancehis.md"],
        columns=[("message", [])],
    )
    invalid_plan["evidence"]["knowledge_issue"] = "rrintenyear is narrative_only."
    valid_plan = _plan_args(
        requirement_quote=question,
        knowledge_quote="| `RRInTenYear` | ten-year cumulative return rate | % |",
        context_paths=["/context/doc/mf_netvalueperformancehis.md"],
        columns=[("RRInTenYear", ["RRInTenYear"])],
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/doc/mf_netvalueperformancehis.md", "max_lines": 1},
                "read-doc-narrative-guard",
            ),
            _tool_response("analyze_plan", invalid_plan, "read-doc-unavailable-plan"),
            _tool_response("analyze_plan", valid_plan, "read-doc-available-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "read-doc-narrative-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "set_answer(\n"
                        "    ['RRInTenYear'],\n"
                        "    [[166.097944]],\n"
                        "    audit={\n"
                        "        'source_paths': "
                        "['/context/doc/mf_netvalueperformancehis.md'],\n"
                        "        'operations': ['source_bound_projection'],\n"
                        "    },\n"
                        ")\n"
                    )
                },
                "read-doc-narrative-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(
        PublicTask(
            record=TaskRecord(
                task_id="task_read_doc_narrative_guard",
                difficulty="easy",
                question=question,
            ),
            assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
        )
    )

    assert result.succeeded
    _, invalid_call = _tool_call(result, "read-doc-unavailable-plan")
    assert invalid_call["status"] == "error"
    assert "observed narrative sources" in invalid_call["result"]["content"]
    _, valid_call = _tool_call(result, "read-doc-available-plan")
    assert valid_call["ok"] is True


def test_knowledge_field_fact_canonicalizes_semantic_neighbor_source_field(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task_knowledge_field_guard"
    context_dir = task_dir / "context"
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True)
    context_dir.joinpath("knowledge.md").write_text(
        "\n".join(
            [
                "| Table | Column | Semantic Definition |",
                "|---|---|---|",
                "| `mf_netvalueperformancehis` | `rrintenyear` | ten-year cumulative return rate (%) |",
            ]
        ),
        encoding="utf-8",
    )
    doc_dir.joinpath("mf_netvalueperformancehis.md").write_text(
        "Archive 266 has a ten-year return rate of 166.097944%.",
        encoding="utf-8",
    )
    question = "Show me the fund ten-year return rate data."
    invalid_plan = _plan_args(
        requirement_quote=question,
        knowledge_quote="| `rrintenyear` | ten-year cumulative return rate (%) |",
        context_paths=["/context/doc/mf_netvalueperformancehis.md"],
        columns=[("FundReturn", ["FundReturn"])],
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/doc/mf_netvalueperformancehis.md"},
                "knowledge-field-guard-doc",
            ),
            _tool_response("analyze_plan", invalid_plan, "semantic-neighbor-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "knowledge-field-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "set_answer(\n"
                        "    ['RRInTenYear'],\n"
                        "    [[166.097944]],\n"
                        "    audit={\n"
                        "        'source_paths': "
                        "['/context/doc/mf_netvalueperformancehis.md'],\n"
                        "        'operations': ['source_bound_projection'],\n"
                        "    },\n"
                        ")\n"
                    )
                },
                "knowledge-field-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(
        PublicTask(
            record=TaskRecord(
                task_id="task_knowledge_field_guard",
                difficulty="easy",
                question=question,
            ),
            assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
        )
    )

    assert result.succeeded
    _, plan_call = _tool_call(result, "semantic-neighbor-plan")
    assert plan_call["ok"] is True
    plan = plan_call["result"]["update"]["analysis_plan"]
    assert plan["output_spec"]["columns"] == [
        {"name": "FundReturn", "source_fields": ["rrintenyear"]}
    ]
    assert plan["execution_spec"]["source_bindings"] == [
        {
            "fact_id": "kf_1",
            "logical_table": "mf_netvalueperformancehis",
            "source_field": "rrintenyear",
            "source_paths": ["/context/doc/mf_netvalueperformancehis.md"],
        }
    ]
    assert any(
        rule.get("fact_id") == "kf_1"
        for rule in plan["evidence"]["knowledge_rules"]
    )


def test_observed_target_field_fact_must_bind_narrative_source(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task_target_field_binding"
    context_dir = task_dir / "context"
    doc_dir = context_dir / "doc"
    json_dir = context_dir / "json"
    doc_dir.mkdir(parents=True)
    json_dir.mkdir()
    context_dir.joinpath("knowledge.md").write_text(
        "\n".join(
            [
                "| Table | Column | Semantic Definition |",
                "|---|---|---|",
                "| `mf_netvalueperformancehis` | `rrintenyear` | ten-year return rate |",
                "| `mf_fundarchives` | `secuabbr` | fund display name |",
            ]
        ),
        encoding="utf-8",
    )
    doc_dir.joinpath("mf_netvalueperformancehis.md").write_text(
        "Archive 266 has a ten-year return rate of 166.097944%.\n",
        encoding="utf-8",
    )
    json_dir.joinpath("mf_fundreturnrank.json").write_text(
        json.dumps(
            {
                "records": [
                    {"IndexCycle": "ten-year", "FundReturn": 75.07, "SecuAbbr": "ETF"}
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    question = "Show me the fund ten-year return rate data."
    invalid_plan = _plan_args(
        requirement_quote=question,
        knowledge_quote="| `mf_fundarchives` | `secuabbr` | fund display name |",
        context_paths=["/context/json/mf_fundreturnrank.json"],
        columns=[("return_rate", ["FundReturn"])],
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/doc/mf_netvalueperformancehis.md"},
                "target-field-doc",
            ),
            _tool_response(
                "read_json",
                {"path": "/context/json/mf_fundreturnrank.json"},
                "target-field-json",
            ),
            _tool_response("analyze_plan", invalid_plan, "omitted-target-field-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Extract source-bound field", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "target-field-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "set_answer(\n"
                        "    ['RRInTenYear'],\n"
                        "    [[166.097944]],\n"
                        "    audit={\n"
                        "        'source_paths': "
                        "['/context/doc/mf_netvalueperformancehis.md'],\n"
                        "        'operations': ['source_bound_projection'],\n"
                        "    },\n"
                        ")\n"
                    )
                },
                "target-field-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(
        PublicTask(
            record=TaskRecord(
                task_id="task_target_field_binding",
                difficulty="easy",
                question=question,
            ),
            assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
        )
    )

    assert result.succeeded
    _, plan_call = _tool_call(result, "omitted-target-field-plan")
    assert plan_call["ok"] is True
    plan = plan_call["result"]["update"]["analysis_plan"]
    assert {
        source["path"] for source in plan["execution_spec"]["sources"]
    } == {"/context/doc/mf_netvalueperformancehis.md"}
    assert {
        source["path"] for source in plan["evidence"]["context_sources"]
    } == {
        "/context/json/mf_fundreturnrank.json",
        "/context/doc/mf_netvalueperformancehis.md",
    }
    assert plan["execution_spec"]["source_bindings"] == [
        {
            "fact_id": "kf_1",
            "logical_table": "mf_netvalueperformancehis",
            "source_field": "rrintenyear",
            "source_paths": ["/context/doc/mf_netvalueperformancehis.md"],
        }
    ]

    _, audit_error = validate_prepared_answer(
        ["RRInTenYear"],
        [[75.07]],
        plan,
        {
            "source_paths": ["/context/json/mf_fundreturnrank.json"],
            "operations": ["filter(IndexCycle='ten-year')"],
        },
    )
    assert audit_error is not None
    assert "source_bindings" in audit_error

    _, mixed_audit_error = validate_prepared_answer(
        ["RRInTenYear"],
        [[75.07]],
        plan,
        {
            "source_paths": [
                "/context/doc/mf_netvalueperformancehis.md",
                "/context/json/mf_fundreturnrank.json",
            ],
            "operations": ["copy source-bound value"],
        },
    )
    assert mixed_audit_error is not None
    assert "source-bound-only outputs" in mixed_audit_error


def test_target_field_fact_requires_binding_discovery_before_plan(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task_binding_discovery"
    context_dir = task_dir / "context"
    doc_dir = context_dir / "doc"
    json_dir = context_dir / "json"
    doc_dir.mkdir(parents=True)
    json_dir.mkdir()
    context_dir.joinpath("knowledge.md").write_text(
        "\n".join(
            [
                "| Table | Column | Semantic Definition |",
                "|---|---|---|",
                "| `mf_netvalueperformancehis` | `rrintenyear` | ten-year return rate |",
            ]
        ),
        encoding="utf-8",
    )
    doc_dir.joinpath("mf_netvalueperformancehis.md").write_text(
        "Archive 266 has a ten-year return rate of 166.097944%.\n",
        encoding="utf-8",
    )
    json_dir.joinpath("mf_fundreturnrank.json").write_text(
        json.dumps(
            {"records": [{"IndexCycle": "ten-year", "FundReturn": 75.07}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    question = "Show me the fund ten-year return rate data."
    json_only_plan = _plan_args(
        requirement_quote=question,
        knowledge_quote=(
            "| `mf_netvalueperformancehis` | `rrintenyear` | ten-year return rate |"
        ),
        context_paths=["/context/json/mf_fundreturnrank.json"],
        columns=[("RRInTenYear", ["rrintenyear"])],
    )
    valid_plan = _plan_args(
        requirement_quote=question,
        knowledge_quote=(
            "| `mf_netvalueperformancehis` | `rrintenyear` | ten-year return rate |"
        ),
        context_paths=["/context/doc/mf_netvalueperformancehis.md"],
        columns=[("RRInTenYear", ["rrintenyear"])],
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_json",
                {"path": "/context/json/mf_fundreturnrank.json"},
                "binding-discovery-json",
            ),
            _tool_response("analyze_plan", json_only_plan, "json-only-field-plan"),
            _tool_response(
                "read_doc",
                {"path": "/context/doc/mf_netvalueperformancehis.md"},
                "binding-discovery-doc",
            ),
            _tool_response("analyze_plan", valid_plan, "discovered-field-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Extract source-bound field", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "binding-discovery-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "set_answer(\n"
                        "    ['RRInTenYear'],\n"
                        "    [[166.097944]],\n"
                        "    audit={\n"
                        "        'source_paths': "
                        "['/context/doc/mf_netvalueperformancehis.md'],\n"
                        "        'operations': ['source_bound_projection'],\n"
                        "    },\n"
                        ")\n"
                    )
                },
                "binding-discovery-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(
        PublicTask(
            record=TaskRecord(
                task_id="task_binding_discovery",
                difficulty="easy",
                question=question,
            ),
            assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
        )
    )

    assert result.succeeded
    _, invalid_call = _tool_call(result, "json-only-field-plan")
    assert invalid_call["status"] == "error"
    assert "physical binding discovery" in invalid_call["result"]["content"]
    _, valid_call = _tool_call(result, "discovered-field-plan")
    assert valid_call["ok"] is True


def test_extract_narrative_records_preserves_source_bound_rows(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    context_dir = workspace / "context"
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True)
    context_dir.joinpath("knowledge.md").write_text(
        "\n".join(
            [
                "| Table | Column | Semantic Definition |",
                "|---|---|---|",
                "| `fund_perf` | `rrintenyear` | ten-year return rate |",
            ]
        ),
        encoding="utf-8",
    )
    doc_dir.joinpath("fund_perf.md").write_text(
        "\n".join(
            [
                "Archive 1 Alpha. The ten-year return rate was reviewed and confirmed as 12.345678%.",
                "Archive 2 Beta. The ten-year return rate data is missing.",
                "Archive 3 Gamma and Archive 4 Delta have ten-year return rate data unavailable.",
                "Since inception results start here. Archive 1 Alpha returned 99%.",
            ]
        ),
        encoding="utf-8",
    )
    plan = _plan_args(
        requirement_quote="Show ten-year return rate.",
        knowledge_quote="| `fund_perf` | `rrintenyear` | ten-year return rate |",
        context_paths=["/context/doc/fund_perf.md"],
        columns=[("RRInTenYear", ["rrintenyear"])],
        expected_row_count=4,
    )
    plan["execution_spec"] = {
        "sources": [
            {
                "path": "/context/doc/fund_perf.md",
                "source_type": "doc",
                "table_or_path": "/context/doc/fund_perf.md",
            }
        ],
        "supporting_fields": [],
        "operations": [],
        "source_bindings": [
            {
                "fact_id": "kf_1",
                "logical_table": "fund_perf",
                "source_field": "rrintenyear",
                "source_paths": ["/context/doc/fund_perf.md"],
            }
        ],
    }
    tool = create_extract_narrative_records_tool(
        workspace,
        DeepAgentConfig(max_output_bytes=100_000),
    )

    command = tool.invoke(
        {
            "type": "tool_call",
            "name": "extract_narrative_records",
            "id": "extract-narrative",
            "args": {
                "source_path": "/context/doc/fund_perf.md",
                "source_field": "rrintenyear",
                "column": "RRInTenYear",
                "state": {"analysis_plan": plan},
            },
        }
    )

    result = _command_tool_message(command)
    assert result.status == "success"
    prepared = getattr(command, "update", {})["prepared_answer"]
    assert prepared.columns == ["RRInTenYear"]
    assert prepared.rows == [["12.345678"], [""], [""], [""]]


def test_extract_narrative_records_tool_can_submit_after_plan(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task_extract_narrative"
    context_dir = task_dir / "context"
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True)
    context_dir.joinpath("knowledge.md").write_text(
        "\n".join(
            [
                "| Table | Column | Semantic Definition |",
                "|---|---|---|",
                "| `fund_perf` | `rrintenyear` | ten-year return rate |",
            ]
        ),
        encoding="utf-8",
    )
    doc_dir.joinpath("fund_perf.md").write_text(
        "\n".join(
            [
                "Archive 1 Alpha. The ten-year return rate was finally 12.345678%.",
                "Archive 2 Beta. The ten-year return rate is unavailable.",
            ]
        ),
        encoding="utf-8",
    )
    question = "Show ten-year return rate."
    plan = _plan_args(
        requirement_quote=question,
        knowledge_quote="| `fund_perf` | `rrintenyear` | ten-year return rate |",
        context_paths=["/context/doc/fund_perf.md"],
        columns=[("RRInTenYear", ["rrintenyear"])],
        expected_row_count=2,
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/doc/fund_perf.md"},
                "extract-source-doc",
            ),
            _tool_response("analyze_plan", plan, "extract-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Extract narrative rows", "status": "in_progress"},
                        {"content": "Submit", "status": "pending"},
                    ]
                },
                "extract-todos",
            ),
            _tool_response(
                "extract_narrative_records",
                {
                    "source_path": "/context/doc/fund_perf.md",
                    "source_field": "rrintenyear",
                    "column": "RRInTenYear",
                },
                "extract-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(
        PublicTask(
            record=TaskRecord(
                task_id="task_extract_narrative",
                difficulty="easy",
                question=question,
            ),
            assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
        )
    )

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.columns == ["RRInTenYear"]
    assert result.answer.rows == [["12.345678"], [""]]


def test_preserve_output_columns_must_cite_source_fields(
    public_task: PublicTask,
) -> None:
    invalid_plan = _plan_args(columns=[("value", [])])
    valid_plan = _plan_args(columns=[("value", ["value"])])
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "source-field-guard-doc",
            ),
            _tool_response("analyze_plan", invalid_plan, "missing-source-field-plan"),
            _tool_response("analyze_plan", valid_plan, "source-field-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "source-field-todos",
            ),
            _tool_response(
                "execute_python",
                {"code": "set_answer(['value'], [['hello from context']])\n"},
                "source-field-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, invalid_call = _tool_call(result, "missing-source-field-plan")
    assert invalid_call["status"] == "error"
    assert "must cite source_fields" in invalid_call["result"]["content"]
    _, valid_call = _tool_call(result, "source-field-plan")
    assert valid_call["ok"] is True


def test_question_structure_limits_plan_output_columns(
    public_task: PublicTask,
) -> None:
    invalid_plan = _plan_args(
        columns=[
            ("value", ["value"]),
            ("observed_at", ["observed_at"]),
        ]
    )
    invalid_plan["output_spec"]["columns"][1]["role"] = "context"
    invalid_plan["intent"]["requirements"].append(
        {
            "statement": "Try to add an extra output column.",
            "requirement_type": "output_column",
            "quote": "Return",
        }
    )
    valid_plan = _plan_args(columns=[("value", ["value"])])
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _question_structure_response(target_quote="Return the observed value."),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "limited-source",
            ),
            _tool_response("analyze_plan", invalid_plan, "limited-invalid-plan"),
            _tool_response("analyze_plan", valid_plan, "limited-valid-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "limited-todos",
            ),
            _answer_response(
                columns=["value"],
                rows=[["hello from context"]],
                tool_call_id="limited-answer",
            ),
        ],
    )

    result = DeepAgent(
        model=model,
        config=DeepAgentConfig(question_structure_enabled=True),
    ).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.columns == ["value"]
    _, invalid_call = _tool_call(result, "limited-invalid-plan")
    _, valid_call = _tool_call(result, "limited-valid-plan")
    assert invalid_call["status"] == "error"
    assert valid_call["ok"] is True


def test_question_structure_does_not_block_quoted_user_calculation(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task_total"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    context_dir.joinpath("data.txt").write_text("value\n1\n2\n", encoding="utf-8")
    context_dir.joinpath("knowledge.md").write_text(
        "Use the observed source value exactly.\n",
        encoding="utf-8",
    )
    task = PublicTask(
        record=TaskRecord(
            task_id="task_total",
            difficulty="easy",
            question="Return the total observed value.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    invalid_plan = _plan_args(
        requirement_quote="total",
        columns=[("value", ["value"])],
    )
    invalid_plan["intent"]["requirements"][0] = {
        "statement": "Aggregate the observed value.",
        "requirement_type": "calculation",
        "quote": "total",
    }
    invalid_plan["output_spec"]["row_policy"] = "transform"
    invalid_plan["output_spec"]["transformations"] = [
        {
            "operation": "aggregate",
            "description": "Aggregate the observed values.",
            "authorization": {"source": "user", "quote": "total"},
        }
    ]
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _question_structure_response(target_quote="Return the total observed value."),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "calculation-source",
            ),
            _tool_response("analyze_plan", invalid_plan, "calculation-invalid-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "calculation-todos",
            ),
            _answer_response(
                columns=["value"],
                rows=[["1"]],
                tool_call_id="calculation-answer",
            ),
        ],
    )

    result = DeepAgent(
        model=model,
        config=DeepAgentConfig(question_structure_enabled=True),
    ).run(task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "calculation-invalid-plan")
    assert plan_call["ok"] is True


def test_filter_requirement_must_declare_filter_operation(
    public_task: PublicTask,
) -> None:
    task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question="Return records above 100.",
        ),
        assets=public_task.assets,
    )
    invalid_plan = _plan_args(
        requirement_quote="above 100",
        requirement_type="filter",
        columns=[("value", ["value"])],
    )
    valid_plan = _plan_args(
        requirement_quote="above 100",
        requirement_type="filter",
        columns=[("value", ["value"])],
        transformations=[
            {
                "operation": "filter",
                "description": "Keep rows above the requested threshold.",
                "authorization": {"source": "user", "quote": "above 100"},
            }
        ],
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "filter-source",
            ),
            _tool_response("analyze_plan", invalid_plan, "missing-filter-plan"),
            _tool_response("analyze_plan", valid_plan, "declared-filter-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "filter-todos",
            ),
            _answer_response(
                columns=["value"],
                rows=[["done"]],
                tool_call_id="filter-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    _, invalid_call = _tool_call(result, "missing-filter-plan")
    _, valid_call = _tool_call(result, "declared-filter-plan")
    assert invalid_call["status"] == "error"
    assert "does not declare them" in invalid_call["result"]["content"]
    assert valid_call["ok"] is True


def test_discovery_does_not_keyword_block_tool_arguments(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _question_structure_response(target_quote="Return the observed value."),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "with open('/context/data.txt', encoding='utf-8') as handle:\n"
                        "    rows = handle.readlines()\n"
                        "print(sum(len(row) for row in rows))\n"
                    )
                },
                "unauthorized-discovery",
                content="Try an unstated total during discovery.",
            ),
            _tool_response("analyze_plan", _plan_args(), "authorized-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "authorized-todos",
            ),
            _answer_response(
                columns=["value"],
                rows=[["hello from context"]],
                tool_call_id="authorized-answer",
            ),
        ],
    )

    result = DeepAgent(
        model=model,
        config=DeepAgentConfig(question_structure_enabled=True),
    ).run(public_task)

    assert result.succeeded
    discovery_step, discovery_call = _tool_call(result, "unauthorized-discovery")
    assert discovery_step.ok is True
    assert discovery_call["ok"] is True


def test_parallel_tool_calls_are_correlated_by_id(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="Inspect two allowed data sources.",
                tool_calls=[
                    {
                        "name": "read_doc",
                        "args": {"path": "/context/data.txt"},
                        "id": "parallel-doc",
                        "type": "tool_call",
                    },
                    {
                        "name": "read_json",
                        "args": {"path": "/context/sample.json"},
                        "id": "parallel-json",
                        "type": "tool_call",
                    },
                ],
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    parallel_step, parallel_doc = _tool_call(result, "parallel-doc")
    same_step, parallel_json = _tool_call(result, "parallel-json")
    assert parallel_step is same_step
    assert "hello from context" in json.dumps(parallel_doc["result"])
    assert "sample.json" in json.dumps(parallel_json["result"])
    assert parallel_doc["name"] == "read_doc"
    assert parallel_json["name"] == "read_json"
    assert "message" not in parallel_step.action_input
    assert "llm_input" in parallel_step.raw_response
    assert parallel_step.action_input["tool_calls"] == [
        {
            "name": "read_doc",
            "tool_call_id": "parallel-doc",
            "args": {"path": "/context/data.txt"},
        },
        {
            "name": "read_json",
            "tool_call_id": "parallel-json",
            "args": {"path": "/context/sample.json"},
        },
    ]


def test_analyze_plan_is_unavailable_before_discovery(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        responses=[_answer_response(columns=["value"], rows=[["done"]])]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert "analyze_plan" not in model.bound_tool_sets[0]
    assert model.bound_tool_sets[0] == DISCOVERY_TOOLS
    assert "analyze_plan" in model.bound_tool_sets[1]
    assert model.bound_tool_sets[1] == DISCOVERY_TOOLS | {"analyze_plan"}
    assert model.bound_tool_choices[1] is None
    _, plan_call = _tool_call(result, "plan-call")
    assert plan_call["args"]["evidence"]["knowledge_status"] == "authoritative"
    assert plan_call["args"]["evidence"]["knowledge_rules"] == [
        {
            "rule_type": "semantic",
            "quote": "Use the observed source value exactly.",
            "source_path": "/context/knowledge.md",
        }
    ]
    assert [
        source["path"] for source in plan_call["args"]["evidence"]["context_sources"]
    ] == ["/context/data.txt"]
    assert plan_call["args"]["output_spec"]["row_policy"] == "preserve"
    assert plan_call["args"]["output_spec"]["transformations"] == []


def test_authoritative_knowledge_rejects_inferred_override() -> None:
    arguments = _plan_args()
    arguments["evidence"]["cross_validated_inference"] = (
        "Use a different unit inferred from data.csv."
    )
    result = analyze_plan_tool.func(
        **arguments,
        original_request="Return the observed value.",
        tool_call_id="invalid-authoritative-plan",
    )

    assert not isinstance(result, ToolMessage)
    assert result.update["analysis_plan"]["evidence"]["knowledge_issue"] == ""
    assert (
        result.update["analysis_plan"]["evidence"]["cross_validated_inference"]
        == ""
    )


def test_invalid_knowledge_accepts_single_relevant_context_source() -> None:
    arguments = _plan_args(
        knowledge_status="invalid",
        context_paths=["/context/data.csv"],
    )
    result = analyze_plan_tool.func(
        **arguments,
        original_request="Return the observed value.",
        tool_call_id="single-source-plan",
    )

    assert not isinstance(result, ToolMessage)

    arguments["evidence"]["context_sources"] = []
    result = analyze_plan_tool.func(
        **arguments,
        original_request="Return the observed value.",
        tool_call_id="missing-source-plan",
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "context_sources must include" in str(result.content)


def test_plan_rejects_quote_not_present_in_original_request(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "quote-knowledge",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "quote-source",
            ),
            _tool_response(
                "analyze_plan",
                _plan_args(requirement_quote="Invented user requirement."),
                "invalid-user-quote",
            ),
            _tool_response(
                "analyze_plan",
                _plan_args(),
                "valid-user-quote",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "quote-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, valid_plan = _tool_call(result, "valid-user-quote")
    assert valid_plan["ok"] is True


def test_invalid_transformation_authorization_can_be_corrected(
    public_task: PublicTask,
) -> None:
    invalid_plan = _plan_args(
        transformations=[
            {
                "operation": "aggregate",
                "description": "Aggregate without structured authorization.",
                "authorization": "user",
            }
        ]
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "auth-knowledge",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "auth-source",
            ),
            _tool_response("analyze_plan", invalid_plan, "invalid-auth-plan"),
            _tool_response("analyze_plan", _plan_args(), "valid-auth-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "auth-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, invalid_call = _tool_call(result, "invalid-auth-plan")
    _, valid_call = _tool_call(result, "valid-auth-plan")
    assert invalid_call["ok"] is False
    assert "authorization" in json.dumps(invalid_call["result"])
    assert valid_call["ok"] is True


def test_stringified_plan_arguments_are_decoded(
    public_task: PublicTask,
) -> None:
    plan = _plan_args()
    stringified_plan = {
        **plan,
        "intent": json.dumps(plan["intent"], ensure_ascii=False),
        "output_spec": json.dumps(plan["output_spec"], ensure_ascii=False),
        "evidence": json.dumps(plan["evidence"], ensure_ascii=False),
        "revision": json.dumps(plan["revision"], ensure_ascii=False),
        "steps": json.dumps(plan["steps"], ensure_ascii=False),
    }
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "stringified-knowledge",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "stringified-source",
            ),
            _tool_response("analyze_plan", stringified_plan, "stringified-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "stringified-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "stringified-plan")
    assert plan_call["ok"] is True


def test_markdown_knowledge_quote_is_canonicalized(
    public_task: PublicTask,
) -> None:
    public_task.context_dir.joinpath("knowledge.md").write_text(
        "| `value` | Use the observed source value exactly. |\n",
        encoding="utf-8",
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "markdown-knowledge",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "markdown-source",
            ),
            _tool_response(
                "analyze_plan",
                _plan_args(
                    knowledge_quote=(
                        "value: Use the observed source value exactly."
                    )
                ),
                "markdown-plan",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "markdown-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    plan_message = next(
        message
        for message in result.steps
        if message.action == "analyze_plan"
    )
    plan_content = json.dumps(plan_message.observation, ensure_ascii=False)
    assert "| `value` | Use the observed source value exactly. |" in plan_content


def test_knowledge_authorization_quote_is_canonicalized(
    public_task: PublicTask,
) -> None:
    public_task.context_dir.joinpath("knowledge.md").write_text(
        "| `value` | Use the observed source value exactly. |\n",
        encoding="utf-8",
    )
    plan = _plan_args(
        transformations=[
            {
                "operation": "derive",
                "description": "Apply the knowledge rule.",
                "authorization": {
                    "source": "knowledge",
                    "quote": "value: Use the observed source value exactly.",
                },
            }
        ],
        knowledge_quote="value: Use the observed source value exactly.",
        expected_row_count=1,
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "auth-canonical-knowledge",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "auth-canonical-source",
            ),
            _tool_response(
                "analyze_plan",
                plan,
                "auth-canonical-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "auth-canonical-todos",
            ),
            _answer_response(
                columns=["value"],
                rows=[["done"]],
                tool_call_id="auth-canonical-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "auth-canonical-plan")
    transformation = plan_call["result"]["update"]["analysis_plan"]["output_spec"][
        "transformations"
    ][0]
    assert transformation["authorization"]["quote"] == (
        "| `value` | Use the observed source value exactly. |"
    )


def test_transformation_knowledge_fact_id_canonicalizes_authorization(
    public_task: PublicTask,
) -> None:
    public_task.context_dir.joinpath("knowledge.md").write_text(
        "\n".join(
            [
                "### Example Table (`example_table`)",
                "| Column | Semantic Definition |",
                "|---|---|",
                "| `value` | Use the observed source value exactly. |",
            ]
        ),
        encoding="utf-8",
    )
    plan = _plan_args(
        transformations=[
            {
                "operation": "filter",
                "description": "Apply the knowledge fact.",
                "authorization": {
                    "source": "knowledge",
                    "quote": "WHERE value > 100",
                },
                "authorization_fact_ids": ["kf_1"],
            }
        ],
        knowledge_quote="| `value` | Use the observed source value exactly. |",
        expected_row_count=1,
    )
    plan["evidence"]["knowledge_rules"][0]["fact_id"] = "kf_1"
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "fact-auth-source",
            ),
            _tool_response("analyze_plan", plan, "fact-auth-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "fact-auth-todos",
            ),
            _answer_response(
                columns=["value"],
                rows=[["done"]],
                tool_call_id="fact-auth-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "fact-auth-plan")
    transformation = plan_call["result"]["update"]["analysis_plan"]["output_spec"][
        "transformations"
    ][0]
    assert transformation["authorization"]["quote"] == (
        "| `value` | Use the observed source value exactly. |"
    )
    assert transformation["authorization_fact_ids"] == ["kf_1"]


def test_knowledge_field_rule_type_uses_fact_id_quote(
    public_task: PublicTask,
) -> None:
    public_task.context_dir.joinpath("knowledge.md").write_text(
        "\n".join(
            [
                "### Example Table (`example_table`)",
                "| Column | Semantic Definition |",
                "|---|---|",
                "| `value` | Use the observed source value exactly. |",
            ]
        ),
        encoding="utf-8",
    )
    plan = _plan_args(
        knowledge_quote="value: approximate wording",
        expected_row_count=1,
    )
    plan["evidence"]["knowledge_rules"][0] = {
        "fact_id": "kf_1",
        "quote": "value: approximate wording",
        "rule_type": "field",
        "source_path": "/context/knowledge.md",
    }
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "field-rule-source",
            ),
            _tool_response("analyze_plan", plan, "field-rule-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "field-rule-todos",
            ),
            _answer_response(
                columns=["value"],
                rows=[["done"]],
                tool_call_id="field-rule-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "field-rule-plan")
    rule = plan_call["result"]["update"]["analysis_plan"]["evidence"][
        "knowledge_rules"
    ][0]
    assert rule["rule_type"] == "semantic"
    assert rule["quote"] == "| `value` | Use the observed source value exactly. |"


def test_user_authorization_quote_is_canonicalized_from_expression(
    public_task: PublicTask,
) -> None:
    question = "我能看一下基金的十年回报率数据吗"
    plan = _plan_args(
        requirement_quote="基金的十年回报率数据",
        transformations=[
            {
                "operation": "filter",
                "description": "Filter to the requested ten-year records.",
                "authorization": {"source": "user", "quote": "IndexCycle='十年'"},
            }
        ],
    )
    task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question=question,
        ),
        assets=public_task.assets,
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _question_structure_response(
                target_quote=question,
                target_constraints=[
                    {
                        "quote": "十年",
                        "constraint_type": "time_range",
                        "value": "10 years",
                        "explicitness": "explicit",
                    }
                ],
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "user-auth-expression-source",
            ),
            _tool_response("analyze_plan", plan, "user-auth-expression-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "user-auth-expression-todos",
            ),
            _answer_response(
                columns=["value"],
                rows=[["done"]],
                tool_call_id="user-auth-expression-answer",
            ),
        ],
    )

    result = DeepAgent(
        model=model,
        config=DeepAgentConfig(question_structure_enabled=True),
    ).run(task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "user-auth-expression-plan")
    transformation = plan_call["result"]["update"]["analysis_plan"]["output_spec"][
        "transformations"
    ][0]
    assert transformation["authorization"]["quote"] == "十年"


def test_unknown_knowledge_fact_id_is_dropped_when_quote_is_observed(
    public_task: PublicTask,
) -> None:
    public_task.context_dir.joinpath("knowledge.md").write_text(
        "| `value` | Use the observed source value exactly. |\n",
        encoding="utf-8",
    )
    plan = _plan_args()
    plan["evidence"]["knowledge_rules"][0]["fact_id"] = "missing-fact-id"
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "unknown-fact-knowledge",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "unknown-fact-source",
            ),
            _tool_response("analyze_plan", plan, "unknown-fact-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "unknown-fact-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "unknown-fact-plan")
    saved_rule = plan_call["result"]["update"]["analysis_plan"]["evidence"][
        "knowledge_rules"
    ][0]
    assert plan_call["ok"] is True
    assert "fact_id" not in saved_rule


def test_empty_knowledge_content_does_not_break_quote_canonicalization() -> None:
    assert _canonical_knowledge_quote("missing quote", None) is None


def test_python_knowledge_read_is_available_for_quote_validation() -> None:
    messages: list[BaseMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute_python",
                    "args": {
                        "code": (
                            "print(open('/context/knowledge.md', "
                            "encoding='utf-8').read())"
                        )
                    },
                    "id": "python-knowledge",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="| `value` | Exact knowledge rule. |",
            name="execute_python",
            tool_call_id="python-knowledge",
            status="success",
        ),
    ]

    discovery = _discovery_state(messages)

    assert discovery.knowledge_available
    assert "| `value` | Exact knowledge rule. |" in discovery.knowledge_content


def test_invalid_plan_tool_json_is_retried(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "retry-knowledge",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "retry-source",
            ),
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "analyze_plan",
                        "args": '{"intent":',
                        "id": "malformed-plan",
                        "error": "invalid JSON",
                        "type": "invalid_tool_call",
                    }
                ],
            ),
            _tool_response("analyze_plan", _plan_args(), "retried-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "retry-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.bound_tool_choices[3] == "analyze_plan"
    _, retried_plan = _tool_call(result, "retried-plan")
    assert retried_plan["ok"] is True


def test_observed_knowledge_quote_can_authorize_aggregation(
    public_task: PublicTask,
) -> None:
    plan = _plan_args(
        transformations=[
            {
                "operation": "aggregate",
                "description": "Sum all source rows.",
                "authorization": {
                    "source": "knowledge",
                    "quote": "Use the observed source value exactly.",
                },
            }
        ]
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "semantic-knowledge",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "semantic-source",
            ),
            _tool_response("analyze_plan", plan, "semantic-aggregate"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "semantic-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "semantic-aggregate")
    assert plan_call["ok"] is True


def test_exact_user_quote_can_authorize_aggregation(
    public_task: PublicTask,
) -> None:
    plan = _plan_args(
        requirement_quote="Return the observed value.",
        transformations=[
            {
                "operation": "aggregate",
                "description": "Sum all source rows.",
                "authorization": {
                    "source": "user",
                    "quote": "Return the observed value.",
                },
            }
        ],
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "output-knowledge",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "output-source",
            ),
            _tool_response("analyze_plan", plan, "output-aggregate"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "output-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "output-aggregate")
    assert plan_call["ok"] is True


@pytest.mark.parametrize(
    ("claim", "operation"),
    [
        ("I need to calculate the sum across all provinces.", "aggregate"),
        ("Filter rows where Province is China before returning the result.", "filter"),
        ("Compute a growth rate from the observed values.", "derive"),
        ("Sort rows by EndDate before returning the result.", "sort"),
        ("Return top 5 records after inspecting the file.", "limit"),
        ("Remove duplicates before returning records.", "deduplicate"),
        ("Pivot the records into wide format.", "reshape"),
        (
            "No total national-level row exists explicitly, but we can aggregate "
            "or select representative rows.",
            "aggregate",
        ),
    ],
)
def test_free_text_plan_actions_are_not_keyword_rejected(
    public_task: PublicTask,
    claim: str,
    operation: str,
) -> None:
    invalid_plan = _plan_args()
    invalid_plan["evidence"]["cross_validated_inference"] = claim
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                f"source-{operation}",
            ),
            _tool_response(
                "analyze_plan",
                invalid_plan,
                f"invalid-{operation}-claim",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                f"todos-{operation}",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, f"invalid-{operation}-claim")
    assert plan_call["ok"] is True


def test_free_text_factual_aggregate_description_is_not_rejected(
    public_task: PublicTask,
) -> None:
    plan = _plan_args()
    plan["evidence"]["cross_validated_inference"] = (
        "The preview contains a precomputed aggregate row named National and "
        "province-level source rows."
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "factual-aggregate-source",
            ),
            _tool_response("analyze_plan", plan, "factual-aggregate-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "factual-aggregate-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "factual-aggregate-plan")
    assert plan_call["ok"] is True


def test_canonical_knowledge_quote_handles_table_row_paraphrase() -> None:
    knowledge = (
        "| Column | Semantic Definition | Unit |\n"
        "|---|---|---|\n"
        "| `thirdindustrygdp` | GDP contribution from the tertiary "
        "(services) sector | 亿元 |\n"
    )

    assert _canonical_knowledge_quote(
        "thirdindustrygdp: GDP contribution from the tertiary "
        "(services) sector | Unit: 亿元",
        knowledge,
    ) == "| `thirdindustrygdp` | GDP contribution from the tertiary (services) sector | 亿元 |"


def test_plan_canonicalizes_table_row_knowledge_rule_paraphrase(
    public_task: PublicTask,
) -> None:
    plan = _plan_args(
        knowledge_quote=(
            "thirdindustrygdp: GDP contribution from the tertiary "
            "(services) sector | Unit: 亿元"
        )
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "table-quote-source",
            ),
            _tool_response("analyze_plan", plan, "table-quote-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "table-quote-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )
    public_task.context_dir.joinpath("knowledge.md").write_text(
        "| Column | Semantic Definition | Unit |\n"
        "|---|---|---|\n"
        "| `thirdindustrygdp` | GDP contribution from the tertiary "
        "(services) sector | 亿元 |\n",
        encoding="utf-8",
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "table-quote-plan")
    rules = plan_call["result"]["update"]["analysis_plan"]["evidence"][
        "knowledge_rules"
    ]
    assert rules[0]["quote"] == (
        "| `thirdindustrygdp` | GDP contribution from the tertiary "
        "(services) sector | 亿元 |"
    )


def test_explicit_user_requirement_can_authorize_transformation(
    public_task: PublicTask,
) -> None:
    task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question="Return the maximum observed value.",
        ),
        assets=public_task.assets,
    )
    plan = _plan_args(
        requirement_quote="maximum observed value",
        requirement_type="calculation",
        transformations=[
            {
                "operation": "aggregate",
                "description": "Return the maximum source value.",
                "authorization": {
                    "source": "user",
                    "quote": "maximum observed value",
                },
            }
        ],
        expected_row_count=1,
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "maximum-knowledge",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "maximum-source",
            ),
            _tool_response("analyze_plan", plan, "maximum-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "maximum-todos",
            ),
            _answer_response(columns=["value"], rows=[["maximum"]]),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded


def test_revision_cannot_rewrite_existing_user_requirement(
    public_task: PublicTask,
) -> None:
    revision = _plan_args(
        version=2,
        changed_fields=["intent"],
    )
    revision["intent"]["requirements"][0]["statement"] = (
        "Replace the original requirement with a new interpretation."
    )
    model = ScriptedChatModel(
        responses=[
            _tool_response("analyze_plan", revision, "invalid-revision"),
            _answer_response(columns=["value"], rows=[["original"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["original"]]


def test_invalid_knowledge_allows_single_relevant_source(
    public_task: PublicTask,
) -> None:
    plan = _plan_args(
        knowledge_status="invalid",
        context_paths=["/context/data.txt"],
        steps=["Validate", "Submit"],
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "knowledge-check",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "first-source",
            ),
            _tool_response("analyze_plan", plan, "invalid-knowledge-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Validate", "status": "in_progress"},
                        {"content": "Submit", "status": "pending"},
                    ]
                },
                "invalid-knowledge-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.bound_tool_sets[:4] == [
        DISCOVERY_TOOLS,
        DISCOVERY_TOOLS,
        PLAN_TOOLS,
        {"write_todos"},
    ]
    _, plan_call = _tool_call(result, "invalid-knowledge-plan")
    assert plan_call["ok"] is True


def test_non_authoritative_knowledge_rules_are_cleared(
    public_task: PublicTask,
) -> None:
    plan = _plan_args(
        knowledge_status="invalid",
        context_paths=["/context/data.txt"],
        steps=["Validate", "Submit"],
    )
    plan["evidence"]["knowledge_rules"] = [
        {
            "rule_type": "semantic",
            "quote": "This paraphrased rule is not present in knowledge.",
            "source_path": "/context/knowledge.md",
        }
    ]
    plan["evidence"]["context_sources"].append(
        {
            "path": "/context/knowledge.md",
            "observations": ["Injected knowledge is not an execution data source."],
        }
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "non-auth-knowledge",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "non-auth-source-1",
            ),
            _tool_response("analyze_plan", plan, "non-auth-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Validate", "status": "in_progress"},
                        {"content": "Submit", "status": "pending"},
                    ]
                },
                "non-auth-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "non-auth-plan")
    assert plan_call["ok"] is True
    evidence = plan_call["result"]["update"]["analysis_plan"]["evidence"]
    assert evidence["knowledge_rules"] == []
    assert [item["path"] for item in evidence["context_sources"]] == [
        "/context/data.txt"
    ]


def test_discovery_forces_plan_after_context_is_ready(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "budget-knowledge",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "budget-source-1",
            ),
            _tool_response(
                "analyze_plan",
                _plan_args(
                    context_paths=["/context/data.txt"],
                    steps=["Validate", "Submit"],
                ),
                "budget-plan",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Validate", "status": "in_progress"},
                        {"content": "Submit", "status": "pending"},
                    ]
                },
                "budget-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.bound_tool_sets[2] == PLAN_TOOLS
    assert model.bound_tool_choices[2] is None


def test_invalid_answer_returns_tool_error_and_can_be_corrected(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        responses=[
            _answer_response(columns=[], rows=[], tool_call_id="invalid-answer"),
            _answer_response(columns=["value"], rows=[["correct"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.call_count == 5
    invalid_step, invalid_call = _tool_call(result, "invalid-answer")
    assert invalid_call["name"] == "execute_python"
    assert invalid_call["ok"] is False
    assert invalid_step.ok is False
    assert "non-empty" in json.dumps(invalid_call["result"])
    assert result.steps[-1].ok is True


def test_unexpected_graph_exception_returns_failed_result(
    public_task: PublicTask,
) -> None:
    model = FailingChatModel(responses=[], auto_discovery_plan=False)

    result = DeepAgent(
        model=model,
        config=DeepAgentConfig(question_structure_enabled=False),
    ).run(public_task)

    assert not result.succeeded
    assert result.failure_reason == "Deep Agent failed with RuntimeError: boom"


def test_direct_set_answer_tool_accepts_json_strings_and_stamps_audit(
    public_task: PublicTask,
) -> None:
    question = "Return the highest observed value."
    plan = _plan_args(
        requirement_quote=question,
        transformations=[
            {
                "operation": "sort",
                "description": "Use highest as the selector operation.",
                "authorization": {"source": "user", "quote": "highest"},
            }
        ],
        expected_row_count=1,
    )
    task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question=question,
        ),
        assets=public_task.assets,
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "direct-set-source",
            ),
            _tool_response("analyze_plan", plan, "direct-set-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "direct-set-todos",
            ),
            _tool_response(
                "set_answer",
                {
                    "columns": '["value"]',
                    "rows": '[["done"]]',
                    "audit": json.dumps(
                        {
                            "source_paths": ["/context/data.txt"],
                            "operations": [{"operation": "sort"}],
                            "output_row_count": 99,
                            "output_hash": "stale",
                        },
                        ensure_ascii=False,
                    ),
                },
                "direct-set-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["done"]]
    _, answer_call = _tool_call(result, "direct-set-answer")
    assert answer_call["name"] == "set_answer"
    assert answer_call["ok"] is True


def test_execute_python_auto_captures_result_dict_after_plan(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "auto-capture-source",
            ),
            _tool_response("analyze_plan", _plan_args(), "auto-capture-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "auto-capture-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "columns = ['value']\n"
                        "rows = [['intermediate cursor row']]\n"
                        "result = {'columns': ['value'], 'rows': [['captured']]}\n"
                        "print('ready')\n"
                    )
                },
                "auto-capture-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["captured"]]
    _, answer_call = _tool_call(result, "auto-capture-answer")
    assert answer_call["ok"] is True


def test_answer_column_alias_and_redundant_columns_are_accepted(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        responses=[
            _answer_response(
                columns=["alias_value", "source_note"],
                rows=[["correct", "extra"]],
                tool_call_id="alias-answer",
            ),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.columns == ["alias_value", "source_note"]
    assert result.answer.rows == [["correct", "extra"]]


def test_empty_answer_rows_are_rejected(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _answer_response(columns=["value"], rows=[], tool_call_id="empty-rows"),
            _answer_response(columns=["value"], rows=[["correct"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    empty_step, empty_call = _tool_call(result, "empty-rows")
    assert empty_step.ok is False
    assert empty_call["ok"] is False
    assert "at least one row" in json.dumps(empty_call["result"])


def test_expected_answer_row_count_is_enforced(public_task: PublicTask) -> None:
    plan = _plan_args(expected_row_count=2)
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "row-count-knowledge",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "row-count-source",
            ),
            _tool_response("analyze_plan", plan, "row-count-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "row-count-todos",
            ),
            _answer_response(
                columns=["value"],
                rows=[["one"]],
                tool_call_id="wrong-row-count",
            ),
            _answer_response(
                columns=["value"],
                rows=[["one"], ["two"]],
                tool_call_id="correct-row-count",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    wrong_step, wrong_call = _tool_call(result, "wrong-row-count")
    assert wrong_step.ok is False
    assert wrong_call["ok"] is False
    assert "expected_row_count=2" in json.dumps(wrong_call["result"])


def test_failed_set_answer_saves_answer_candidate_for_recovery(
    public_task: PublicTask,
) -> None:
    plan = _plan_args(expected_row_count=2)
    revised_plan = _plan_args(expected_row_count=1)
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "candidate-source",
            ),
            _tool_response("analyze_plan", plan, "candidate-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "candidate-todos",
            ),
            _answer_response(
                columns=["value"],
                rows=[["candidate"]],
                tool_call_id="candidate-wrong-row-count",
            ),
            _tool_response(
                "analyze_plan",
                revised_plan,
                "candidate-replan",
            ),
            _tool_response(
                "finalize_answer_candidate",
                {"column_indexes": [0], "columns": ["value"]},
                "candidate-finalize",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["candidate"]]
    _, wrong_call = _tool_call(result, "candidate-wrong-row-count")
    assert wrong_call["ok"] is False
    assert wrong_call["result"]["update"]["answer_candidate"]["rows"] == [
        ["candidate"]
    ]
    assert "candidate_saved" in json.dumps(wrong_call["result"])


def test_pre_plan_set_answer_candidate_can_be_finalized_after_plan(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "open('/context/data.txt', encoding='utf-8').read()\n"
                        "set_answer(['value'], [['early candidate']])\n"
                    )
                },
                "early-candidate",
            ),
            _tool_response("analyze_plan", _plan_args(expected_row_count=1), "early-plan"),
            _tool_response(
                "finalize_answer_candidate",
                {"column_indexes": "[0]", "columns": "[\"value\"]"},
                "early-finalize",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["early candidate"]]
    _, early_call = _tool_call(result, "early-candidate")
    assert early_call["ok"] is False
    assert early_call["result"]["update"]["answer_candidate"]["rows"] == [
        ["early candidate"]
    ]


def test_finalize_answer_candidate_accepts_json_encoded_lists(
    public_task: PublicTask,
) -> None:
    plan = _plan_args(expected_row_count=2)
    revised_plan = _plan_args(expected_row_count=1)
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "json-list-candidate-source",
            ),
            _tool_response("analyze_plan", plan, "json-list-candidate-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "json-list-candidate-todos",
            ),
            _answer_response(
                columns=["value", "extra"],
                rows=[["candidate", "ignored"]],
                tool_call_id="json-list-candidate-wrong-row-count",
            ),
            _tool_response(
                "analyze_plan",
                revised_plan,
                "json-list-candidate-replan",
            ),
            _tool_response(
                "finalize_answer_candidate",
                {
                    "column_indexes": "[0]",
                    "columns": "[{\"name\": \"value\", \"role\": \"measure\"}]",
                },
                "json-list-candidate-finalize",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.columns == ["value"]
    assert result.answer.rows == [["candidate"]]


def test_execute_python_set_answer_normalizes_column_specs(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "column-spec-source",
            ),
            _tool_response("analyze_plan", _plan_args(), "column-spec-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "column-spec-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "open('/context/data.txt', encoding='utf-8').read()\n"
                        "columns = [{'name': 'value', 'role': 'measure'}]\n"
                        "rows = [{'value': 'normalized'}]\n"
                        "set_answer(columns, rows)\n"
                    )
                },
                "column-spec-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.columns == ["value"]
    assert result.answer.rows == [["normalized"]]


def test_transform_candidate_projection_updates_execution_audit(
    public_task: PublicTask,
) -> None:
    transformations = [
        {
            "operation": "aggregate",
            "description": "Aggregate from the source.",
            "authorization": {
                "source": "user",
                "quote": "Return the observed value.",
            },
        }
    ]
    plan = _plan_args(transformations=transformations, expected_row_count=2)
    revised_plan = _plan_args(transformations=transformations, expected_row_count=1)
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "projection-source",
            ),
            _tool_response("analyze_plan", plan, "projection-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "projection-todos",
            ),
            _answer_response(
                columns=["value", "extra"],
                rows=[["candidate", "ignored"]],
                tool_call_id="projection-wrong-row-count",
            ),
            _tool_response(
                "analyze_plan",
                revised_plan,
                "projection-replan",
            ),
            _tool_response(
                "finalize_answer_candidate",
                {"column_indexes": [0], "columns": ["value"]},
                "projection-finalize",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["candidate"]]
    _, wrong_call = _tool_call(result, "projection-wrong-row-count")
    assert wrong_call["ok"] is False
    assert wrong_call["result"]["update"]["answer_candidate"]["columns"] == [
        "value",
        "extra",
    ]


def test_transform_answer_requires_execution_audit(
    public_task: PublicTask,
) -> None:
    plan = _plan_args(
        transformations=[
            {
                "operation": "aggregate",
                "description": "Aggregate from the source.",
                "authorization": {
                    "source": "user",
                    "quote": "Return the observed value.",
                },
            }
        ],
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "audit-source",
            ),
            _tool_response("analyze_plan", plan, "audit-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "audit-todos",
            ),
            _tool_response(
                "execute_python",
                {"code": "set_answer(['value'], [['without audit']])\n"},
                "missing-audit-answer",
            ),
            _answer_response(
                columns=["value"],
                rows=[["with audit"]],
                tool_call_id="audited-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    missing_step, missing_call = _tool_call(result, "missing-audit-answer")
    assert missing_step.ok is False
    assert missing_call["ok"] is False
    assert "execution audit" in json.dumps(missing_call["result"])
    assert missing_call["result"]["update"]["answer_candidate"]["rows"] == [
        ["without audit"]
    ]


def test_set_answer_stamps_transform_audit_hash(
    public_task: PublicTask,
) -> None:
    plan = _plan_args(
        transformations=[
            {
                "operation": "aggregate",
                "description": "Aggregate from the source.",
                "authorization": {
                    "source": "user",
                    "quote": "Return the observed value.",
                },
            }
        ],
        expected_row_count=1,
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "stamp-audit-source",
            ),
            _tool_response("analyze_plan", plan, "stamp-audit-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "stamp-audit-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "columns = ['value']\n"
                        "rows = [['with audit']]\n"
                        "set_answer(columns, rows, audit={\n"
                        "    'source_paths': ['/context/data.txt'],\n"
                        "    'operations': ['aggregate'],\n"
                        "    'output_row_count': 999,\n"
                        "    'output_hash': 'wrong',\n"
                        "})\n"
                    )
                },
                "stamp-audit-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["with audit"]]


def test_execute_python_synthesizes_transform_audit_from_context_paths(
    public_task: PublicTask,
) -> None:
    plan = _plan_args(
        transformations=[
            {
                "operation": "filter",
                "description": "Filter to the requested source row.",
                "authorization": {
                    "source": "user",
                    "quote": "Return the observed value.",
                },
            }
        ],
        expected_row_count=1,
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "synth-audit-source",
            ),
            _tool_response("analyze_plan", plan, "synth-audit-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "synth-audit-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "open('/context/data.txt', encoding='utf-8').read()\n"
                        "set_answer(['value'], [['synthesized audit']])\n"
                    )
                },
                "synth-audit-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["synthesized audit"]]


def test_preserve_plan_derives_expected_row_count_from_json(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task_rows"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "knowledge.md").write_text(
        "Use the observed source value exactly.\n",
        encoding="utf-8",
    )
    (context_dir / "sample.json").write_text(
        json.dumps(
            {
                "records": [
                    {"value": 1},
                    {"value": 2},
                    {"value": 3},
                ]
            }
        ),
        encoding="utf-8",
    )
    task = PublicTask(
        record=TaskRecord(
            task_id="task_rows",
            difficulty="easy",
            question="Return the observed value.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "derived-count-knowledge",
            ),
            _tool_response(
                "read_json",
                {"path": "/context/sample.json", "max_items": 2},
                "derived-count-source",
            ),
            _tool_response(
                "analyze_plan",
                _plan_args(context_paths=["/context/sample.json"]),
                "derived-count-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Prepare", "status": "in_progress"}]},
                "derived-count-todos",
            ),
            _answer_response(
                columns=["value"],
                rows=[[6]],
                tool_call_id="wrong-derived-row-count",
            ),
            _answer_response(columns=["value"], rows=[[1], [2], [3]]),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "derived-count-plan")
    output_spec = plan_call["result"]["update"]["analysis_plan"]["output_spec"]
    assert output_spec["expected_row_count"] == 3
    wrong_step, wrong_call = _tool_call(result, "wrong-derived-row-count")
    assert wrong_step.ok is False
    assert wrong_call["ok"] is False
    assert "expected_row_count=3" in json.dumps(wrong_call["result"])


def test_preserve_plan_reuses_observed_row_count_on_revision(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task_row_revision"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "knowledge.md").write_text(
        "Use the observed source value exactly.\n",
        encoding="utf-8",
    )
    (context_dir / "sample.json").write_text(
        json.dumps({"records": [{"value": 1}, {"value": 2}, {"value": 3}]}),
        encoding="utf-8",
    )
    task = PublicTask(
        record=TaskRecord(
            task_id="task_row_revision",
            difficulty="easy",
            question="Return the observed value.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    revision = _plan_args(
        context_paths=["/context/sample.json"],
        expected_row_count=1,
        version=1,
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "revision-count-knowledge",
            ),
            _tool_response(
                "read_json",
                {"path": "/context/sample.json", "max_items": 2},
                "revision-count-source",
            ),
            _tool_response(
                "analyze_plan",
                _plan_args(context_paths=["/context/sample.json"]),
                "revision-count-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Prepare", "status": "in_progress"}]},
                "revision-count-todos",
            ),
            _answer_response(
                columns=["value"],
                rows=[[6]],
                tool_call_id="revision-wrong-row-count",
            ),
            _tool_response(
                "analyze_plan",
                revision,
                "revision-count-replan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Prepare", "status": "in_progress"}]},
                "revision-count-retodos",
            ),
            _answer_response(
                columns=["value"],
                rows=[[1], [2], [3]],
                tool_call_id="revision-correct-row-count",
            ),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    _, replan_call = _tool_call(result, "revision-count-replan")
    assert replan_call["ok"] is True
    output_spec = replan_call["result"]["update"]["analysis_plan"]["output_spec"]
    assert output_spec["expected_row_count"] == 3
    revision_info = replan_call["result"]["update"]["analysis_plan"]["revision"]
    assert revision_info["version"] == 2
    assert revision_info["changed_fields"] == []


def test_observed_sqlite_table_source_drives_preserve_row_count(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task_sqlite_observed"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "knowledge.md").write_text(
        "Use the observed source value exactly.\n",
        encoding="utf-8",
    )
    with closing(sqlite3.connect(context_dir / "sample.sqlite")) as connection:
        connection.execute("CREATE TABLE metrics (name TEXT, value INTEGER)")
        connection.executemany(
            "INSERT INTO metrics (name, value) VALUES (?, ?)",
            [("alpha", 1), ("beta", 2)],
        )
        connection.commit()
    task = PublicTask(
        record=TaskRecord(
            task_id="task_sqlite_observed",
            difficulty="easy",
            question="Return the observed value.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "inspect_sqlite",
                {
                    "path": "/context/sample.sqlite",
                    "table": "metrics",
                    "sample_rows": 1,
                },
                "observed-sqlite-table",
            ),
            _tool_response(
                "analyze_plan",
                _plan_args(context_paths=["/context/sample.sqlite::metrics"]),
                "sqlite-table-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Prepare", "status": "in_progress"}]},
                "sqlite-table-todos",
            ),
            _answer_response(columns=["value"], rows=[[1], [2]]),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    _, source_call = _tool_call(result, "observed-sqlite-table")
    observed = source_call["result"]["update"]["observed_sources"]
    assert "/context/sample.sqlite::metrics" in {
        source["path"] for source in observed
    }
    _, plan_call = _tool_call(result, "sqlite-table-plan")
    output_spec = plan_call["result"]["update"]["analysis_plan"]["output_spec"]
    assert output_spec["expected_row_count"] == 2


def test_question_structure_does_not_rewrite_plan_requirements(
    public_task: PublicTask,
) -> None:
    plan = _plan_args(
        requirement_quote="these records",
        columns=[
            ("EndDate", ["EndDate"]),
            ("Province", ["Province"]),
            ("ThirdIndustryGDP", ["ThirdIndustryGDP"]),
        ],
    )
    plan["intent"]["requirements"] = [
        {
            "statement": "Misclassify geography and measure together.",
            "requirement_type": "entity",
            "quote": "China tertiary GDP",
        },
        {
            "statement": "Misclassify records as the metric.",
            "requirement_type": "measure",
            "quote": "these records",
        },
    ]
    plan["output_spec"]["columns"][0]["role"] = "time_key"
    plan["output_spec"]["columns"][1]["role"] = "entity_key"
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _question_structure_response(
                target_quote="China tertiary GDP these records",
                target_constraints=[
                    {
                        "quote": "China",
                        "constraint_type": "geography",
                        "value": "China",
                        "explicitness": "explicit",
                    },
                    {
                        "quote": "these",
                        "constraint_type": "time_range",
                        "value": "these years",
                        "explicitness": "ambiguous",
                    },
                ],
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "canonical-requirements-source",
            ),
            _tool_response(
                "analyze_plan",
                plan,
                "canonical-requirements-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Prepare", "status": "in_progress"}]},
                "canonical-requirements-todos",
            ),
            _answer_response(
                columns=["EndDate", "Province", "ThirdIndustryGDP"],
                rows=[["2026-01-01", "China", 1]],
            ),
        ],
    )

    task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question="China tertiary GDP these records",
        ),
        assets=public_task.assets,
    )
    result = DeepAgent(
        model=model,
        config=DeepAgentConfig(question_structure_enabled=True),
    ).run(task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "canonical-requirements-plan")
    requirements = plan_call["result"]["update"]["analysis_plan"]["intent"][
        "requirements"
    ]
    assert requirements == plan["intent"]["requirements"]


def test_unrequested_output_columns_are_rejected(public_task: PublicTask) -> None:
    invalid_plan = _plan_args(
        columns=[
            ("value", ["value"]),
            ("observed_at", ["observed_at"]),
        ]
    )
    invalid_plan["output_spec"]["columns"][1]["role"] = "context"
    valid_plan = _plan_args(columns=[("value", ["value"])])
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "extra-column-source",
            ),
            _tool_response("analyze_plan", invalid_plan, "extra-column-plan"),
            _tool_response("analyze_plan", valid_plan, "single-column-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "single-column-todos",
            ),
            _answer_response(columns=["value"], rows=[["hello from context"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.columns == ["value"]
    _, invalid_call = _tool_call(result, "extra-column-plan")
    _, valid_call = _tool_call(result, "single-column-plan")
    assert invalid_call["status"] == "error"
    assert invalid_call["ok"] is False
    assert valid_call["ok"] is True


def test_execution_spec_supporting_selector_field_is_not_final_output(
    public_task: PublicTask,
) -> None:
    question = "Which company has the highest commission?"
    plan = _plan_args(
        requirement_quote=question,
        columns=[("ChiNameAbbr", ["ChiNameAbbr"])],
        transformations=[
            {
                "operation": "sort",
                "description": "Sort by commission to identify the selector row.",
                "authorization": {"source": "user", "quote": "highest"},
            },
            {
                "operation": "limit",
                "description": "Keep the selected highest row.",
                "authorization": {"source": "user", "quote": "highest"},
            },
        ],
        expected_row_count=1,
    )
    plan["output_spec"]["ordering"] = "specified"
    plan["output_spec"]["sort_keys"] = [
        {"field": "Commission", "direction": "descending"}
    ]
    plan["execution_spec"] = {
        "sources": [{"path": "/context/data.txt", "source_type": "doc"}],
        "supporting_fields": [
            {
                "name": "Commission",
                "source_fields": ["Commission"],
                "purpose": "selector",
            }
        ],
        "operations": [
            {
                "operation": "sort",
                "description": "Use highest as the selector operation.",
                "authorization": {"source": "user", "quote": "highest"},
            },
            {
                "operation": "limit",
                "description": "Return only the selected company.",
                "authorization": {"source": "user", "quote": "highest"},
            },
        ],
    }
    task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question=question,
        ),
        assets=public_task.assets,
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _question_structure_response(target_quote=question),
            _tool_response("read_doc", {"path": "/context/data.txt"}, "selector-source"),
            _tool_response("analyze_plan", plan, "selector-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "selector-todos",
            ),
            _answer_response(columns=["ChiNameAbbr"], rows=[["Example Co"]]),
        ],
    )

    result = DeepAgent(
        model=model,
        config=DeepAgentConfig(question_structure_enabled=True),
    ).run(task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "selector-plan")
    assert plan_call["ok"] is True
    assert plan_call["result"]["update"]["analysis_plan"]["output_spec"]["columns"] == [
        {"name": "ChiNameAbbr", "source_fields": ["ChiNameAbbr"]}
    ]


def test_execution_spec_drops_supporting_fields_that_overlap_outputs(
    public_task: PublicTask,
) -> None:
    plan = _plan_args(columns=[("value", ["value"])])
    plan["execution_spec"] = {
        "sources": [{"path": "/context/data.txt"}],
        "supporting_fields": [
            {"name": "value", "source_fields": ["value"], "purpose": "context"},
            {"name": "selector", "source_fields": ["selector"], "purpose": "selector"},
        ],
        "operations": [],
    }
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "overlap-support-source",
            ),
            _tool_response("analyze_plan", plan, "overlap-support-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "overlap-support-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "overlap-support-plan")
    assert plan_call["ok"] is True
    supporting_fields = plan_call["result"]["update"]["analysis_plan"][
        "execution_spec"
    ]["supporting_fields"]
    assert supporting_fields == [
        {"name": "selector", "source_fields": ["selector"], "purpose": "selector"}
    ]


def test_analyze_plan_normalizes_non_object_execution_spec(
    public_task: PublicTask,
) -> None:
    plan = _plan_args()
    plan["execution_spec"] = []
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "list-execution-spec-source",
            ),
            _tool_response("analyze_plan", plan, "list-execution-spec-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "list-execution-spec-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "list-execution-spec-plan")
    assert plan_call["ok"] is True
    assert "execution_spec" not in plan_call["result"]["update"]["analysis_plan"]


def test_execute_sql_is_allowed_during_discovery(public_task: PublicTask) -> None:
    with closing(sqlite3.connect(public_task.context_dir / "sample.sqlite")) as connection:
        connection.execute(
            "INSERT INTO metrics (name, value) VALUES (?, ?)",
            ("alpha", 1),
        )
        connection.commit()
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "execute_sql",
                {
                    "path": "/context/sample.sqlite",
                    "sql": "SELECT COUNT(*) AS row_count FROM metrics",
                },
                "discovery-sql",
            ),
            _tool_response("analyze_plan", _plan_args(context_paths=["/context/sample.sqlite"]), "sql-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "sql-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, sql_call = _tool_call(result, "discovery-sql")
    assert sql_call["ok"] is True


def test_execute_sql_observed_table_source_can_drive_plan(
    public_task: PublicTask,
) -> None:
    with closing(sqlite3.connect(public_task.context_dir / "sample.sqlite")) as connection:
        connection.executemany(
            "INSERT INTO metrics (name, value) VALUES (?, ?)",
            [("alpha", 1), ("beta", 2)],
        )
        connection.commit()
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "execute_sql",
                {
                    "path": "/context/sample.sqlite",
                    "sql": "SELECT value FROM metrics ORDER BY name",
                },
                "sql-observed-table",
            ),
            _tool_response(
                "analyze_plan",
                _plan_args(
                    context_paths=["/context/sample.sqlite::metrics"],
                    expected_row_count=2,
                ),
                "sql-observed-plan",
            ),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "sql-observed-todos",
            ),
            _answer_response(columns=["value"], rows=[[1], [2]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, sql_call = _tool_call(result, "sql-observed-table")
    observed = sql_call["result"]["update"]["observed_sources"]
    table_source = next(
        source for source in observed if source["path"].endswith("::metrics")
    )
    assert sql_call["ok"] is True
    assert table_source["observed_by"] == "execute_sql"
    assert table_source["row_count"] == 2
    assert table_source["fields"] == ["name", "value"]
    _, plan_call = _tool_call(result, "sql-observed-plan")
    assert plan_call["ok"] is True


def test_execute_sql_result_can_prepare_preserve_answer(
    public_task: PublicTask,
) -> None:
    with closing(sqlite3.connect(public_task.context_dir / "sample.sqlite")) as connection:
        connection.executemany(
            "INSERT INTO metrics (name, value) VALUES (?, ?)",
            [("alpha", 1), ("beta", 2)],
        )
        connection.commit()
    plan = _plan_args(
        context_paths=["/context/sample.sqlite::metrics"],
        columns=[("value", ["value"])],
        expected_row_count=2,
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "inspect_sqlite",
                {
                    "path": "/context/sample.sqlite",
                    "table": "metrics",
                    "sample_rows": 1,
                },
                "sql-preserve-source",
            ),
            _tool_response("analyze_plan", plan, "sql-preserve-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "sql-preserve-todos",
            ),
            _tool_response(
                "execute_sql",
                {
                    "path": "/context/sample.sqlite",
                    "sql": "SELECT value FROM metrics ORDER BY name",
                },
                "sql-preserve-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.columns == ["value"]
    assert result.answer.rows == [[1.0], [2.0]]
    _, sql_call = _tool_call(result, "sql-preserve-answer")
    assert sql_call["ok"] is True
    assert "prepared_answer" in sql_call["result"]["update"]


def test_execute_sql_result_can_prepare_transform_answer(
    public_task: PublicTask,
) -> None:
    question = "Return the highest observed value."
    public_task.context_dir.joinpath("knowledge.md").write_text(
        "Use the observed source value exactly.\n",
        encoding="utf-8",
    )
    with closing(sqlite3.connect(public_task.context_dir / "sample.sqlite")) as connection:
        connection.executemany(
            "INSERT INTO metrics (name, value) VALUES (?, ?)",
            [("alpha", 1), ("beta", 2)],
        )
        connection.commit()
    plan = _plan_args(
        requirement_quote=question,
        context_paths=["/context/sample.sqlite::metrics"],
        columns=[("value", ["value"])],
        transformations=[
            {
                "operation": "sort",
                "description": "Use highest as the selector operation.",
                "authorization": {"source": "user", "quote": "highest"},
            },
            {
                "operation": "limit",
                "description": "Keep the highest row.",
                "authorization": {"source": "user", "quote": "highest"},
            },
        ],
        expected_row_count=1,
    )
    task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question=question,
        ),
        assets=public_task.assets,
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "inspect_sqlite",
                {
                    "path": "/context/sample.sqlite",
                    "table": "metrics",
                    "sample_rows": 1,
                },
                "sql-transform-source",
            ),
            _tool_response("analyze_plan", plan, "sql-transform-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "sql-transform-todos",
            ),
            _tool_response(
                "execute_sql",
                {
                    "path": "/context/sample.sqlite",
                    "sql": "SELECT value FROM metrics ORDER BY value DESC LIMIT 1",
                },
                "sql-transform-answer",
            ),
        ],
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [[2.0]]
    _, sql_call = _tool_call(result, "sql-transform-answer")
    assert sql_call["ok"] is True
    messages = sql_call["result"]["update"]["messages"]
    assert messages[0]["name"] == "execute_sql"


def test_transformations_canonicalize_row_policy(public_task: PublicTask) -> None:
    question = "Return the highest observed value."
    plan = _plan_args(
        requirement_quote=question,
        transformations=[
            {
                "operation": "sort",
                "description": "Use highest as the selector operation.",
                "authorization": {"source": "user", "quote": "highest"},
            }
        ],
    )
    plan["output_spec"]["row_policy"] = "preserve"
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "row-policy-source",
            ),
            _tool_response("analyze_plan", plan, "row-policy-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "row-policy-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )
    task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question=question,
        ),
        assets=public_task.assets,
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "row-policy-plan")
    assert plan_call["ok"] is True
    assert (
        plan_call["result"]["update"]["analysis_plan"]["output_spec"]["row_policy"]
        == "transform"
    )


def test_execution_spec_rejects_unauthorized_operations(public_task: PublicTask) -> None:
    invalid_plan = _plan_args()
    invalid_plan["execution_spec"] = {
        "sources": [{"path": "/context/data.txt"}],
        "supporting_fields": [],
        "operations": [
            {
                "operation": "aggregate",
                "description": "Aggregate because the source has many rows.",
            }
        ],
    }
    valid_plan = _plan_args()
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "unauthorized-execution-source",
            ),
            _tool_response(
                "analyze_plan",
                invalid_plan,
                "unauthorized-execution-plan",
            ),
            _tool_response("analyze_plan", valid_plan, "authorized-execution-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "authorized-execution-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, invalid_call = _tool_call(result, "unauthorized-execution-plan")
    _, valid_call = _tool_call(result, "authorized-execution-plan")
    assert invalid_call["status"] == "error"
    assert "requires an exact user quote" in invalid_call["result"]["content"]
    assert valid_call["ok"] is True


def test_execution_operations_inherit_transformation_authorization(
    public_task: PublicTask,
) -> None:
    question = "Return the highest observed value."
    plan = _plan_args(
        requirement_quote=question,
        transformations=[
            {
                "operation": "sort",
                "description": "Use highest as the selector operation.",
                "authorization": {"source": "user", "quote": "highest"},
            }
        ],
    )
    plan["execution_spec"] = {
        "sources": [{"path": "/context/data.txt"}],
        "supporting_fields": [],
        "operations": [
            {
                "operation": "sort",
                "description": "Implement output_spec.transformations.sort.",
            }
        ],
    }
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "inherited-auth-source",
            ),
            _tool_response("analyze_plan", plan, "inherited-auth-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "inherited-auth-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )
    task = PublicTask(
        record=TaskRecord(
            task_id=public_task.task_id,
            difficulty=public_task.difficulty,
            question=question,
        ),
        assets=public_task.assets,
    )

    result = DeepAgent(model=model).run(task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "inherited-auth-plan")
    assert plan_call["ok"] is True


def test_execution_spec_allows_structured_join_operation(
    public_task: PublicTask,
) -> None:
    question = "Return the observed value."
    public_task.context_dir.joinpath("left.txt").write_text("id,value\n1,a\n", encoding="utf-8")
    public_task.context_dir.joinpath("right.txt").write_text("id,value\n1,b\n", encoding="utf-8")
    plan = _plan_args(
        requirement_quote=question,
        context_paths=["/context/left.txt", "/context/right.txt"],
    )
    plan["execution_spec"] = {
        "sources": [
            {"path": "/context/left.txt", "source_type": "doc"},
            {"path": "/context/right.txt", "source_type": "doc"},
        ],
        "supporting_fields": [
            {
                "name": "left_id",
                "source_fields": ["id"],
                "purpose": "join",
            },
            {
                "name": "right_id",
                "source_fields": ["id"],
                "purpose": "join",
            },
        ],
        "operations": [
            {
                "operation": "join",
                "description": "Join the two observed sources on their id fields.",
                "authorization": {"source": "user", "quote": question},
                "left_source": "/context/left.txt",
                "right_source": "/context/right.txt",
                "left_key": "id",
                "right_key": "id",
                "how": "inner",
            }
        ],
    }
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/left.txt"},
                "join-left-source",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/right.txt"},
                "join-right-source",
            ),
            _tool_response("analyze_plan", plan, "join-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "join-todos",
            ),
            _answer_response(columns=["value"], rows=[["joined"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "join-plan")
    assert plan_call["ok"] is True
    operation = plan_call["result"]["update"]["analysis_plan"]["execution_spec"][
        "operations"
    ][0]
    assert operation["operation"] == "join"
    assert operation["left_key"] == "id"


def test_execution_spec_rejects_join_with_unobserved_source(
    public_task: PublicTask,
) -> None:
    question = "Return the observed value."
    public_task.context_dir.joinpath("left.txt").write_text("id,value\n1,a\n", encoding="utf-8")
    invalid_plan = _plan_args(
        requirement_quote=question,
        context_paths=["/context/left.txt"],
    )
    invalid_plan["execution_spec"] = {
        "sources": [
            {"path": "/context/left.txt", "source_type": "doc"},
            {"path": "/context/missing.txt", "source_type": "doc"},
        ],
        "supporting_fields": [],
        "operations": [
            {
                "operation": "join",
                "description": "Join an unobserved source.",
                "authorization": {"source": "user", "quote": question},
                "left_source": "/context/left.txt",
                "right_source": "/context/missing.txt",
                "left_key": "id",
                "right_key": "id",
            }
        ],
    }
    valid_plan = _plan_args(
        requirement_quote=question,
        context_paths=["/context/left.txt"],
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/left.txt"},
                "join-observed-source",
            ),
            _tool_response(
                "analyze_plan",
                invalid_plan,
                "invalid-join-plan",
            ),
            _tool_response("analyze_plan", valid_plan, "valid-after-join-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Submit", "status": "in_progress"}]},
                "valid-after-join-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, invalid_call = _tool_call(result, "invalid-join-plan")
    assert invalid_call["ok"] is False
    assert "unobserved sources" in invalid_call["result"]["content"]


def test_preserve_plan_allows_source_context_columns(
    public_task: PublicTask,
) -> None:
    plan = _plan_args(
        columns=[
            ("EndDate", ["EndDate"]),
            ("Province", ["Province"]),
            ("value", ["value"]),
        ]
    )
    plan["output_spec"]["columns"][0]["role"] = "time_key"
    plan["output_spec"]["columns"][1]["role"] = "entity_key"
    plan["output_spec"]["ordering"] = "unspecified"
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _question_structure_response(
                target_quote="Return the observed value.",
                target_constraints=[
                    {
                        "quote": "Return the observed value.",
                        "constraint_type": "time_range",
                        "value": "observed period",
                        "explicitness": "ambiguous",
                    },
                    {
                        "quote": "Return the observed value.",
                        "constraint_type": "geography",
                        "value": "observed geography",
                        "explicitness": "ambiguous",
                    },
                ],
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "context-columns-source",
            ),
            _tool_response("analyze_plan", plan, "context-columns-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "context-columns-todos",
            ),
            _answer_response(
                columns=["EndDate", "Province", "value"],
                rows=[["2026-01-01", "Beijing", "hello from context"]],
            ),
        ],
    )

    result = DeepAgent(
        model=model,
        config=DeepAgentConfig(question_structure_enabled=True),
    ).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "context-columns-plan")
    assert plan_call["ok"] is True
    assert (
        plan_call["result"]["update"]["analysis_plan"]["output_spec"]["ordering"]
        == "source"
    )


def test_task_2_preserves_source_rows_without_aggregation() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    task_dir = repository_root / "data" / "input" / "task_2"
    context_dir = task_dir / "context"
    task_record = json.loads(
        task_dir.joinpath("task.json").read_text(encoding="utf-8")
    )
    source_records = json.loads(
        context_dir.joinpath("json", "ed_grossdomesticproduct.json").read_text(
            encoding="utf-8"
        )
    )["records"]
    source_rows = [
        [
            record.get("EndDate"),
            record.get("Province"),
            record.get("ThirdIndustryGDP"),
        ]
        for record in source_records
    ]
    steps = [
        "Project EndDate, Province, and ThirdIndustryGDP while preserving source rows",
        "Validate source-shaped rows and submit",
    ]
    question_quote = task_record["question"]
    plan = _plan_args(
        requirement_quote=question_quote,
        knowledge_quote="GDP contribution from the tertiary (services) sector",
        context_paths=["/context/json/ed_grossdomesticproduct.json"],
        columns=[
            ("EndDate", ["EndDate"]),
            ("Province", ["Province"]),
            ("ThirdIndustryGDP", ["ThirdIndustryGDP"]),
        ],
        expected_row_count=354,
        steps=steps,
    )
    plan["output_spec"]["columns"][0]["role"] = "time_key"
    plan["output_spec"]["columns"][1]["role"] = "entity_key"
    plan["intent"]["requirements"][0] = {
        "statement": "Return the historical third-industry GDP records.",
        "requirement_type": "measure",
        "quote": question_quote,
    }
    plan["intent"]["unresolved"] = [
        (
            "No aggregation, sorting, filtering, or null replacement was explicitly "
            "requested; source records include period and region context."
        )
    ]
    plan["output_spec"]["row_grain"] = "one source region-period GDP record"
    task = PublicTask(
        record=TaskRecord(
            task_id=task_record["task_id"],
            difficulty=str(task_record.get("difficulty", "")),
            question=task_record["question"],
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _question_structure_response(
                target_quote=question_quote,
                target_constraints=[
                    {
                        "quote": "我国",
                        "constraint_type": "geography",
                        "value": "China",
                        "explicitness": "explicit",
                    },
                    {
                        "quote": "这些年",
                        "constraint_type": "time_range",
                        "value": "recent years",
                        "explicitness": "ambiguous",
                    },
                ],
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "task-2-knowledge",
            ),
            _tool_response(
                "read_json",
                {"path": "/context/json/ed_grossdomesticproduct.json", "max_items": 40},
                "task-2-source",
            ),
            _tool_response("analyze_plan", plan, "task-2-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": steps[0], "status": "in_progress"},
                        {"content": steps[1], "status": "pending"},
                    ]
                },
                "task-2-todos",
            ),
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "import json\n"
                        "with open("
                        "'/context/json/ed_grossdomesticproduct.json', "
                        "encoding='utf-8') as handle:\n"
                        "    records = json.load(handle)['records']\n"
                        "rows = [[record.get('EndDate'), record.get('Province'), "
                        "record.get('ThirdIndustryGDP')] "
                        "for record in records]\n"
                        "set_answer(['EndDate', 'Province', 'ThirdIndustryGDP'], rows)\n"
                    )
                },
                "task-2-answer",
            ),
        ],
    )

    result = DeepAgent(
        model=model,
        config=DeepAgentConfig(question_structure_enabled=True),
    ).run(task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.columns == ["EndDate", "Province", "ThirdIndustryGDP"]
    assert result.answer.rows == source_rows
    assert len(result.answer.rows) == 354
    assert sum(row[2] is None for row in result.answer.rows) == 41


def test_successful_answer_stops_before_another_model_call(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _answer_response(columns=["value"], rows=[["done"]]),
            AIMessage(content="This response must never be used."),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.call_count == 4


def test_plain_text_completion_is_not_a_valid_answer(public_task: PublicTask) -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="The answer is one.")])

    result = DeepAgent(model=model).run(public_task)

    assert not result.succeeded
    assert result.answer is None
    assert result.failure_reason == "Agent completed without preparing an answer."
    assert result.steps[-1].action == "llm_response"


def test_completed_todos_force_structured_answer(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "completed"},
                        {"content": "Submit the result", "status": "completed"},
                    ]
                },
                "completed-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["done"]]


def test_default_subagent_does_not_receive_answer_tool(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "task",
                {
                    "description": "Inspect the context and return a short report.",
                    "subagent_type": "general-purpose",
                },
                "task-call",
            ),
            AIMessage(content="Subagent report."),
            _answer_response(columns=["value"], rows=[["from report"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert any(
        "execute_python" in tool_names and "task" in tool_names
        for tool_names in model.bound_tool_sets
    )
    assert any(
        "answer" not in tool_names and "task" not in tool_names
        for tool_names in model.bound_tool_sets
    )
    _, task_call = _tool_call(result, "task-call")
    assert "Subagent report." in json.dumps(task_call["result"])
    subagent_steps = [
        step
        for step in result.steps
        if step.action_input.get("scope") == "subagent:general-purpose"
    ]
    assert any(step.action == "system_prompt" for step in subagent_steps)
    assert any("message" in step.action_input for step in subagent_steps)


def test_model_call_limit_returns_failure(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[_tool_response("ls", {"path": "/context"}, f"ls-{index}") for index in range(2)]
    )

    result = DeepAgent(
        model=model,
        config=DeepAgentConfig(max_steps=2),
    ).run(public_task)

    assert not result.succeeded
    assert model.call_count == 2
    assert result.failure_reason == "Agent did not prepare an answer within 2 model calls."


def test_inline_python_execution_is_isolated(
    public_task: PublicTask,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "must-not-reach-shell")
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "import os\n"
                        "from pathlib import Path\n"
                        "Path('context/data.txt').write_text('changed', encoding='utf-8')\n"
                        "print(os.getenv('API_KEY'))\n"
                    ),
                },
                "run-code",
            ),
            _answer_response(columns=["value"], rows=[["isolated"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert public_task.context_dir.joinpath("data.txt").read_text(encoding="utf-8") == (
        "hello from context\n"
    )
    _, environment_call = _tool_call(result, "run-code")
    assert "must-not-reach-shell" not in json.dumps(environment_call["result"])
    assert "None" in json.dumps(environment_call["result"])


def test_python_virtual_context_paths_are_mapped(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "from pathlib import Path\n"
                        "print(Path('/context/data.txt').read_text(encoding='utf-8'))\n"
                        "print([path.name for path in Path('/context').iterdir()])\n"
                    )
                },
                "virtual-paths",
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, virtual_path_call = _tool_call(result, "virtual-paths")
    output = json.dumps(virtual_path_call["result"], ensure_ascii=False)
    assert virtual_path_call["ok"] is True
    assert "hello from context" in output
    assert "sample.csv" in output


def test_python_virtual_scratch_path_is_mapped(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "from pathlib import Path\n"
                        "target = Path('/scratch/result.txt')\n"
                        "target.write_text('完成', encoding='utf-8')\n"
                        "print(target.read_text(encoding='utf-8'))\n"
                    )
                },
                "virtual-scratch",
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, virtual_scratch_call = _tool_call(result, "virtual-scratch")
    output = json.dumps(virtual_scratch_call["result"], ensure_ascii=False)
    assert virtual_scratch_call["ok"] is True
    assert "完成" in output


def test_eve_data_tools_are_callable(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "read_csv",
                {"path": "/context/sample.csv", "max_rows": 1},
                "eve-read-csv",
            ),
            _tool_response(
                "read_json",
                {"path": "/context/sample.json"},
                "eve-read-json",
            ),
            _tool_response(
                "inspect_sqlite",
                {"path": "/context/sample.sqlite", "sample_rows": 1},
                "eve-inspect-sqlite",
            ),
            _tool_response(
                "query_schema",
                {"field": "value"},
                "eve-query-schema",
            ),
            _tool_response(
                "execute_sql",
                {
                    "path": "/context/sample.sqlite",
                    "sql": "SELECT COUNT(*) FROM metrics",
                    "max_rows": 5,
                },
                "eve-execute-sql",
            ),
            _tool_response(
                "grep_file",
                {
                    "path": "/context/data.txt",
                    "pattern": "hello",
                    "output_mode": "content",
                },
                "eve-grep-file",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt", "max_lines": 5},
                "eve-read-doc",
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    for tool_call_id in [
        "eve-read-csv",
        "eve-read-json",
        "eve-inspect-sqlite",
        "eve-query-schema",
        "eve-execute-sql",
        "eve-grep-file",
        "eve-read-doc",
    ]:
        _, call = _tool_call(result, tool_call_id)
        assert call["ok"] is True
    _, csv_call = _tool_call(result, "eve-read-csv")
    assert "sample.csv" in json.dumps(csv_call["result"])
    assert "value" in json.dumps(csv_call["result"])
    _, sql_call = _tool_call(result, "eve-execute-sql")
    assert "COUNT(*)" in json.dumps(sql_call["result"])


def test_read_json_returns_paged_functional_view(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    context_dir = workspace / "context"
    context_dir.mkdir(parents=True)
    context_dir.joinpath("records.json").write_text(
        json.dumps(
            {
                "table": "demo",
                "records": [
                    {"id": 1, "value": 10},
                    {"id": 2, "value": None},
                    {"id": 3, "other": "x"},
                ],
            }
        ),
        encoding="utf-8",
    )
    tool = create_read_json_tool(
        workspace,
        DeepAgentConfig(max_output_bytes=100_000),
    )

    result = tool.invoke(
        {
            "type": "tool_call",
            "name": "read_json",
            "id": "paged-json",
            "args": {"path": "/context/records.json", "max_items": 2, "state": {}},
        }
    )

    command = result
    result = _command_tool_message(result)
    assert result.status == "success"
    payload = json.loads(result.content)
    assert payload["selected_path"] == "records"
    assert payload["total_items"] == 3
    assert payload["returned_items"] == 2
    assert payload["next_start_item"] == 2
    assert payload["previous_start_item"] is None
    assert payload["read_strategy"] == (
        "preview_pages_then_execute_python_for_full_collection"
    )
    assert payload["recommended_next_actions"][0]["tool"] == "read_json"
    assert payload["has_more"] is True
    assert payload["items"] == [{"id": 1, "value": 10}, {"id": 2, "value": None}]
    assert payload["metadata"] == {"table": "demo"}
    assert "data" not in payload
    assert "records" not in payload
    assert payload["schema"]["fields"]["value"]["types"] == ["int", "null"]
    observed = getattr(command, "update", {})["observed_sources"]
    assert observed[0]["path"] == "/context/records.json"
    assert observed[0]["row_count"] == 3
    assert observed[0]["fields"] == ["id", "value", "other"]


def test_read_doc_defaults_to_paged_window(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    context_dir = workspace / "context" / "doc"
    context_dir.mkdir(parents=True)
    context_dir.joinpath("large.md").write_text(
        "\n".join(f"line {index}" for index in range(1, 201)),
        encoding="utf-8",
    )
    tool = create_read_doc_tool(
        workspace,
        DeepAgentConfig(max_output_bytes=100_000),
    )

    result = tool.invoke(
        {
            "type": "tool_call",
            "name": "read_doc",
            "id": "paged-doc",
            "args": {"path": "/context/doc/large.md", "state": {}},
        }
    )

    command = result
    result = _command_tool_message(result)
    assert result.status == "success"
    payload = json.loads(result.content)
    assert payload["returned_lines"] == 120
    assert payload["total_lines"] == 200
    assert payload["start_line_arg"] == 0
    assert payload["end_line"] == 120
    assert payload["next_start_line"] == 120
    assert payload["previous_start_line"] is None
    assert payload["has_more"] is True
    assert payload["truncated"] is True
    assert payload["read_strategy"] == "grep_file_to_line_anchors_then_read_doc_slices"
    assert payload["recommended_next_actions"][0]["tool"] == "grep_file"
    assert payload["recommended_next_actions"][1]["args"]["start_line"] == 120
    assert "   120->line 120" in payload["content"]
    assert "   121->line 121" not in payload["content"]
    observed = getattr(command, "update", {})["observed_sources"]
    assert observed[0]["path"] == "/context/doc/large.md"
    assert observed[0]["line_count"] == 200


def test_read_doc_surfaces_semantic_windows_from_question(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    context_dir = workspace / "context"
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True)
    context_dir.joinpath("knowledge.md").write_text(
        "\n".join(
            [
                "# Table `mf_netvalueperformancehis`",
                "| Column | Semantic Definition | Unit |",
                "|---|---|---|",
                "| `RRInTenYear` | 近十年累计回报率 | % |",
            ]
        ),
        encoding="utf-8",
    )
    doc_dir.joinpath("mf_netvalueperformancehis.md").write_text(
        "\n".join(
            [
                "背景说明。",
                "档案 266 的十年回报率为 166.097944%。",
            ]
        ),
        encoding="utf-8",
    )
    tool = create_read_doc_tool(
        workspace,
        DeepAgentConfig(max_output_bytes=100_000),
    )

    command = tool.invoke(
        {
            "type": "tool_call",
            "name": "read_doc",
            "id": "semantic-window-doc",
            "args": {
                "path": "/context/doc/mf_netvalueperformancehis.md",
                "max_lines": 1,
                "state": {
                    "original_request": "我能看一下基金的十年回报率数据吗",
                    "question_structure": {
                        "target_constraints": [
                            {"quote": "十年", "value": "ten_years"}
                        ]
                    },
                },
            },
        }
    )

    message = _command_tool_message(command)
    payload = json.loads(message.content)
    assert "semantic_windows" in payload
    assert "十年回报率" in payload["semantic_windows"][0]["content"]
    observed = getattr(command, "update", {})["observed_sources"]
    assert observed[0]["logical_name"] == "mf_netvalueperformancehis"
    assert "十年回报率" in observed[0]["matched_lines"][0]["content"]


def test_inspect_sqlite_overview_omits_sample_rows(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    db_dir = workspace / "context" / "db"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "sample.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE metrics (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, text TEXT)")
        connection.executemany(
            "INSERT INTO metrics (value) VALUES (?)",
            [("alpha",), ("beta",)],
        )
    tool = create_inspect_sqlite_tool(
        workspace,
        DeepAgentConfig(max_output_bytes=100_000),
    )

    overview = tool.invoke(
        {
            "type": "tool_call",
            "name": "inspect_sqlite",
            "id": "sqlite-overview",
            "args": {"path": "/context/db/sample.sqlite", "state": {}},
        }
    )
    overview_command = overview
    overview = _command_tool_message(overview)
    detail = tool.invoke(
        {
            "type": "tool_call",
            "name": "inspect_sqlite",
            "id": "sqlite-detail",
            "args": {
                "path": "/context/db/sample.sqlite",
                "table": "metrics",
                "sample_rows": 1,
                "state": getattr(overview_command, "update", {}),
            },
        }
    )
    detail_command = detail
    detail = _command_tool_message(detail)

    assert overview.status == "success"
    overview_payload = json.loads(overview.content)
    assert overview_payload["mode"] == "database_overview"
    assert overview_payload["table_count"] == 2
    assert set(overview_payload["tables"]) == {"metrics", "notes"}
    assert "sample_rows" not in overview_payload["tables"]["metrics"]

    assert detail.status == "success"
    detail_payload = json.loads(detail.content)
    assert detail_payload["mode"] == "table_detail"
    assert detail_payload["tables"]["metrics"]["sample_rows"] == [
        {"id": 1, "value": "alpha"}
    ]
    observed = getattr(detail_command, "update", {})["observed_sources"]
    assert {source["path"] for source in observed} == {
        "/context/db/sample.sqlite",
        "/context/db/sample.sqlite::metrics",
    }
    table_source = next(
        source for source in observed if source["path"].endswith("::metrics")
    )
    assert table_source["row_count"] == 2
    assert table_source["fields"] == ["id", "value"]


def test_python_tool_schema_keeps_answer_rows_out_of_model_input(
    tmp_path: Path,
) -> None:
    (tmp_path / "scratch").mkdir()
    tool = create_execute_python_tool(tmp_path, DeepAgentConfig())

    schema = tool.tool_call_schema.model_json_schema()

    assert schema["required"] == ["code"]
    assert set(schema["properties"]) == {"code"}


def test_read_csv_returns_paging_hints(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    context_dir = workspace / "context"
    context_dir.mkdir(parents=True)
    context_dir.joinpath("records.csv").write_text(
        "id,value\n1,a\n2,b\n3,c\n",
        encoding="utf-8",
    )
    tool = create_read_csv_tool(
        workspace,
        DeepAgentConfig(max_output_bytes=100_000),
    )

    command = tool.invoke(
        {
            "type": "tool_call",
            "name": "read_csv",
            "id": "paged-csv",
            "args": {"path": "/context/records.csv", "max_rows": 2, "state": {}},
        }
    )

    message = _command_tool_message(command)
    payload = json.loads(message.content)
    assert message.status == "success"
    assert payload["returned_rows"] == 2
    assert payload["total_rows"] == 3
    assert payload["next_start_row"] == 2
    assert payload["previous_start_row"] is None
    assert payload["read_strategy"] == "preview_pages_then_execute_python_for_full_table"
    assert payload["recommended_next_actions"][0]["tool"] == "read_csv"


def test_set_answer_accepts_object_rows(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "rows = [{'value': 'one'}, {'value': 'two'}]\n"
                        "set_answer(['value'], rows)\n"
                    )
                },
                "object-rows",
            )
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["one"], ["two"]]


def test_empty_python_source_is_rejected(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {"code": " \n"},
                "invalid-code",
            ),
            _answer_response(columns=["value"], rows=[["recovered"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    invalid_step, invalid_call = _tool_call(result, "invalid-code")
    assert invalid_step.ok is False
    assert invalid_call["ok"] is False
    assert "non-empty" in json.dumps(invalid_call["result"])


def test_python_nonzero_exit_is_a_tool_error(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {"code": "raise RuntimeError('expected failure')\n"},
                "run-failure",
            ),
            _answer_response(columns=["value"], rows=[["recovered"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    failed_step, failed_call = _tool_call(result, "run-failure")
    assert failed_step.ok is False
    assert failed_call["ok"] is False
    assert "Exit code: 1" in json.dumps(failed_call["result"])
    assert "expected failure" in json.dumps(failed_call["result"])


def test_unavailable_tools_are_hidden_from_main_and_subagent(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "task",
                {
                    "description": "Inspect the context.",
                    "subagent_type": "general-purpose",
                },
                "task-call",
            ),
            AIMessage(content="Subagent report."),
            _answer_response(columns=["value"], rows=[["done"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    assert model.bound_tool_sets
    assert model.bound_tool_sets[0] == DISCOVERY_TOOLS
    assert model.bound_tool_choices[0] is None
    assert model.bound_tool_sets[1] == PLAN_TOOLS
    assert model.bound_tool_choices[1] is None
    assert model.bound_tool_sets[2] == {"write_todos"}
    assert model.bound_tool_choices[2] == "write_todos"
    hidden_tools = {
        "edit_file",
        "execute",
        "glob",
        "grep",
        "list_context",
        "list_videos",
        "ls",
        "read_file",
        "write_file",
    }
    for tool_names in model.bound_tool_sets[3:]:
        assert hidden_tools.isdisjoint(tool_names)
    assert all("execute_python" in tool_names for tool_names in model.bound_tool_sets[3:])


def test_write_todos_canonicalizes_plan_step_contents(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "todo-source",
            ),
            _tool_response("analyze_plan", _plan_args(), "todo-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute the source value", "status": "in_progress"},
                        {"content": "Return the table", "status": "pending"},
                    ]
                },
                "todo-canonicalized",
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, todo_call = _tool_call(result, "todo-canonicalized")
    assert todo_call["ok"] is True
    assert "must exactly match" not in json.dumps(todo_call["result"])


def test_write_todos_canonicalizes_mismatched_plan_step_count(
    public_task: PublicTask,
) -> None:
    plan_steps = ["Read source", "Inspect records", "Project fields", "Submit answer"]
    expected_steps = [
        "Read relevant source data from /context/data.txt",
        "Project value from source rows preserving source order and nulls",
        "Submit final answer with analysis_plan.output_spec columns",
    ]
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "todo-count-source",
            ),
            _tool_response(
                "analyze_plan",
                _plan_args(steps=plan_steps),
                "todo-count-plan",
            ),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Read", "status": "in_progress"},
                        {"content": "Project", "status": "pending"},
                        {"content": "Submit", "status": "pending"},
                    ]
                },
                "todo-count-canonicalized",
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, todo_call = _tool_call(result, "todo-count-canonicalized")
    assert todo_call["ok"] is True
    todos = todo_call["result"]["update"]["todos"]
    assert [todo["content"] for todo in todos] == expected_steps
    assert [todo["status"] for todo in todos] == [
        "in_progress",
        "pending",
        "pending",
    ]


def test_plan_steps_are_derived_from_structured_spec(
    public_task: PublicTask,
) -> None:
    plan = _plan_args(
        steps=[
            "Filter rows where Province is China or national total",
            "Submit answer",
        ]
    )
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "derived-steps-source",
            ),
            _tool_response("analyze_plan", plan, "derived-steps-plan"),
            _tool_response(
                "write_todos",
                {"todos": [{"content": "Filter rows", "status": "in_progress"}]},
                "derived-steps-todos",
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, plan_call = _tool_call(result, "derived-steps-plan")
    steps = plan_call["result"]["update"]["analysis_plan"]["steps"]
    assert steps == [
        "Read relevant source data from /context/data.txt",
        "Project value from source rows preserving source order and nulls",
        "Submit final answer with analysis_plan.output_spec columns",
    ]


def test_hidden_tools_are_rejected_if_called(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "read_file",
                {
                    "file_path": "/large_tool_results/read-json-call",
                    "offset": 0,
                    "limit": 100,
                },
                "disabled-read-file",
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    read_step, read_call = _tool_call(result, "disabled-read-file")
    assert read_step.action == "read_file"
    assert read_step.ok is False
    assert read_call["ok"] is False
    assert "disabled" in json.dumps(read_call["result"]).lower()


def test_custom_system_prompt_replaces_deepagents_defaults(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        responses=[
            _answer_response(columns=["value"], rows=[["ok"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    write_todos_description = load_tool_prompt("write_todos")
    task_description_template = load_tool_prompt("task")
    combined_system_text = "\n\n".join(model.system_texts)
    assert "你是一个基准数据任务代理" in combined_system_text
    assert "## 工具使用" in combined_system_text
    assert "{tool_descriptions}" not in combined_system_text
    assert "`read_json`" in combined_system_text
    assert "`execute_sql`" in combined_system_text
    assert "`analyze_plan`" in combined_system_text
    assert "`write_todos`" in combined_system_text
    assert " ".join(write_todos_description.split()) in combined_system_text
    assert "`task`" in combined_system_text
    assert "You are a deep agent" not in combined_system_text
    assert "Filesystem Tools" not in combined_system_text
    assert "Large Tool Results" not in combined_system_text
    assert "Available subagent types" not in combined_system_text
    assert "You have access to the `write_todos` tool" not in combined_system_text
    assert "`read_file`" not in combined_system_text
    tool_descriptions = {
        name: description
        for batch in model.bound_tool_descriptions
        for name, description in batch.items()
    }
    assert tool_descriptions["write_todos"] == write_todos_description
    assert tool_descriptions["task"].startswith(
        task_description_template.split("{available_agents}", maxsplit=1)[0]
    )
    assert "general-purpose" in tool_descriptions["task"]
    assert "{available_agents}" not in tool_descriptions["task"]


def test_python_output_is_utf8(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {"code": "print('北京 上海 全国')\n"},
                "unicode-output",
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, output_call = _tool_call(result, "unicode-output")
    output = json.dumps(output_call["result"], ensure_ascii=False)
    assert "北京 上海 全国" in output
    assert "�" not in output


def test_execute_python_reads_pdf_text_with_helper(public_task: PublicTask) -> None:
    fitz = pytest.importorskip("fitz")
    pdf_path = public_task.context_dir / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Portable context text")
    document.save(pdf_path)
    document.close()
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "text = read_context_text('/context/sample.pdf')\n"
                        "print(text[:80])\n"
                    )
                },
                "pdf-helper-output",
            ),
            _answer_response(columns=["value"], rows=[["ok"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, output_call = _tool_call(result, "pdf-helper-output")
    output = json.dumps(output_call["result"], ensure_ascii=False)
    assert "Portable context text" in output


def test_binary_context_read_is_returned_as_text_error(
    public_task: PublicTask,
) -> None:
    model = ScriptedChatModel(
        auto_discovery_plan=False,
        responses=[
            _tool_response(
                "read_doc",
                {"path": "/context/knowledge.md"},
                "binary-knowledge",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/data.txt"},
                "binary-source",
            ),
            _tool_response(
                "read_doc",
                {"path": "/context/sample.pdf"},
                "binary-read",
            ),
            _tool_response("analyze_plan", _plan_args(), "binary-plan"),
            _tool_response(
                "write_todos",
                {
                    "todos": [
                        {"content": "Compute and validate", "status": "in_progress"},
                        {"content": "Submit the result", "status": "pending"},
                    ]
                },
                "binary-todos",
            ),
            _answer_response(columns=["value"], rows=[["done"]]),
        ],
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    _, read_call = _tool_call(result, "binary-read")
    assert read_call["ok"] is False
    assert "Failed to read document" in json.dumps(read_call["result"])


def test_utf8_filesystem_backend_grep_handles_chinese(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    context_dir.joinpath("data.txt").write_text(
        "北京 上海 全国\n",
        encoding="utf-8",
    )
    backend = Utf8FilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    result = backend.grep("全国", "/context")

    assert result.error is None
    assert result.matches == [
        {"path": "/context/data.txt", "line": 1, "text": "北京 上海 全国"}
    ]


def test_builtin_write_is_denied_for_context(public_task: PublicTask) -> None:
    model = ScriptedChatModel(
        responses=[
            _tool_response(
                "write_file",
                {"file_path": "/context/new.txt", "content": "blocked"},
                "blocked-write",
            ),
            _answer_response(columns=["value"], rows=[["unchanged"]]),
        ]
    )

    result = DeepAgent(model=model).run(public_task)

    assert result.succeeded
    blocked_step, blocked_call = _tool_call(result, "blocked-write")
    assert blocked_step.ok is False
    assert blocked_call["ok"] is False
    assert "disabled" in json.dumps(blocked_call["result"]).lower()
    assert not public_task.context_dir.joinpath("new.txt").exists()

