from __future__ import annotations

from pathlib import Path
from typing import Any

from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.evidence_agent.codex_loop.context import build_context_fragments
from data_agent_baseline.evidence_agent.codex_loop.controller import CodexEvidenceController
from data_agent_baseline.evidence_agent.codex_loop.protocol import (
    KnowledgeSemanticCard,
    KnowledgeSourceMapping,
    LoopState,
    ModelAction,
    SourceRef,
)
from data_agent_baseline.evidence_agent.codex_loop.registry import EvidenceActionRegistry
from data_agent_baseline.evidence_agent.codex_loop.state_views import (
    primary_next_action,
    selected_source_candidates,
    semantic_selection_view,
)


def _dispatch(state: LoopState, tool_name: str, arguments: dict):
    return EvidenceActionRegistry().dispatch(
        state,
        ModelAction(kind="tool_call", tool_name=tool_name, arguments=arguments),
    )


def test_declare_answer_contract_is_saved_and_visible(tmp_path: Path) -> None:
    state = LoopState(question="show target value", context_dir=tmp_path)

    evidence = _dispatch(
        state,
        "declare_answer_contract",
        {
            "intent_summary": "show the target value",
            "answer_grain": "one row per source record",
            "requested_outputs": ["target_value"],
            "constraints": [
                {
                    "semantic_field": "target_value",
                    "operator": "requested",
                    "reason": "the question asks for this value",
                }
            ],
            "operations": {"row_shape": "preserve_rows", "reason": "no row reduction requested"},
            "row_shape": "preserve_rows",
            "null_policy": "preserve",
            "transform_intent": "show the requested value without changing row grain",
            "unresolved_terms": [],
        },
    )

    assert evidence.ok
    assert state.answer_contract is not None
    assert state.answer_contract.intent_summary == "show the target value"
    assert state.answer_contract.answer_grain == "one row per source record"
    assert state.answer_contract.requested_outputs == ("target_value",)
    assert state.answer_contract.constraints[0]["semantic_field"] == "target_value"

    fragments = build_context_fragments(state)
    kinds = {fragment.kind for fragment in fragments}
    assert "answer_contract" in kinds
    assert "semantic_selection" in kinds
    guidance = next(fragment for fragment in fragments if fragment.kind == "next_action_guidance")
    assert "target_value" in guidance.text


def test_primary_next_action_requires_semantic_selection_after_contract(tmp_path: Path) -> None:
    state = LoopState(question="show target value", context_dir=tmp_path)
    _dispatch(
        state,
        "declare_answer_contract",
        {
            "intent_summary": "show the target value",
            "answer_grain": "one row per source record",
            "final_outputs": ["target_value"],
            "row_shape": "preserve_rows",
            "null_policy": "preserve",
            "transform_intent": "show target value",
        },
    )
    state.semantic_cards = [
        KnowledgeSemanticCard(
            id="sem_0001",
            kind="field_definition",
            semantic_scope="logical_table",
            semantic_slot="target_value",
            name="logical_table.target_value",
            definition="target value",
        )
    ]

    action = primary_next_action(state)

    assert action["tool_name"] == "select_semantic_cards"


def test_controller_requires_native_tool_calls() -> None:
    class BoundModel:
        def __init__(self, tool_choice: Any) -> None:
            self.tool_choice = tool_choice

        def invoke(self, messages: list[Any]) -> Any:
            del messages

            class Response:
                content = ""
                tool_calls = []
                invalid_tool_calls = []
                response_metadata = {"finish_reason": "tool_calls"}
                additional_kwargs: dict[str, Any] = {}

            return Response()

    class Model:
        def __init__(self) -> None:
            self.bound_tool_choices: list[Any] = []

        def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> BoundModel:
            del tools
            self.bound_tool_choices.append(kwargs.get("tool_choice"))
            return BoundModel(kwargs.get("tool_choice"))

    model = Model()
    controller = CodexEvidenceController(model=model, config=DeepAgentConfig())

    assert model.bound_tool_choices[0] == "required"

    _response, tool_calls, raw = controller._call_model([])

    assert tool_calls == []
    assert raw["tool_choice"] == "required"


def test_select_semantic_cards_saves_selection_and_filters_source_candidates(tmp_path: Path) -> None:
    state = LoopState(question="show target value", context_dir=tmp_path)
    source_a = _source(tmp_path, "src_0001", "a.csv", "csv_records")
    source_b = _source(tmp_path, "src_0002", "b.csv", "csv_records")
    state.sources = {source_a.id: source_a, source_b.id: source_b}
    state.semantic_cards = [
        KnowledgeSemanticCard(
            id="sem_0001",
            kind="field_definition",
            semantic_scope="table_a",
            semantic_slot="target_value",
            name="table_a.target_value",
            definition="target value",
        ),
        KnowledgeSemanticCard(
            id="sem_0002",
            kind="field_definition",
            semantic_scope="table_b",
            semantic_slot="other_value",
            name="table_b.other_value",
            definition="other value",
        ),
    ]
    state.source_mappings = [
        KnowledgeSourceMapping(
            card_id="sem_0001",
            source_id=source_a.id,
            source_path=source_a.virtual_path,
            data_form=source_a.data_form,
            status="unverified_structured_candidate",
            semantic_scope="table_a",
            semantic_slot="target_value",
            physical_field="TargetValue",
        ),
        KnowledgeSourceMapping(
            card_id="sem_0002",
            source_id=source_b.id,
            source_path=source_b.virtual_path,
            data_form=source_b.data_form,
            status="unverified_structured_candidate",
            semantic_scope="table_b",
            semantic_slot="other_value",
            physical_field="OtherValue",
        ),
    ]

    evidence = _dispatch(
        state,
        "select_semantic_cards",
        {"card_ids": ["sem_0001"], "rationale": "target value maps to sem_0001"},
    )

    assert evidence.ok
    assert state.semantic_selection is not None
    assert state.semantic_selection.card_ids == ("sem_0001",)
    assert [item.card_id for item in state.selected_source_mappings] == ["sem_0001"]

    view = selected_source_candidates(state)
    assert [item["card_id"] for item in view["candidates"]] == ["sem_0001"]
    assert semantic_selection_view(state)["present"]


def test_no_source_candidate_fallback_without_semantic_selection(tmp_path: Path) -> None:
    state = LoopState(question="show target value", context_dir=tmp_path)
    source = _source(tmp_path, "src_0001", "a.csv", "csv_records")
    state.sources[source.id] = source
    state.semantic_cards = [
        KnowledgeSemanticCard(
            id="sem_0001",
            kind="field_definition",
            semantic_scope="logical_table",
            semantic_slot="target_value",
            name="logical_table.target_value",
            definition="target value",
        )
    ]
    state.source_mappings = [
        KnowledgeSourceMapping(
            card_id="sem_0001",
            source_id=source.id,
            source_path=source.virtual_path,
            data_form=source.data_form,
            status="unverified_structured_candidate",
            semantic_scope="logical_table",
            semantic_slot="target_value",
            physical_field="TargetValue",
        )
    ]

    view = selected_source_candidates(state)

    assert view["candidates"] == []


def test_compute_final_can_project_explicit_row_indices_without_semantic_gate(tmp_path: Path) -> None:
    state = LoopState(question="return selected row", context_dir=tmp_path)
    observed = state.add_evidence(tool_name="sample_records", ok=True, summary="sample")
    binding = state.add_binding(
        binding_type="structured_source",
        evidence_refs=(observed.id,),
        allowed_columns=("name", "score"),
    )
    compute = state.add_compute_result(
        sql="select name, score from rel_0001 order by score desc",
        columns=("name", "score"),
        rows=(("alpha", 10), ("beta", 7)),
        binding_refs=(binding.id,),
        evidence_refs=(observed.id,),
    )

    result = EvidenceActionRegistry().dispatch(
        state,
        ModelAction(
            kind="final",
            compute_ref=compute.id,
            answer={"columns": ["name"], "row_indices": [0]},
        ),
    )

    assert result.ok
    assert state.final_answer is not None
    assert state.final_answer["columns"] == ["name"]
    assert state.final_answer["rows"] == [["alpha"]]
    assert state.final_answer["row_indices"] == [0]


def test_compute_final_rejects_invalid_projection_mechanically(tmp_path: Path) -> None:
    state = LoopState(question="return selected row", context_dir=tmp_path)
    observed = state.add_evidence(tool_name="sample_records", ok=True, summary="sample")
    binding = state.add_binding(
        binding_type="structured_source",
        evidence_refs=(observed.id,),
        allowed_columns=("name",),
    )
    compute = state.add_compute_result(
        sql="select name from rel_0001",
        columns=("name",),
        rows=(("alpha",),),
        binding_refs=(binding.id,),
        evidence_refs=(observed.id,),
    )

    result = EvidenceActionRegistry().dispatch(
        state,
        ModelAction(
            kind="final",
            compute_ref=compute.id,
            answer={"columns": ["missing"], "row_indices": [2]},
        ),
    )

    assert not result.ok
    assert result.negative_scope is not None
    assert result.negative_scope["kind"] in {
        "invalid_final_row_indices",
        "missing_final_projection",
    }


def test_bind_records_semantic_mappings(tmp_path: Path) -> None:
    state = LoopState(question="bind observed mapping", context_dir=tmp_path)
    source = _source(tmp_path, "src_0001", "records.csv", "csv_records")
    state.sources[source.id] = source
    evidence = state.add_evidence(
        tool_name="inspect_source",
        ok=True,
        summary="observed",
        payload={"columns": ["ObservedValue"]},
        source_id=source.id,
        data_form=source.data_form,
    )

    result = CodexEvidenceController._apply_bind(
        object(),
        state,
        ModelAction(
            kind="bind",
            binding_type="structured_source",
            source_ref=source.id,
            evidence_refs=(evidence.id,),
            arguments={
                "allowed_columns": ["ObservedValue"],
                "semantic_mappings": [
                    {
                        "card_id": "sem_0001",
                        "semantic_field": "logical_table.target_value",
                        "source_id": source.id,
                        "physical_field_or_record_field": "ObservedValue",
                        "evidence_refs": [evidence.id],
                        "alignment": "LLM judged the observed field as target value.",
                    }
                ],
                "alignment": "bind observed semantic mapping",
            },
        ),
    )

    assert result.ok
    binding = next(iter(state.bindings.values()))
    assert binding.metadata["semantic_mappings"][0]["semantic_field"] == "logical_table.target_value"


def _source(tmp_path: Path, source_id: str, filename: str, data_form: str) -> SourceRef:
    path = tmp_path / filename
    path.write_text("", encoding="utf-8")
    return SourceRef(
        id=source_id,
        path=path,
        virtual_path=f"/context/{filename}",
        basename=path.name,
        stem=path.stem,
        suffix=path.suffix,
        data_form=data_form,  # type: ignore[arg-type]
        size_bytes=path.stat().st_size,
    )
