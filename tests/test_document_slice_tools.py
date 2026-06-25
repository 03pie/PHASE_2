from __future__ import annotations

from pathlib import Path
from typing import Any

from data_agent_baseline.evidence_agent.codex_loop.protocol import LoopState, ModelAction, SourceRef
from data_agent_baseline.evidence_agent.codex_loop.registry import EvidenceActionRegistry


ROOT = Path(__file__).resolve().parents[1]


def _state_with_doc(task_id: str, relative_path: str, data_form: str = "markdown_document") -> LoopState:
    path = ROOT / "data" / "input" / task_id / "context" / relative_path
    normalized_relative_path = relative_path.replace("\\", "/")
    virtual_path = f"/context/{normalized_relative_path}"
    source = SourceRef(
        id="src_0001",
        path=path,
        virtual_path=virtual_path,
        basename=path.name,
        stem=path.stem,
        suffix=path.suffix,
        data_form=data_form,  # type: ignore[arg-type]
        size_bytes=path.stat().st_size,
    )
    state = LoopState(question="", context_dir=path.parent.parent)
    state.sources[source.id] = source
    state.source_by_path[source.id] = source.id
    state.source_by_path[source.virtual_path] = source.id
    state.source_by_path[source.path.as_posix()] = source.id
    return state


def _dispatch(state: LoopState, tool_name: str, arguments: dict[str, Any]):
    return EvidenceActionRegistry().dispatch(
        state,
        ModelAction(kind="tool_call", tool_name=tool_name, arguments=arguments),
    )


def test_document_tool_surface_is_converged() -> None:
    registry = EvidenceActionRegistry()

    assert "preview_document" in registry.tool_names
    assert "search_document" in registry.tool_names
    assert "read_document_slice" in registry.tool_names
    assert "profile_document" not in registry.tool_names
    assert "read_document_window" not in registry.tool_names


def test_preview_document_returns_start_end_and_slice_catalog() -> None:
    state = _state_with_doc("task_40", "doc/ed_otherdepositorycorpbs.md")

    evidence = _dispatch(
        state,
        "preview_document",
        {"source_ref": "src_0001", "preview_lines": 8, "slice_lines": 50},
    )

    assert evidence.ok
    assert evidence.payload["line_count"] > 1000
    assert evidence.payload["slice_count"] > 10
    assert evidence.payload["slice_lines"] == 50
    assert evidence.payload["slice_lines_source"] == "explicit_argument"
    assert evidence.payload["start_preview"]["text"]
    assert evidence.payload["end_preview"]["text"]
    assert evidence.payload["slice_catalog"][0]["slice_id"] == "src_0001_slice_0001"
    assert evidence.payload["slice_catalog"][0]["end_line"] == 50
    assert evidence.payload["recommended_first_slice"]["slice_lines"] == 50
    assert any("Total Assets" in heading["text"] for heading in evidence.payload["headings"])


def test_search_document_only_locates_and_recommends_slices() -> None:
    state = _state_with_doc("task_40", "doc/ed_otherdepositorycorpbs.md")

    evidence = _dispatch(
        state,
        "search_document",
        {"source_ref": "src_0001", "query": "Total Assets", "slice_lines": 50},
    )

    assert evidence.ok
    assert evidence.payload["slice_lines"] == 50
    assert evidence.payload["slice_lines_source"] == "explicit_argument"
    assert evidence.payload["total_matches"] > 0
    assert evidence.payload["matches"]
    assert evidence.payload["slice_matches"]
    assert "windows" not in evidence.payload
    assert "text" not in evidence.payload
    first_read = evidence.payload["matches"][0]["recommended_read"]
    assert first_read["source_ref"] == "src_0001"
    assert first_read["slice_id"].startswith("src_0001_slice_")
    assert first_read["slice_lines"] == 50


def test_read_document_slice_returns_complete_navigation_slice() -> None:
    state = _state_with_doc("task_40", "doc/ed_otherdepositorycorpbs.md")

    evidence = _dispatch(
        state,
        "read_document_slice",
        {"source_ref": "src_0001", "slice_index": 2, "slice_lines": 80},
    )

    assert evidence.ok
    assert evidence.payload["slice_id"] == "src_0001_slice_0002"
    assert evidence.payload["start_line"] == 81
    assert evidence.payload["end_line"] == 160
    assert evidence.payload["slice_lines_source"] == "explicit_argument"
    assert "Total Assets" in evidence.payload["text"]
    assert evidence.payload["previous_slice"]["slice_id"] == "src_0001_slice_0001"
    assert evidence.payload["next_slice"]["slice_id"] == "src_0001_slice_0003"
    assert evidence.payload["previous_slice"]["slice_lines"] == 80
    assert evidence.payload["next_slice"]["slice_lines"] == 80
    assert evidence.payload["expand_slice"]["slice_lines"] == 80
    assert evidence.payload["coverage"]["remaining_after"] is True


def test_document_slice_size_follows_model_intent_for_same_source() -> None:
    state = _state_with_doc("task_40", "doc/ed_otherdepositorycorpbs.md")

    preview = _dispatch(
        state,
        "preview_document",
        {"source_ref": "src_0001", "slice_lines": 50},
    )
    search = _dispatch(
        state,
        "search_document",
        {"source_ref": "src_0001", "query": "Total Assets"},
    )
    read = _dispatch(
        state,
        "read_document_slice",
        {"source_ref": "src_0001", "slice_index": 2},
    )

    assert preview.payload["slice_lines"] == 50
    assert search.payload["slice_lines"] == 50
    assert search.payload["slice_lines_source"] == "source_preference"
    assert search.payload["matches"][0]["recommended_read"]["slice_lines"] == 50
    assert read.payload["slice_lines"] == 50
    assert read.payload["slice_lines_source"] == "source_preference"
    assert read.payload["start_line"] == 51
    assert read.payload["end_line"] == 100


def test_extract_records_requires_read_document_slice_not_search_locator() -> None:
    state = _state_with_doc("task_40", "doc/ed_otherdepositorycorpbs.md")
    search = _dispatch(
        state,
        "search_document",
        {"source_ref": "src_0001", "query": "Total Assets", "slice_lines": 80},
    )

    extraction = _dispatch(
        state,
        "extract_records",
        {
            "evidence_refs": [search.id],
            "spec": {
                "regex": r"Total Assets of (?P<value>\d+(?:\.\d+)?) million",
                "dotall": True,
            },
        },
    )

    assert not extraction.ok
    assert extraction.negative_scope["kind"] == "missing_document_slice_evidence"
    assert extraction.recommended_next_actions[0]["tool_name"] == "read_document_slice"


def test_extract_records_succeeds_from_read_document_slice() -> None:
    state = _state_with_doc("task_40", "doc/ed_otherdepositorycorpbs.md")
    slice_evidence = _dispatch(
        state,
        "read_document_slice",
        {"source_ref": "src_0001", "slice_index": 2, "slice_lines": 80},
    )

    extraction = _dispatch(
        state,
        "extract_records",
        {
            "evidence_refs": [slice_evidence.id],
            "spec": {
                "regex": r"Total Assets (?:of|were recorded at) (?P<value>\d+(?:\.\d+)?) million",
                "dotall": True,
            },
        },
    )

    assert extraction.ok
    assert extraction.payload["records"]
    assert all("value" in record for record in extraction.payload["records"])
