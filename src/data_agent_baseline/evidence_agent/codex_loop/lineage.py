from __future__ import annotations

DIRECT_FINAL_BINDING_TYPES = frozenset(
    {"document_window", "value", "operation", "answer_candidate"}
)
META_EVIDENCE_TOOLS = frozenset(
    {"declare_answer_contract", "select_semantic_cards", "bind", "submit_final"}
)
