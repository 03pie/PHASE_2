from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools._helpers import (
    error,
    navigate_json_path,
    resolve_context_path,
    success,
    virtual_path,
)
from data_agent_baseline.tools.observed_sources import (
    observed_sources_command,
    sample_hash,
)

_DEFAULT_COLLECTION_KEYS = ("records", "items", "rows", "data", "values")
_SCHEMA_SAMPLE_LIMIT = 200
_EXAMPLE_VALUE_LIMIT = 3
_EXAMPLE_STRING_LIMIT = 80
_MAX_JSON_ITEMS_PER_READ = 200
_MESSAGE_BUDGET_RATIO = 0.85
_COMPACT_ITEM_FIELD_LIMIT = 24
_COMPACT_STRING_LIMIT = 160


def _read_strategy_payload(
    *,
    path: str,
    json_path: str,
    total_items: int,
    returned_items: int,
    start_item: int,
    effective_max_items: int,
    next_start_item: int | None,
    previous_start_item: int | None,
) -> dict[str, Any]:
    base_args: dict[str, Any] = {"path": path, "max_items": effective_max_items}
    if json_path:
        base_args["json_path"] = json_path
    actions: list[dict[str, Any]] = []
    if next_start_item is not None:
        actions.append(
            {
                "tool": "read_json",
                "reason": "read the next preview page",
                "args": {**base_args, "start_item": next_start_item},
            }
        )
    if previous_start_item is not None:
        actions.append(
            {
                "tool": "read_json",
                "reason": "read the previous preview page",
                "args": {**base_args, "start_item": previous_start_item},
            }
        )
    actions.append(
        {
            "tool": "execute_python",
            "reason": "run full-collection computation after schema and relevant fields are confirmed",
            "args": {"path": path, **({"json_path": json_path} if json_path else {})},
        }
    )
    return {
        "large_file": total_items > effective_max_items,
        "read_strategy": "preview_pages_then_execute_python_for_full_collection",
        "recommended_next_actions": actions,
        "page_window": {
            "start_item": start_item,
            "returned_items": returned_items,
            "max_items": effective_max_items,
        },
    }


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, Mapping):
        return "dict"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def _example_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _EXAMPLE_STRING_LIMIT:
        return {
            "type": "str",
            "length": len(value),
            "preview": value[:_EXAMPLE_STRING_LIMIT],
        }
    return value


def _json_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        keys = [str(key) for key in value.keys()]
        return {"type": "dict", "key_count": len(keys), "keys": keys[:50]}
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    return {"type": _type_name(value), "value": value}


def _serialized_json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def _compact_json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        if len(value) <= _COMPACT_STRING_LIMIT:
            return value
        return {
            "type": "str",
            "length": len(value),
            "preview": value[:_COMPACT_STRING_LIMIT],
        }
    if isinstance(value, Mapping):
        items = list(value.items())
        compacted = {
            str(key): _compact_json_value(item)
            for key, item in items[:_COMPACT_ITEM_FIELD_LIMIT]
        }
        if len(items) > _COMPACT_ITEM_FIELD_LIMIT:
            compacted["__omitted_fields__"] = len(items) - _COMPACT_ITEM_FIELD_LIMIT
        return compacted
    if isinstance(value, list):
        sample = [_compact_json_value(item) for item in value[:5]]
        payload: dict[str, Any] = {
            "type": "list",
            "length": len(value),
            "sample": sample,
        }
        if len(value) > len(sample):
            payload["omitted_items"] = len(value) - len(sample)
        return payload
    return str(value)


def _shrink_schema_examples(payload: dict[str, Any], *, keep: int) -> None:
    schema = payload.get("schema")
    if not isinstance(schema, dict):
        return
    fields = schema.get("fields")
    if not isinstance(fields, dict):
        return
    for field_payload in fields.values():
        if not isinstance(field_payload, dict):
            continue
        examples = field_payload.get("examples")
        if isinstance(examples, list):
            field_payload["examples"] = examples[:keep]


def _fit_collection_payload_to_budget(
    payload: dict[str, Any],
    *,
    max_output_bytes: int,
) -> dict[str, Any]:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return payload
    budget = max(1_000, int(max_output_bytes * _MESSAGE_BUDGET_RATIO))
    if _serialized_json_size(payload) <= budget:
        return payload

    updated = dict(payload)
    original_returned_items = len(items)
    compacted_items = [_compact_json_value(item) for item in items]
    updated["items"] = compacted_items
    updated["item_payload_truncated"] = True

    if _serialized_json_size(updated) > budget:
        _shrink_schema_examples(updated, keep=1)
    if _serialized_json_size(updated) > budget:
        _shrink_schema_examples(updated, keep=0)

    while (
        len(updated["items"]) > 1
        and _serialized_json_size(updated) > budget
    ):
        keep = max(1, len(updated["items"]) // 2)
        updated["items"] = updated["items"][:keep]

    if _serialized_json_size(updated) > budget:
        updated["items"] = []
        updated["item_payload_omitted"] = original_returned_items

    returned_items = len(updated["items"])
    if returned_items != original_returned_items:
        start_item = int(updated.get("start_item") or 0)
        total_items = int(updated.get("total_items") or 0)
        next_start_item = (
            start_item + returned_items
            if start_item + returned_items < total_items
            else None
        )
        updated["returned_items"] = returned_items
        updated["next_start_item"] = next_start_item
        updated["has_more"] = next_start_item is not None
        updated["truncated"] = True
        page_window = updated.get("page_window")
        if isinstance(page_window, dict):
            updated["page_window"] = {
                **page_window,
                "returned_items": returned_items,
            }
    return updated


def _parent_metadata(value: Any, collection_key: str | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    metadata: dict[str, Any] = {}
    for key, item in value.items():
        rendered_key = str(key)
        if rendered_key == collection_key:
            continue
        metadata[rendered_key] = (
            item
            if item is None or isinstance(item, str | bool | int | float)
            else _json_summary(item)
        )
    return metadata


def _find_default_collection(value: Any) -> tuple[str | None, list[Any] | None]:
    if isinstance(value, list):
        return "", value
    if not isinstance(value, Mapping):
        return None, None

    for key in _DEFAULT_COLLECTION_KEYS:
        item = value.get(key)
        if isinstance(item, list):
            return key, item

    list_items = [
        (str(key), item)
        for key, item in value.items()
        if isinstance(item, list)
    ]
    if len(list_items) == 1:
        return list_items[0]
    return None, None


def _field_schema(items: Sequence[Any]) -> dict[str, Any]:
    sample = list(items[:_SCHEMA_SAMPLE_LIMIT])
    if not sample:
        return {"sample_size": 0, "fields": {}}

    if all(isinstance(item, Mapping) for item in sample):
        fields: dict[str, dict[str, Any]] = {}
        for item in sample:
            assert isinstance(item, Mapping)
            for key, value in item.items():
                rendered_key = str(key)
                field = fields.setdefault(
                    rendered_key,
                    {"types": set(), "null_count": 0, "examples": []},
                )
                field["types"].add(_type_name(value))
                if value is None:
                    field["null_count"] += 1
                elif (
                    len(field["examples"]) < _EXAMPLE_VALUE_LIMIT
                    and _example_value(value) not in field["examples"]
                    and not isinstance(value, Mapping | list)
                ):
                    field["examples"].append(_example_value(value))
        return {
            "sample_size": len(sample),
            "fields": {
                key: {
                    "types": sorted(value["types"]),
                    "null_count": value["null_count"],
                    "examples": value["examples"],
                }
                for key, value in fields.items()
            },
        }

    types = sorted({_type_name(item) for item in sample})
    examples = [
        item
        for item in sample
        if item is not None and not isinstance(item, Mapping | list)
    ][:_EXAMPLE_VALUE_LIMIT]
    return {"sample_size": len(sample), "item_types": types, "examples": examples}


def _window_items(items: list[Any], start_item: int, max_items: int) -> tuple[list[Any], bool]:
    end = start_item + max_items
    return items[start_item:end], end < len(items)


def _join_json_path(prefix: str, child: str | None) -> str:
    if not child:
        return prefix
    if not prefix:
        return child
    return f"{prefix}.{child}"


def create_read_json_tool(workspace: Path, config: DeepAgentConfig) -> BaseTool:
    """Create a JSON reader with dotted-path navigation and list paging."""

    context_root = (workspace / "context").resolve()

    @tool("read_json", description=load_tool_prompt("read_json"))
    def read_json(
        path: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict[str, Any], InjectedState],
        json_path: str = "",
        start_item: int = 0,
        max_items: int = 50,
    ) -> Any:
        """Run the read_json tool."""

        resolved, path_error = resolve_context_path(
            context_root,
            path,
            allowed_suffixes={".json"},
        )
        if path_error:
            return error(
                name="read_json",
                tool_call_id=tool_call_id,
                message=path_error,
                max_output_bytes=config.max_output_bytes,
            )
        if start_item < 0 or max_items < 1:
            return error(
                name="read_json",
                tool_call_id=tool_call_id,
                message="start_item >= 0 and max_items >= 1 required.",
                max_output_bytes=config.max_output_bytes,
            )

        try:
            assert resolved is not None
            effective_max_items = min(max_items, _MAX_JSON_ITEMS_PER_READ)
            data = json.loads(resolved.read_text(encoding="utf-8"))
            selected = navigate_json_path(data, json_path) if json_path else data
            collection_path, collection_items = _find_default_collection(selected)

            payload: dict[str, Any] = {
                "path": virtual_path(resolved, context_root),
                "json_path": json_path,
                "root": _json_summary(data),
                "selected": _json_summary(selected),
            }

            source_fields: list[str] = []
            row_count: int | None = None
            selected_path = ""
            sample_value: Any = selected
            if collection_items is not None:
                page_items, has_more = _window_items(
                    collection_items,
                    start_item,
                    effective_max_items,
                )
                selected_path = _join_json_path(json_path, collection_path)
                schema = _field_schema(collection_items)
                fields = schema.get("fields")
                if isinstance(fields, Mapping):
                    source_fields = [str(field) for field in fields.keys()]
                row_count = len(collection_items)
                sample_value = page_items
                next_start_item = (
                    start_item + len(page_items)
                    if start_item + len(page_items) < len(collection_items)
                    else None
                )
                previous_start_item = (
                    max(0, start_item - effective_max_items)
                    if start_item > 0
                    else None
                )
                payload.update(
                    {
                        "selected_path": selected_path,
                        "items": page_items,
                        "total_items": len(collection_items),
                        "returned_items": len(page_items),
                        "start_item": start_item,
                        "requested_max_items": max_items,
                        "max_items": effective_max_items,
                        "next_start_item": next_start_item,
                        "previous_start_item": previous_start_item,
                        "has_more": has_more,
                        "truncated": start_item > 0 or has_more,
                        "schema": schema,
                        "metadata": _parent_metadata(selected, collection_path),
                        **_read_strategy_payload(
                            path=payload["path"],
                            json_path=json_path,
                            total_items=len(collection_items),
                            returned_items=len(page_items),
                            start_item=start_item,
                            effective_max_items=effective_max_items,
                            next_start_item=next_start_item,
                            previous_start_item=previous_start_item,
                        ),
                    }
                )
            else:
                if isinstance(selected, Mapping):
                    source_fields = [str(field) for field in selected.keys()]
                payload.update(
                    {
                        "data": selected,
                        "truncated": False,
                        "hint": (
                            "No list-like collection was selected. Use json_path to "
                            "navigate to an array field when paging is needed."
                        ),
                    }
                )
            payload = _fit_collection_payload_to_budget(
                payload,
                max_output_bytes=config.max_output_bytes,
            )
            if isinstance(payload.get("items"), list):
                sample_value = payload["items"]
            message = success(
                name="read_json",
                tool_call_id=tool_call_id,
                payload=payload,
                max_output_bytes=config.max_output_bytes,
            )
            return observed_sources_command(
                state=state,
                message=message,
                sources=[
                    {
                        "path": payload["path"],
                        "source_type": "json",
                        "row_count": row_count,
                        "fields": source_fields,
                        "selected_path": selected_path,
                        "sample_hash": sample_hash(sample_value),
                        "observed_by": "read_json",
                    }
                ],
            )
        except Exception as exc:
            return error(
                name="read_json",
                tool_call_id=tool_call_id,
                message=f"Failed to read JSON: {exc}",
                max_output_bytes=config.max_output_bytes,
            )

    return read_json
