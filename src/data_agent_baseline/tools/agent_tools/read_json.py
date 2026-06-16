from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from data_agent_baseline.agents.deep_state import DeepAgentConfig
from data_agent_baseline.prompts.loader import load_tool_prompt
from data_agent_baseline.tools._helpers import (
    error,
    navigate_json_path,
    resolve_context_path,
    success,
    virtual_path,
)

_DEFAULT_COLLECTION_KEYS = ("records", "items", "rows", "data", "values")
_SCHEMA_SAMPLE_LIMIT = 200
_EXAMPLE_VALUE_LIMIT = 3


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


def _json_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        keys = [str(key) for key in value.keys()]
        return {"type": "dict", "key_count": len(keys), "keys": keys[:50]}
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    return {"type": _type_name(value), "value": value}


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
                    and value not in field["examples"]
                    and not isinstance(value, Mapping | list)
                ):
                    field["examples"].append(value)
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
            data = json.loads(resolved.read_text(encoding="utf-8"))
            selected = navigate_json_path(data, json_path) if json_path else data
            collection_path, collection_items = _find_default_collection(selected)

            payload: dict[str, Any] = {
                "path": virtual_path(resolved, context_root),
                "json_path": json_path,
                "root": _json_summary(data),
                "selected": _json_summary(selected),
            }

            if collection_items is not None:
                page_items, has_more = _window_items(
                    collection_items,
                    start_item,
                    max_items,
                )
                selected_path = _join_json_path(json_path, collection_path)
                payload.update(
                    {
                        "selected_path": selected_path,
                        "items": page_items,
                        "total_items": len(collection_items),
                        "returned_items": len(page_items),
                        "start_item": start_item,
                        "has_more": has_more,
                        "truncated": start_item > 0 or has_more,
                        "schema": _field_schema(collection_items),
                        "metadata": _parent_metadata(selected, collection_path),
                    }
                )
            else:
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
            return success(
                name="read_json",
                tool_call_id=tool_call_id,
                payload=payload,
                max_output_bytes=config.max_output_bytes,
            )
        except Exception as exc:
            return error(
                name="read_json",
                tool_call_id=tool_call_id,
                message=f"Failed to read JSON: {exc}",
                max_output_bytes=config.max_output_bytes,
            )

    return read_json
