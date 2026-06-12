from __future__ import annotations

from data_agent_baseline.benchmark.schema import PublicTask


def list_context_tree(
    task: PublicTask,
    *,
    max_depth: int = 4,
) -> dict[str, object]:
    """生成 CLI 展示所需的任务文件树。"""

    entries: list[dict[str, object]] = []

    def walk(depth: int, parent_path: str = "") -> None:
        if depth > max_depth:
            return
        directory = task.context_dir / parent_path
        for child in sorted(
            directory.iterdir(),
            key=lambda item: (item.is_file(), item.name),
        ):
            relative_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": relative_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if child.is_dir():
                walk(depth + 1, relative_path)

    walk(1)
    return {
        "root": str(task.context_dir),
        "entries": entries,
    }
