from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from pathlib import Path

import wcmatch.glob as wcglob
from deepagents.backends.filesystem import (
    FilesystemBackend,
    _resolve_ripgrep_path,
)
from deepagents.backends.protocol import DEFAULT_GREP_TIMEOUT

logger = logging.getLogger(__name__)


class Utf8FilesystemBackend(FilesystemBackend):
    """Filesystem backend with UTF-8-safe grep on Windows."""

    def _ripgrep_search(
        self,
        pattern: str,
        base_full: Path,
        include_glob: str | None,
    ) -> dict[str, list[tuple[int, str]]] | None:
        rg_path = _resolve_ripgrep_path()
        if rg_path is None:
            return None

        cmd = [rg_path, "--json", "-F"]
        if include_glob:
            cmd.extend(["--glob", include_glob])

        rg_cwd: str | None = None
        if base_full.is_dir():
            cmd.extend(["--", pattern, "."])
            rg_cwd = str(base_full)
        else:
            cmd.extend(["--", pattern, str(base_full)])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=DEFAULT_GREP_TIMEOUT,
                check=False,
                cwd=rg_cwd,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "ripgrep timed out after %ds; using Python grep fallback",
                DEFAULT_GREP_TIMEOUT,
            )
            return None
        except (FileNotFoundError, PermissionError, NotADirectoryError) as exc:
            logger.warning(
                "ripgrep subprocess failed (%s: %s); using Python grep fallback",
                type(exc).__name__,
                exc,
            )
            _resolve_ripgrep_path.cache_clear()
            return None

        stderr = proc.stderr.decode("utf-8", errors="replace")
        if proc.returncode not in (0, 1):
            logger.warning(
                "ripgrep exited %d (stderr=%r); using Python grep fallback",
                proc.returncode,
                stderr.strip()[:500],
            )
            return None

        results: dict[str, list[tuple[int, str]]] = {}
        base_resolved = base_full.resolve()
        stdout = proc.stdout.decode("utf-8", errors="replace")
        for line in stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "error":
                logger.debug("ripgrep per-file error frame: %s", data.get("data"))
                continue
            if data.get("type") != "match":
                continue

            payload = data.get("data", {})
            matched_path = payload.get("path", {}).get("text")
            line_number = payload.get("line_number")
            if not matched_path or line_number is None:
                continue

            raw_path = Path(matched_path)
            path = raw_path if raw_path.is_absolute() else (base_full / raw_path)
            try:
                path.resolve().relative_to(base_resolved)
            except (ValueError, OSError):
                logger.warning(
                    "Skipping ripgrep result outside search root: path=%s root=%s",
                    path,
                    base_full,
                )
                continue

            if self.virtual_mode:
                try:
                    rendered_path = self._to_virtual_path(path)
                except ValueError:
                    logger.debug("Skipping grep result outside root: %s", path)
                    continue
                except (OSError, RuntimeError):
                    logger.warning(
                        "Could not resolve grep result path: %s",
                        path,
                        exc_info=True,
                    )
                    continue
            else:
                rendered_path = str(path)

            text = payload.get("lines", {}).get("text", "").rstrip("\r\n")
            results.setdefault(rendered_path, []).append((int(line_number), text))

        return results

    def _python_search(
        self,
        pattern: str,
        base_full: Path,
        include_glob: str | None,
        *,
        timeout: int = DEFAULT_GREP_TIMEOUT,
    ) -> tuple[dict[str, list[tuple[int, str]]], str | None]:
        deadline = time.monotonic() + timeout
        regex = re.compile(pattern)
        results: dict[str, list[tuple[int, str]]] = {}
        root = base_full if base_full.is_dir() else base_full.parent
        candidates = root.rglob("*") if base_full.is_dir() else [base_full]

        try:
            for path in candidates:
                if time.monotonic() > deadline:
                    message = (
                        f"Grep of '{base_full}' timed out after {timeout}s "
                        f"with {len(results)} matching file(s); try a more "
                        "specific pattern or a narrower path."
                    )
                    logger.warning("%s", message)
                    return results, message
                try:
                    if not path.is_file():
                        continue
                    if include_glob:
                        relative_path = str(path.relative_to(root))
                        if not wcglob.globmatch(
                            relative_path,
                            include_glob,
                            flags=wcglob.BRACE | wcglob.GLOBSTAR,
                        ):
                            continue
                    if path.stat().st_size > self.max_file_size_bytes:
                        continue
                    content = path.read_text(encoding="utf-8", errors="replace")
                except (PermissionError, OSError, RuntimeError, UnicodeDecodeError):
                    continue

                for line_number, line in enumerate(content.splitlines(), 1):
                    if not regex.search(line):
                        continue
                    if self.virtual_mode:
                        try:
                            rendered_path = self._to_virtual_path(path)
                        except ValueError:
                            logger.debug("Skipping grep result outside root: %s", path)
                            continue
                        except (OSError, RuntimeError):
                            logger.warning(
                                "Could not resolve grep result path: %s",
                                path,
                                exc_info=True,
                            )
                            continue
                    else:
                        rendered_path = str(path)
                    results.setdefault(rendered_path, []).append((line_number, line))
        except (OSError, RuntimeError) as exc:
            message = (
                f"Grep of '{base_full}' aborted after "
                f"{len(results)} matching file(s): {exc}"
            )
            logger.warning("%s", message, exc_info=True)
            return results, message

        return results, None
