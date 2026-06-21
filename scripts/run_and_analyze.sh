#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  exec "$ROOT_DIR/.venv/bin/python" scripts/run_benchmark_with_analysis.py "$@"
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run python scripts/run_benchmark_with_analysis.py "$@"
fi

echo "Neither .venv/bin/python nor uv is available. Please run 'uv sync' first." >&2
exit 127
