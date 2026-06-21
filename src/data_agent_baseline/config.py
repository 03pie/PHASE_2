from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _strip_inline_comment(value: str) -> str:
    in_single_quote = False
    in_double_quote = False
    for index, character in enumerate(value):
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif character == "#" and not in_single_quote and not in_double_quote:
            return value[:index].rstrip()
    return value


def _parse_env_value(raw_value: str) -> str:
    value = _strip_inline_comment(raw_value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_project_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, _parse_env_value(raw_value))


_load_project_dotenv()

API_KEY = os.environ.get("API_KEY")
API_BASE = os.environ.get("API_BASE")
MODEL = os.environ.get("MODEL")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
INPUT_DIR = os.environ.get("INPUT_DIR")


def _default_dataset_root() -> Path:
    # 加载环境变量中的输入目录，如果没有则报错
    return PROJECT_ROOT / INPUT_DIR if INPUT_DIR else PROJECT_ROOT / "data"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / OUTPUT_DIR if OUTPUT_DIR else PROJECT_ROOT / "artifacts/runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str | None = MODEL
    api_base: str | None = API_BASE
    api_key: str | None = API_KEY
    max_retries: int = 3
    max_steps: int = 16
    temperature: float = 0.0
    execute_timeout_seconds: int = 30
    max_output_bytes: int = 100_000
    model_call_interval_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_env_config(payload: dict[str, Any]) -> None:
    env_overrides = {
        "dataset": {
            "root_path": INPUT_DIR,
        },
        "agent": {
            "model": MODEL,
            "api_base": API_BASE,
            "api_key": API_KEY,
        },
        "run": {
            "output_dir": OUTPUT_DIR,
            "run_id": os.environ.get("RUN_ID"),
        },
    }
    for section, overrides in env_overrides.items():
        if section not in payload:
            payload[section] = {}
        for key, env_value in overrides.items():
            if env_value is not None:
                payload[section][key] = env_value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    if any([MODEL, API_BASE, API_KEY, INPUT_DIR, OUTPUT_DIR]):
        load_env_config(payload)

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    agent_config = AgentConfig(
        model=_optional_string(agent_payload.get("model", agent_defaults.model)),
        api_base=_optional_string(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=_optional_string(agent_payload.get("api_key", agent_defaults.api_key)),
        max_retries=int(agent_payload.get("max_retries", agent_defaults.max_retries)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
        execute_timeout_seconds=int(
            agent_payload.get(
                "execute_timeout_seconds",
                agent_defaults.execute_timeout_seconds,
            )
        ),
        max_output_bytes=int(
            agent_payload.get("max_output_bytes", agent_defaults.max_output_bytes)
        ),
        model_call_interval_seconds=float(
            agent_payload.get(
                "model_call_interval_seconds",
                agent_defaults.model_call_interval_seconds,
            )
        ),
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(
            run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)
        ),
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config)
