"""YAML config loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml

from quant.config.schema import AppConfig, FuturesAppConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
ConfigT = Union[AppConfig, FuturesAppConfig]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping in {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path, default_path: Path | None = None) -> ConfigT:
    raw = _read_yaml(Path(path))
    mode = raw.get("mode", "equity")
    if mode == "futures":
        return FuturesAppConfig.model_validate(raw)
    if mode != "equity":
        raise ValueError(f"unknown config mode: {mode!r}")
    base = _read_yaml(default_path or DEFAULT_CONFIG_PATH)
    return AppConfig.model_validate(_deep_merge(base, raw))


def config_to_dict(cfg: ConfigT) -> dict[str, Any]:
    return cfg.model_dump(mode="python")

