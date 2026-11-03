"""YAML configuration loading, composition, and CLI overrides."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from cnn_classifier.config.resolve import resolve_config, validate_config_semantics
from cnn_classifier.config.schema import RunConfig
from cnn_classifier.core.errors import ConfigError

_DEFAULT_CONFIG = Path(__file__).with_name("presets") / "default.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"config {path} must contain a YAML mapping")
    return payload


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; lists and scalars replace."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_chain(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ConfigError(f"cyclic defaults chain at {path}")
    seen = {*seen, resolved}
    payload = _load_yaml(resolved)
    defaults = payload.pop("defaults", [])
    if not isinstance(defaults, list):
        raise ConfigError(f"defaults in {path} must be a list")
    merged: dict[str, Any] = {}
    for entry in defaults:
        if not isinstance(entry, str):
            raise ConfigError(f"defaults entries in {path} must be strings")
        merged = deep_merge(merged, _load_chain(resolved.parent / entry, seen))
    return deep_merge(merged, payload)


def apply_overrides(raw: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``a.b=value`` overrides without silently growing the schema."""
    result = copy.deepcopy(dict(raw))
    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"override must be key=value; got {override!r}")
        dotted, value_text = override.split("=", 1)
        parts = dotted.split(".")
        cursor: Any = result
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ConfigError(f"unknown override path: {dotted}")
            cursor = cursor[part]
        leaf = parts[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise ConfigError(f"unknown override path: {dotted}")
        try:
            parsed = yaml.safe_load(value_text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid override value for {dotted}: {value_text!r}") from exc
        cursor[leaf] = parsed
    return result


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> RunConfig:
    """Load, compose, resolve, and validate an experiment configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file does not exist: {config_path}")
    packaged = _load_yaml(_DEFAULT_CONFIG)
    merged = deep_merge(packaged, _load_chain(config_path, seen=set()))
    merged = apply_overrides(merged, overrides)
    try:
        config = RunConfig.model_validate(merged)
    except Exception as exc:
        raise ConfigError(f"invalid configuration {config_path}: {exc}") from exc
    config = resolve_config(config, config_path.resolve().parent)
    validate_config_semantics(config)
    return config
