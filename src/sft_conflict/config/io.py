"""Strict YAML loading and serialization for experiment configs."""

from __future__ import annotations

import types
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

from .schema import AnchorGeometryConfig, ExperimentConfig, LVDConfig, LVDPlusConfig, S0Config

FORBIDDEN_LEGACY_KEYS = frozenset(
    {
        "component_scores",
        "head_rank",
        "lora_info_dir",
        "lora_mode",
        "mask_path",
        "mask_ratio",
        "mask_score_type",
        "probing",
        "projection_matrix",
        "rank_pattern",
        "sophia",
        "two_stage_alternating",
        "unlearning",
    }
)


class ConfigError(ValueError):
    """Raised when a configuration violates the public schema."""


def _reject_legacy_keys(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_LEGACY_KEYS:
                raise ConfigError(
                    f"legacy key '{child_path}' is not supported; "
                    "use the explicit config migration tool"
                )
            _reject_legacy_keys(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_legacy_keys(child, f"{path}[{index}]")


def _decode_value(annotation: Any, value: Any, path: str) -> Any:
    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if value is None:
        if type(None) in arguments:
            return None
        raise ConfigError(f"{path} may not be null")

    if origin in {Union, types.UnionType}:
        failures: list[str] = []
        for option in arguments:
            if option is type(None):
                continue
            try:
                return _decode_value(option, value, path)
            except ConfigError as exc:
                failures.append(str(exc))
        raise ConfigError(f"{path} does not match any allowed type: {'; '.join(failures)}")

    if is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ConfigError(f"{path} must be a mapping")
        return _decode_dataclass(annotation, value, path)

    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{path} must be a list")
        item_type = arguments[0] if arguments else Any
        return tuple(
            _decode_value(item_type, item, f"{path}[{index}]") for index, item in enumerate(value)
        )

    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a list")
        item_type = arguments[0] if arguments else Any
        return [
            _decode_value(item_type, item, f"{path}[{index}]") for index, item in enumerate(value)
        ]

    if origin is Literal:
        if value not in arguments:
            raise ConfigError(f"{path} must be one of {arguments}, got {value!r}")
        return value

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except ValueError as exc:
            choices = ", ".join(member.value for member in annotation)
            raise ConfigError(f"{path} must be one of: {choices}") from exc

    if annotation is float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ConfigError(f"{path} must be a number, got {type(value).__name__}")
        try:
            return float(value)
        except OverflowError as exc:
            raise ConfigError(f"{path} must be representable as a finite float") from exc
    if annotation in {str, int, bool} and not isinstance(value, annotation):
        raise ConfigError(f"{path} must be {annotation.__name__}, got {type(value).__name__}")
    return value


def _decode_dataclass(cls: type[Any], raw: dict[str, Any], path: str) -> Any:
    field_map = {field.name: field for field in fields(cls)}
    unknown = sorted(set(raw) - set(field_map))
    if unknown:
        joined = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {joined}")

    hints = get_type_hints(cls)
    values = {key: _decode_value(hints[key], value, f"{path}.{key}") for key, value in raw.items()}
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"invalid {path}: {exc}") from exc


def parse_config(raw: dict[str, Any]) -> ExperimentConfig:
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    _reject_legacy_keys(raw)
    config = _decode_dataclass(ExperimentConfig, raw, "config")
    try:
        config.validate()
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read configuration {config_path}: {exc}") from exc
    return parse_config(raw)


def load_lvd_config(path: str | Path) -> LVDConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read LVD configuration {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("LVD configuration root must be a mapping")
    _reject_legacy_keys(raw, "lvd")
    config = _decode_dataclass(LVDConfig, raw, "lvd")
    try:
        config.validate()
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return config


def load_lvd_plus_config(path: str | Path) -> LVDPlusConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read LVD+ configuration {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("LVD+ configuration root must be a mapping")
    _reject_legacy_keys(raw, "lvd_plus")
    config = _decode_dataclass(LVDPlusConfig, raw, "lvd_plus")
    try:
        config.validate()
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return config


def load_s0_config(path: str | Path) -> S0Config:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read S0 configuration {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("S0 configuration root must be a mapping")
    _reject_legacy_keys(raw, "s0")
    config = _decode_dataclass(S0Config, raw, "s0")
    try:
        config.validate()
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return config


def load_anchor_geometry_config(path: str | Path) -> AnchorGeometryConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read geometry configuration {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("anchor_geometry configuration root must be a mapping")
    _reject_legacy_keys(raw, "anchor_geometry")
    config = _decode_dataclass(AnchorGeometryConfig, raw, "anchor_geometry")
    try:
        config.validate()
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return config


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _serialize(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_serialize(child) for child in value]
    return value


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(_serialize(asdict(config)), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
