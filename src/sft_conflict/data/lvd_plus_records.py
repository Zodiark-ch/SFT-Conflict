"""Immutable JSON snapshots of accepted LVD+ hybrid scales and automatic lambda.

Only JSON structure, digests and measurement arithmetic are checked here. The
caller constructs the actual model/data/objective binding and applies stability
policy. No model, configuration, tensor or other artifact is opened by this module.
Digests establish integrity, not authenticity of the producer's measurements.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["LVDPlusRecord", "canonical_sha256", "publish_lvd_plus_record"]

_KINDS = {"lvd_plus_scales", "lvd_plus_lambda"}
_PAYLOAD_KEYS = {"schema_version", "kind", "binding", "binding_sha256", "values", "report"}
_RECORD_KEYS = _PAYLOAD_KEYS | {"payload_sha256"}


def _check_json(value: Any, active: set[int]) -> None:
    """Reject coercions (including tuples, Paths, enums and non-string keys)."""
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return
    if value_type not in (dict, list):
        raise ValueError("non-JSON type")
    identity = id(value)
    if identity in active:
        raise ValueError("cyclic JSON value")
    active.add(identity)
    try:
        if value_type is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("non-string JSON key")
            children = value.values()
        else:
            children = value
        for child in children:
            _check_json(child, active)
    finally:
        active.remove(identity)


def _canonical_json(value: Any, field: str = "value") -> str:
    try:
        _check_json(value, set())
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        # Do not include values or third-party repr/error messages in diagnostics.
        raise ValueError(f"{field} must contain only finite plain JSON with string keys") from None


def canonical_sha256(value: Any) -> str:
    """Hash strict JSON using conventional sorted, compact, ASCII-escaped UTF-8."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _object(value: Any, field: str, *, nonempty: bool = False) -> dict[str, Any]:
    if type(value) is not dict or (nonempty and not value):
        qualifier = "nonempty " if nonempty else ""
        raise ValueError(f"{field} must be a {qualifier}JSON object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{field} must have string keys")
    return value


def _exact_keys(value: Any, keys: set[str], field: str) -> None:
    _object(value, field)
    missing, unknown = keys - value.keys(), value.keys() - keys
    if missing or unknown:
        raise ValueError(
            f"{field} fields mismatch: missing keys {json.dumps(sorted(missing))}; "
            f"unknown keys {json.dumps(sorted(unknown))}"
        )


def _kind(value: Any) -> str:
    if type(value) is not str or value not in _KINDS:
        raise ValueError("record kind must be lvd_plus_scales or lvd_plus_lambda")
    return value


def _digest(value: Any, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def _number(value: Any, field: str, *, positive: bool = True) -> float:
    qualifier = "positive" if positive else "nonnegative"
    message = f"{field} must be a finite {qualifier} number, not bool"
    if type(value) not in (int, float):
        raise ValueError(message)
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(message) from None
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise ValueError(message)
    return number


def _count(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer, not bool")
    return value


def _array(value: Any, count: int, field: str, *, positive: bool = False) -> list[float]:
    if type(value) is not list or len(value) != count:
        raise ValueError(f"{field} must be a list matching the measurement count")
    return [
        _number(item, f"{field}[{index}]", positive=positive) for index, item in enumerate(value)
    ]


def _close(actual: float, expected: float, field: str, *, tolerance: float) -> None:
    # No absolute tolerance: tiny positive scales/ratios must not become zero.
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=0.0):
        raise ValueError(f"{field} is inconsistent with the recorded measurements")


def _mean(values: list[float]) -> float:
    try:
        return math.fsum(values) / len(values)
    except OverflowError:
        # A finite nonnegative mean can exist even when its unscaled sum overflows.
        peak = max(values)
        return peak * (math.fsum(value / peak for value in values) / len(values))


def _rms(values: list[float]) -> float:
    peak = max(values)
    if peak == 0:
        return 0.0
    return peak * math.sqrt(math.fsum((value / peak) ** 2 for value in values) / len(values))


def _validate_scales(values: dict[str, Any], report: dict[str, Any]) -> None:
    prefix = "lvd_plus_scales"
    _exact_keys(values, {"outer_scale", "risk_scale"}, f"{prefix}.values")
    count = _count(report.get("microbatch_count"), f"{prefix}.report.microbatch_count")
    positions = report.get("micro_position_counts")
    if type(positions) is not list or len(positions) != count:
        raise ValueError(f"{prefix}.report.micro_position_counts must match microbatch_count")
    for index, position_count in enumerate(positions):
        _count(position_count, f"{prefix}.report.micro_position_counts[{index}]")
    for component in ("outer", "risk"):
        field = f"{prefix}.values.{component}_scale"
        scale = _number(values[f"{component}_scale"], field)
        measurements = _array(
            report.get(f"per_micro_{component}"), count, f"{prefix}.report.per_micro_{component}"
        )
        mean = _number(_mean(measurements), f"{prefix}.report.per_micro_{component} mean")
        _close(scale, mean, field, tolerance=1e-12)


def _validate_lambda(values: dict[str, Any], report: dict[str, Any]) -> None:
    prefix = "lvd_plus_lambda"
    _exact_keys(values, {"resolved_lambda"}, f"{prefix}.values")
    resolved = _number(values["resolved_lambda"], f"{prefix}.values.resolved_lambda")
    if report.get("method") != "target_plus_anchor_lvd+":
        raise ValueError(f"{prefix}.report.method must be target_plus_anchor_lvd+")
    if report.get("calibration_statistic") != "rms":
        raise ValueError(f"{prefix}.report.calibration_statistic must be rms")
    stability = _object(report.get("stability"), f"{prefix}.report.stability")
    if stability.get("accepted") is not True:
        raise ValueError(f"{prefix}.report.stability.accepted must be true")
    count = _count(
        report.get("calibration_group_count"), f"{prefix}.report.calibration_group_count"
    )
    ratios = _array(
        report.get("auxiliary_to_target_ratios"),
        count,
        f"{prefix}.report.auxiliary_to_target_ratios",
    )
    targets = _array(
        report.get("target_gradient_norms"),
        count,
        f"{prefix}.report.target_gradient_norms",
        positive=True,
    )
    auxiliaries = _array(
        report.get("auxiliary_gradient_norms"), count, f"{prefix}.report.auxiliary_gradient_norms"
    )
    for index, (ratio, target, auxiliary) in enumerate(
        zip(ratios, targets, auxiliaries, strict=True)
    ):
        field = f"{prefix}.report.auxiliary_to_target_ratios[{index}]"
        measured = auxiliary / target
        if not math.isfinite(measured) or (auxiliary > 0 and measured == 0):
            raise ValueError(f"{field} has non-finite or underflowed norm arithmetic")
        _close(ratio, measured, field, tolerance=1e-9)
    rms = _number(_rms(ratios), f"{prefix}.report RMS")
    for key in ("normalization_value", "rms_auxiliary_to_target_ratio"):
        field = f"{prefix}.report.{key}"
        _close(_number(report.get(key), field), rms, field, tolerance=1e-9)
    beta = _number(report.get("target_gradient_ratio"), f"{prefix}.report.target_gradient_ratio")
    calculated = _number(beta / rms, f"{prefix}.report beta / RMS resolved_lambda")
    _close(resolved, calculated, f"{prefix}.values.resolved_lambda", tolerance=1e-9)
    for key in ("resolved_lambda", "candidate_lambda"):
        field = f"{prefix}.report.{key}"
        _close(_number(report.get(key), field), resolved, field, tolerance=1e-9)
    weighted = [resolved * ratio for ratio in ratios]
    if any(
        not math.isfinite(value) or (ratio > 0 and value == 0)
        for value, ratio in zip(weighted, ratios, strict=True)
    ):
        raise ValueError(
            f"{prefix}.report weighted ratios have non-finite or underflowed arithmetic"
        )
    weighted_rms = _number(_rms(weighted), f"{prefix}.report weighted RMS")
    _close(
        weighted_rms, beta, f"{prefix}.report weighted RMS / target_gradient_ratio", tolerance=1e-9
    )
    field = f"{prefix}.report.rms_weighted_auxiliary_to_target_ratio"
    claimed = _number(report.get("rms_weighted_auxiliary_to_target_ratio"), field)
    _close(claimed, weighted_rms, field, tolerance=1e-9)
    _close(claimed, beta, field, tolerance=1e-9)
    # The producer emits this array; retain compatibility with minimal reports
    # but verify the extra numerical claim whenever it is present.
    if "weighted_auxiliary_to_target_ratios" in report:
        field = f"{prefix}.report.weighted_auxiliary_to_target_ratios"
        claimed_ratios = _array(report["weighted_auxiliary_to_target_ratios"], count, field)
        for index, (claim, measured) in enumerate(zip(claimed_ratios, weighted, strict=True)):
            _close(claim, measured, f"{field}[{index}]", tolerance=1e-9)


def _validate_record(payload: Any) -> None:
    _exact_keys(payload, _RECORD_KEYS, "LVD+ record")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("LVD+ record schema_version must be integer 1")
    kind = _kind(payload["kind"])
    binding = _object(payload["binding"], f"{kind}.binding", nonempty=True)
    values = _object(payload["values"], f"{kind}.values")
    report = _object(payload["report"], f"{kind}.report")
    _canonical_json(payload, kind)
    binding_digest = _digest(payload["binding_sha256"], f"{kind}.binding_sha256")
    if binding_digest != canonical_sha256(binding):
        raise ValueError(f"{kind}.binding_sha256 does not match binding")
    payload_digest = _digest(payload["payload_sha256"], f"{kind}.payload_sha256")
    if payload_digest != canonical_sha256({key: payload[key] for key in _PAYLOAD_KEYS}):
        raise ValueError(f"{kind}.payload_sha256 does not match record content")
    if report.get("status") != "passed":
        raise ValueError(f"{kind}.report.status must be passed")
    if report.get("accepted") is not True:
        raise ValueError(f"{kind}.report.accepted must be true")
    if kind == "lvd_plus_scales":
        _validate_scales(values, report)
    else:
        _validate_lambda(values, report)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("LVD+ record contains a duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError("LVD+ record contains a non-finite JSON number")


@dataclass(frozen=True, init=False, repr=False)
class LVDPlusRecord:
    """An immutable, verified snapshot of one JSON file, never a directory.

    Properties return fresh JSON objects. ``sha256`` identifies the actual bytes
    read, including whitespace; ``payload_sha256`` verifies canonical content.
    Reformatting a file changes its identity even if its content proof still holds.
    """

    _kind: str
    _sha256: str
    _binding_sha256: str
    _binding_json: str
    _values_json: str
    _report_json: str

    def __init__(self, path: str | Path) -> None:
        source = Path(path)
        if source.is_dir():
            raise ValueError("LVD+ record path must be a JSON file, not a directory")
        raw = source.read_bytes()
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, json.JSONDecodeError, RecursionError):
            raise ValueError("LVD+ record must contain one valid UTF-8 JSON object") from None
        _validate_record(payload)
        object.__setattr__(self, "_kind", payload["kind"])
        object.__setattr__(self, "_sha256", hashlib.sha256(raw).hexdigest())
        object.__setattr__(self, "_binding_sha256", payload["binding_sha256"])
        for name in ("binding", "values", "report"):
            object.__setattr__(self, f"_{name}_json", _canonical_json(payload[name]))

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def sha256(self) -> str:
        return self._sha256

    @property
    def binding(self) -> dict[str, Any]:
        return json.loads(self._binding_json)

    @property
    def values(self) -> dict[str, Any]:
        return json.loads(self._values_json)

    @property
    def report(self) -> dict[str, Any]:
        return json.loads(self._report_json)

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": self.kind,
            "manifest_sha256": self.sha256,
            "binding_sha256": self._binding_sha256,
        }

    def validate_binding(self, expected: dict, *, expected_sha256: str | None = None) -> None:
        """Match strict canonical JSON and, optionally, an externally pinned file SHA.

        Diagnostics name differing top-level binding keys but never their values.
        Integer, float and bool representations are intentionally not interchangeable.
        """
        _object(expected, f"{self.kind}.expected binding", nonempty=True)
        expected_json = _canonical_json(expected, f"{self.kind}.expected binding")
        if expected_json != self._binding_json:
            actual = self.binding
            missing = sorted(expected.keys() - actual.keys())
            unexpected = sorted(actual.keys() - expected.keys())
            different = sorted(
                key
                for key in actual.keys() & expected.keys()
                if _canonical_json(actual[key]) != _canonical_json(expected[key])
            )
            raise ValueError(
                f"{self.kind} binding mismatch: missing keys {json.dumps(missing)}; "
                f"unexpected keys {json.dumps(unexpected)}; differing keys {json.dumps(different)}"
            )
        if expected_sha256 is not None:
            _digest(expected_sha256, f"{self.kind}.expected_sha256")
            if expected_sha256 != self.sha256:
                raise ValueError(f"{self.kind} expected_sha256 does not match actual file SHA256")


def publish_lvd_plus_record(
    path: str | Path, *, kind: str, binding: dict, values: dict, report: dict
) -> LVDPlusRecord:
    """Publish a complete, fsynced JSON file with an exclusive atomic hard link.

    Every occupied destination raises FileExistsError, even for identical bytes
    or a dangling symlink. Reuse is the caller's separate load/validate decision.
    Only this call's randomized temporary file is cleaned up, never a destination.
    The caller owns the artifact directory; parent symlinks may resolve normally.
    """
    requested = Path(path)
    # Resolve the parent, NOT the final component: a dangling final symlink is
    # an occupied name and must not redirect publication into its absent target.
    destination = requested.parent.resolve() / requested.name
    if os.path.lexists(destination):
        raise FileExistsError(errno.EEXIST, "LVD+ record path already exists", str(destination))
    kind = _kind(kind)
    snapshot = {}
    for name, value in (("binding", binding), ("values", values), ("report", report)):
        _object(value, f"{kind}.{name}", nonempty=name == "binding")
        snapshot[name] = json.loads(_canonical_json(value, f"{kind}.{name}"))
    payload = {
        "schema_version": 1,
        "kind": kind,
        **snapshot,
        "binding_sha256": canonical_sha256(snapshot["binding"]),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    _validate_record(payload)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return LVDPlusRecord(destination)
