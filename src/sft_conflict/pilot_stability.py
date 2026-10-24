"""Standalone, standard-library-only engineering checks for calibration spread."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["calibration_normalizer", "calibration_stability_decision"]

_ARRAY_NAMES = (
    "auxiliary_to_target_ratios",
    "target_gradient_norms",
    "auxiliary_gradient_norms",
)
_QUANTILE_METHOD = "linear interpolation at (n - 1) * probability"


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_array(value: Any, *, allow_zeros: bool = False) -> list[float] | None:
    if not isinstance(value, list):
        return None
    result = []
    for item in value:
        number = _finite_float(item)
        if number is None or (number < 0 if allow_zeros else number <= 0):
            return None
        result.append(number)
    return result


def _linear_quantile(ordered: list[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    # Unlike (lower_value + upper_value) / 2, this also handles very large
    # positive medians without overflow and constant subnormals without underflow.
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def calibration_normalizer(ratios: Iterable[float], statistic: str = "median") -> float:
    """Return the median or RMS of every finite non-negative group ratio.

    Booleans, non-numeric entries, negative/non-finite values, and empty inputs
    raise ValueError. Zeros are retained, never dropped or replaced by epsilon;
    an all-zero input returns 0. RMS scales by the largest observation before
    squaring, so finite inputs cannot overflow the sum of squares. The median
    uses ordinary arithmetic except for an overflow-safe even-median fallback.

    For G measured groups, lambda = beta / RMS(r) implies
    max(lambda * r) <= beta * sqrt(G). This is an algebraic bound on these
    observations, NOT statistical robustness, outlier-proofing, or a bound on
    future training groups. Callers must reject a zero normalizer.
    """
    if not isinstance(statistic, str) or statistic not in {"median", "rms"}:
        raise ValueError("calibration statistic must be 'median' or 'rms'")
    message = "calibration ratios must be a non-empty iterable of finite non-negative numbers"
    if isinstance(ratios, str | bytes | Mapping):
        raise ValueError(message)
    try:
        iterator = iter(ratios)
    except TypeError as error:
        raise ValueError(message) from error
    values = []
    for item in iterator:
        number = _finite_float(item)
        if number is None or number < 0:
            raise ValueError(message)
        values.append(number)
    if not values:
        raise ValueError(message)
    if statistic == "median":
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        total = ordered[middle - 1] + ordered[middle]
        return total / 2 if math.isfinite(total) else _linear_quantile(ordered, 0.5)
    peak = max(values)
    if peak == 0:
        return 0.0
    return peak * math.sqrt(math.fsum((value / peak) ** 2 for value in values) / len(values))


def calibration_stability_decision(
    report: Mapping[str, Any],
    *,
    maximum_log10_iqr: float = 2.0,
    maximum_half_median_factor: float = 3.0,
    maximum_peak_to_median: float = 200.0,
    minimum_groups: int = 8,
    max_half_rms_factor: float = 3.0,
    max_group_energy_share: float = 0.75,
) -> dict[str, Any]:
    """Return a fail-closed decision and descriptions recomputed from raw ratios.

    Required evidence is three equal-length, one-dimensional JSON lists:
    ``auxiliary_to_target_ratios``, ``target_gradient_norms``, and
    ``auxiliary_gradient_norms``. Every entry must be a finite positive number
    (not a boolean), except that ``response_mode == 'three_segment'`` permits
    zero auxiliary norms and their matching raw ratios. Target norms remain
    strictly positive; positive-norm ratio underflow is not a legitimate zero.
    Auxiliary / target must match the corresponding ratio
    within relative tolerance 1e-9, with no absolute tolerance. Optional
    ``calibration_group_count`` and ``calibration_batches`` must match the actual
    array length. The ENTIRE report must serialize with ``allow_nan=False``.
    Stored summaries, weighted ratios, and lambda are not used for the statistics.

    Quartiles linearly interpolate sorted log10 ratios at (n - 1) * p, using
    only positive ratios for three-segment reports. The
    arithmetic medians of the first floor(n / 2) and remaining ordered ratios
    give half_median_factor = max(m1, m2) / min(m1, m2). Peak-to-median uses the
    maximum raw ratio / the arithmetic median of all raw ratios. Raw medians
    and half-splits always include zeros. Under median calibration, a zero
    median or either zero half median fails closed, with undefined factors
    returned as None. Threshold equality passes; values above any maximum fail.

    The default requires eight estimates, independently of report metadata.
    Explicit minimum_groups >= 2 supports tiny mathematical tests, not a claim
    of robust calibration. Valid evidence with 2 <= n < minimum_groups still
    receives descriptions but is rejected. The motivating 5-step and 10-step
    reference pilots each contain 32 estimates, not 5 and 10 estimates.
    Three-segment reports additionally require minimum_groups positive estimates
    (eight by default, at least two even with an explicit tiny-test minimum).

    An absent calibration_statistic means 'median', retaining all legacy gates.
    Explicit 'rms' is only supported for three-segment reports: all groups,
    including zeros, enter the full and ordered-half RMS values. Both halves
    must have positive RMS, their factor must be <= max_half_rms_factor, and
    max(r**2) / sum(r**2) must be <= max_group_energy_share. Energy shares also
    use scaled squares to avoid overflow. Median/log-IQR/peak-to-median remain
    descriptive ONLY in this branch, even when zero or overflowing. The RMS
    thresholds are included only in RMS decisions; legacy decisions are unchanged.

    Malformed/insufficient evidence returns accepted=False and reasons; invalid
    threshold arguments raise ValueError. Unavailable or overflowing statistics
    are None, never NaN/Infinity. No unvalidated input is echoed into the result,
    and neither the input nor any workflow, lambda, or model is changed.
    These are engineering thresholds, not a theorem, statistical confidence
    test, pilot-health check, or guarantee of downstream quality.
    """
    if type(minimum_groups) is not int or minimum_groups < 2:
        raise ValueError("minimum_groups must be an integer >= 2")
    thresholds: dict[str, int | float] = {"minimum_groups": minimum_groups}
    for name, value, lower_bound in (
        ("maximum_log10_iqr", maximum_log10_iqr, 0.0),
        ("maximum_half_median_factor", maximum_half_median_factor, 1.0),
        ("maximum_peak_to_median", maximum_peak_to_median, 1.0),
    ):
        number = _finite_float(value)
        if number is None or number < lower_bound:
            raise ValueError(f"{name} must be a finite number >= {lower_bound}")
        thresholds[name] = number
    half_rms_limit = _finite_float(max_half_rms_factor)
    if half_rms_limit is None or half_rms_limit < 1:
        raise ValueError("max_half_rms_factor must be a finite number >= 1")
    energy_share_limit = _finite_float(max_group_energy_share)
    if energy_share_limit is None or not 0 < energy_share_limit <= 1:
        raise ValueError("max_group_energy_share must be a finite number in (0, 1]")

    reasons: list[str] = []
    decision: dict[str, Any] = {
        "version": 1,
        "accepted": False,
        "reasons": reasons,
        "thresholds": thresholds,
        "group_count": None,
        "quantile_method": _QUANTILE_METHOD,
        "median_ratio": None,
        "log10_quartiles": None,
        "log10_iqr": None,
        "half_split": None,
        "half_median_factor": None,
        "peak_to_median": None,
        "notes": (
            "Engineering thresholds only, not a theorem or a guarantee of robust calibration, "
            "pilot health, generalization, or downstream quality. Statistics use unweighted "
            "raw group ratios; passing does not bound every weighted gradient ratio."
        ),
    }
    if not isinstance(report, Mapping):
        reasons.append("calibration report must be a mapping")
        return decision
    statistic = report.get("calibration_statistic", "median")
    if not isinstance(statistic, str) or statistic not in {"median", "rms"}:
        reasons.append("calibration_statistic must be 'median' or 'rms'")
        return decision
    rms = statistic == "rms"
    three_segment = report.get("response_mode") == "three_segment"
    if rms:
        thresholds = {
            "minimum_groups": minimum_groups,
            "max_half_rms_factor": half_rms_limit,
            "max_group_energy_share": energy_share_limit,
        }
        decision.update(
            thresholds=thresholds,
            normalization_statistic="rms",
            normalization_value=None,
            rms_half_factor=None,
            maximum_group_energy_share=None,
            notes=(
                "Engineering thresholds only, not a theorem of statistical robustness, "
                "outlier-proofing, pilot health, generalization, or downstream quality. "
                "All raw groups including zeros enter RMS and energy shares. For these G "
                "measured groups only, lambda = beta / RMS(r) gives the algebraic bound "
                "max(lambda * r) <= beta * sqrt(G), not a bound on future groups. "
                "Median, half medians, log10 IQR and peak-to-median are descriptive, not gates."
            ),
        )
        if not three_segment:
            reasons.append("calibration_statistic='rms' requires response_mode='three_segment'")
            return decision
    if three_segment:
        decision.update(
            zero_group_count=None,
            positive_group_count=None,
            log10_notes=(
                "Log10 quartiles and IQR use strictly positive raw group ratios only. "
                "At least minimum_groups positive estimates are required. All zeros remain "
                "in the full median, ordered half-split medians, and peak-to-median ratio; "
                "they are never replaced with epsilon. Undefined statistics are null."
            ),
        )
        if rms:
            decision["log10_notes"] = (
                "Log10 quartiles and IQR describe strictly positive raw ratios only. "
                "At least minimum_groups positive estimates are required. All zeros remain "
                "in full/half RMS values and raw medians, without epsilon. Median-based "
                "descriptions do not gate RMS calibration; undefined statistics are null."
            )
    try:
        json.dumps(dict(report), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        reasons.append("entire calibration report must be JSON-serializable without NaN/Infinity")

    arrays = {}
    raw_ratios = report.get("auxiliary_to_target_ratios")
    if isinstance(raw_ratios, list):
        decision["group_count"] = len(raw_ratios)
    for name in _ARRAY_NAMES:
        allow_zeros = three_segment and name != "target_gradient_norms"
        values = _positive_array(report.get(name), allow_zeros=allow_zeros)
        if values is None:
            requirement = "non-negative" if allow_zeros else "positive"
            reasons.append(f"{name} must be a one-dimensional list of finite {requirement} numbers")
        else:
            arrays[name] = values

    for name in ("calibration_group_count", "calibration_batches"):
        if name in report and (
            type(report[name]) is not int or report[name] != decision["group_count"]
        ):
            reasons.append(f"{name} must match the raw ratio group count")
    if len(arrays) == len(_ARRAY_NAMES):
        if len({len(values) for values in arrays.values()}) != 1:
            reasons.append("gradient norm and raw ratio arrays must have the same group count")
        else:
            for index, (ratio, target, auxiliary) in enumerate(
                zip(*(arrays[name] for name in _ARRAY_NAMES), strict=True)
            ):
                measured = auxiliary / target
                if (
                    not math.isfinite(measured)
                    or (three_segment and auxiliary > 0 and measured == 0)
                    or not math.isclose(ratio, measured, rel_tol=1e-9, abs_tol=0.0)
                ):
                    reasons.append(
                        f"auxiliary_to_target_ratios[{index}] is inconsistent with gradient norms"
                    )
                    break
    if reasons:
        return decision

    ratios = arrays["auxiliary_to_target_ratios"]
    count = len(ratios)
    if count < minimum_groups:
        reasons.append(
            f"insufficient calibration groups: {count} < minimum_groups ({minimum_groups})"
        )
    if three_segment:
        positive_count = sum(ratio > 0 for ratio in ratios)
        decision.update(
            zero_group_count=count - positive_count, positive_group_count=positive_count
        )
        if positive_count < minimum_groups:
            reasons.append(
                f"insufficient positive calibration groups: {positive_count} "
                f"< minimum_groups ({minimum_groups})"
            )
    if count < 2:
        return decision

    ordered = sorted(ratios)
    logs = [math.log10(ratio) for ratio in ordered if not three_segment or ratio > 0]
    if logs:
        lower_quartile = _linear_quantile(logs, 0.25)
        upper_quartile = _linear_quantile(logs, 0.75)
        decision.update(
            log10_quartiles={"p25": lower_quartile, "p75": upper_quartile},
            log10_iqr=upper_quartile - lower_quartile,
        )
    median = _linear_quantile(ordered, 0.5)
    split = count // 2
    first = _linear_quantile(sorted(ratios[:split]), 0.5)
    second = _linear_quantile(sorted(ratios[split:]), 0.5)
    decision.update(
        median_ratio=median,
        half_split={
            "first_group_count": split,
            "second_group_count": count - split,
            "first_median": first,
            "second_median": second,
        },
        half_median_factor=max(first, second) / min(first, second)
        if min(first, second) > 0
        else None,
        peak_to_median=ordered[-1] / median if median > 0 else None,
    )
    if rms:
        # Descriptive overflows are not evidence against RMS normalization.
        for name in ("log10_iqr", "half_median_factor", "peak_to_median"):
            if decision[name] is not None and not math.isfinite(decision[name]):
                decision[name] = None
        normalizer = calibration_normalizer(ratios, "rms")
        first_rms = calibration_normalizer(ratios[:split], "rms")
        second_rms = calibration_normalizer(ratios[split:], "rms")
        peak = ordered[-1]
        energy = math.fsum((ratio / peak) ** 2 for ratio in ratios) if peak > 0 else 0.0
        decision["half_split"].update(first_rms=first_rms, second_rms=second_rms)
        decision.update(
            normalization_value=normalizer,
            rms_half_factor=max(first_rms, second_rms) / min(first_rms, second_rms)
            if min(first_rms, second_rms) > 0
            else None,
            maximum_group_energy_share=1.0 / energy if energy > 0 else None,
        )
        if normalizer <= 0:
            reasons.append("RMS auxiliary gradient ratio is zero; calibration cannot normalize")
        if min(first_rms, second_rms) <= 0:
            reasons.append("ordered half RMS values must be positive; rms_half_factor is undefined")
        for name, maximum_name in (
            ("rms_half_factor", "max_half_rms_factor"),
            ("maximum_group_energy_share", "max_group_energy_share"),
        ):
            value = decision[name]
            if value is None:
                continue
            if not math.isfinite(value):
                decision[name] = None
                reasons.append(f"{name} overflowed and exceeds {maximum_name}")
            elif value > thresholds[maximum_name]:
                reasons.append(f"{name} exceeds {maximum_name}")
        decision["accepted"] = not reasons
        return decision
    if median <= 0:
        reasons.append(
            "median auxiliary gradient ratio is zero; pilot drift remains inside safety core, "
            "increase covered groups or use validated later pilot"
        )
    if min(first, second) <= 0:
        reasons.append("ordered half medians must be positive; half_median_factor is undefined")
    for name in ("log10_iqr", "half_median_factor", "peak_to_median"):
        maximum_name = f"maximum_{name}"
        value = decision[name]
        if value is None:
            continue
        if not math.isfinite(value):
            # With positive finite inputs, an overflowing factor exceeds every
            # permitted finite threshold. Reject without emitting invalid JSON.
            decision[name] = None
            reasons.append(f"{name} overflowed and exceeds {maximum_name}")
        elif value > thresholds[maximum_name]:
            reasons.append(f"{name} exceeds {maximum_name}")
    decision["accepted"] = not reasons
    return decision
