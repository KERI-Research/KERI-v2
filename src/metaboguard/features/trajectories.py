"""Descriptive pre-index change and trajectory calculations."""

from __future__ import annotations

from datetime import date

import numpy as np


def summaries(values: list[tuple[date, float]]) -> dict[str, float | int | None]:
    """Calculate deterministic summary statistics without imputation."""
    ordered = sorted(values, key=lambda item: item[0])
    numbers = np.array([value for _, value in ordered], dtype=float)
    if not len(numbers):
        return {
            name: None for name in ("mean", "median", "min", "max", "std", "range", "recency_days")
        }
    return {
        "mean": float(np.mean(numbers)),
        "median": float(np.median(numbers)),
        "min": float(np.min(numbers)),
        "max": float(np.max(numbers)),
        "std": float(np.std(numbers, ddof=1)) if len(numbers) >= 2 else None,
        "range": float(np.max(numbers) - np.min(numbers)) if len(numbers) >= 2 else None,
        "recency_days": 0,
    }


def change(values: list[tuple[date, float]]) -> dict[str, float | int | None]:
    ordered = sorted(values, key=lambda item: item[0])
    if len(ordered) < 2 or ordered[0][0] == ordered[-1][0]:
        return {
            "baseline": ordered[0][1] if ordered else None,
            "absolute_change": None,
            "relative_change": None,
            "time_span_days": 0 if ordered else None,
        }
    baseline = ordered[0][1]
    latest = ordered[-1][1]
    return {
        "baseline": baseline,
        "absolute_change": latest - baseline,
        "relative_change": (latest - baseline) / abs(baseline) if baseline != 0 else None,
        "time_span_days": (ordered[-1][0] - ordered[0][0]).days,
    }


def trajectory(
    values: list[tuple[date, float]], epsilon: float = 0.001
) -> dict[str, float | int | str | None]:
    ordered = sorted(values, key=lambda item: item[0])
    dates = sorted({item[0] for item in ordered})
    if len(dates) < 3 or (dates[-1] - dates[0]).days < 30:
        return {
            "slope_per_year": None,
            "slope_measurement_count": len(ordered),
            "slope_time_span_days": (dates[-1] - dates[0]).days if len(dates) >= 2 else None,
            "trend_direction": "unknown",
        }
    x = np.array([(item[0] - dates[0]).days / 365.25 for item in ordered], dtype=float)
    y = np.array([item[1] for item in ordered], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return {
        "slope_per_year": slope,
        "slope_measurement_count": len(ordered),
        "slope_time_span_days": (dates[-1] - dates[0]).days,
        "trend_direction": "increasing"
        if slope > epsilon
        else "decreasing"
        if slope < -epsilon
        else "stable",
    }
