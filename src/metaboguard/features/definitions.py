"""Typed, configuration-driven engineered-feature registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from metaboguard.config import load_config
from metaboguard.features.dictionary import FEATURE_DICTIONARY, get_feature_definition


@dataclass(frozen=True, slots=True)
class EngineeredFeatureDefinition:
    feature_id: str
    display_name: str
    source_feature_id: str
    feature_family: str
    value_type: Literal["numeric", "categorical", "integer"]
    unit: str
    window_days: int | None
    minimum_measurements: int
    allowed_cohort_classes: tuple[str, ...]
    allowed_endpoint_families: tuple[str, ...]
    uses_conditions: bool
    uses_events: bool
    uses_encounters: bool
    requires_numeric_value: bool
    missingness_policy: str
    leakage_risk: str
    definition_version: str = "1.0.0"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _windows() -> dict[str, int | None]:
    return {
        str(name): (None if value is None else int(value))
        for name, value in load_config()["features"]["feature_windows_days"].items()
    }


def build_feature_registry() -> dict[str, EngineeredFeatureDefinition]:
    """Build all allowed initial feature families from the Step 2 dictionary."""
    registry: dict[str, EngineeredFeatureDefinition] = {}
    windows = _windows()
    allowed_sources = [
        name for name, definition in FEATURE_DICTIONARY.items() if definition.allowed
    ]
    for source in allowed_sources:
        definition = get_feature_definition(source)
        for window_name, window_days in windows.items():
            suffix = "lifetime" if window_days is None else window_name
            base: dict[str, Any] = {
                "source_feature_id": source,
                "unit": definition.canonical_unit,
                "allowed_cohort_classes": (
                    "ordinary_incidence",
                    "enriched_incidence",
                    "enriched_pancreatic",
                    "enriched_multicancer",
                ),
                "allowed_endpoint_families": ("cancer", "diabetes"),
                "uses_conditions": False,
                "uses_events": True,
                "uses_encounters": False,
                "requires_numeric_value": True,
                "missingness_policy": "null_summary_binary_missingness_no_imputation",
                "leakage_risk": "pre_index_measurement_only",
            }
            for family, minimum in (("summary", 1), ("change", 2), ("trajectory", 3)):
                names = {
                    "summary": (
                        "count",
                        "mean",
                        "median",
                        "min",
                        "max",
                        "std",
                        "range",
                        "recency_days",
                    ),
                    "change": (
                        "baseline",
                        "absolute_change",
                        "relative_change",
                        "time_span_days",
                    ),
                    "trajectory": (
                        "slope_per_year",
                        "slope_measurement_count",
                        "slope_time_span_days",
                        "trend_direction",
                    ),
                }[family]
                for metric in names:
                    feature_id = f"{source}__{metric}_{suffix}"
                    registry[feature_id] = EngineeredFeatureDefinition(
                        feature_id=feature_id,
                        display_name=feature_id,
                        feature_family=family,
                        value_type=(
                            "categorical" if metric == "trend_direction" else "numeric"
                        ),
                        window_days=window_days,
                        minimum_measurements=minimum,
                        **base,
                    )
            for metric in ("latest", "latest_date_offset_days", "latest_window"):
                feature_id = f"{source}__{metric}_{window_name}"
                registry[feature_id] = EngineeredFeatureDefinition(
                    feature_id=feature_id,
                    display_name=feature_id,
                    source_feature_id=source,
                    feature_family="latest",
                    value_type=(
                        "numeric" if metric != "latest_window" else "categorical"
                    ),
                    unit=(
                        definition.canonical_unit
                        if metric == "latest"
                        else "days" if metric == "latest_date_offset_days" else "coded"
                    ),
                    window_days=window_days,
                    minimum_measurements=1,
                    allowed_cohort_classes=(
                        "ordinary_incidence",
                        "enriched_incidence",
                        "enriched_pancreatic",
                        "enriched_multicancer",
                    ),
                    allowed_endpoint_families=("cancer", "diabetes"),
                    uses_conditions=False,
                    uses_events=True,
                    uses_encounters=False,
                    requires_numeric_value=True,
                    missingness_policy="null_when_unobserved",
                    leakage_risk="pre_index_measurement_only",
                )
        for scope, window_days in (
            ("lifetime", None),
            ("recent", windows.get("recent")),
        ):
            suffix = scope
            for metric in ("observed", "missing"):
                feature_id = f"{source}__{metric}_{suffix}"
                registry[feature_id] = EngineeredFeatureDefinition(
                    feature_id=feature_id,
                    display_name=feature_id,
                    source_feature_id=source,
                    feature_family="missingness",
                    value_type="integer",
                    unit="binary",
                    window_days=window_days,
                    minimum_measurements=1,
                    allowed_cohort_classes=(
                        "ordinary_incidence",
                        "enriched_incidence",
                        "enriched_pancreatic",
                        "enriched_multicancer",
                    ),
                    allowed_endpoint_families=("cancer", "diabetes"),
                    uses_conditions=False,
                    uses_events=True,
                    uses_encounters=False,
                    requires_numeric_value=True,
                    missingness_policy="binary_complement",
                    leakage_risk="pre_index_measurement_only",
                )
        for metric in ("latest", "latest_date_offset_days", "latest_window"):
            feature_id = f"{source}__{metric}"
            registry[feature_id] = EngineeredFeatureDefinition(
                feature_id=feature_id,
                display_name=feature_id,
                source_feature_id=source,
                feature_family="latest",
                value_type="numeric" if metric != "latest_window" else "categorical",
                unit=(
                    definition.canonical_unit
                    if metric == "latest"
                    else "days" if metric == "latest_date_offset_days" else "coded"
                ),
                window_days=None,
                minimum_measurements=1,
                allowed_cohort_classes=(
                    "ordinary_incidence",
                    "enriched_incidence",
                    "enriched_pancreatic",
                    "enriched_multicancer",
                ),
                allowed_endpoint_families=("cancer", "diabetes"),
                uses_conditions=False,
                uses_events=True,
                uses_encounters=False,
                requires_numeric_value=True,
                missingness_policy="null_when_unobserved",
                leakage_risk="pre_index_measurement_only",
            )
    return dict(sorted(registry.items()))


def registry_hash(registry: dict[str, EngineeredFeatureDefinition]) -> str:
    payload = json.dumps(
        [item.to_dict() for item in registry.values()],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
