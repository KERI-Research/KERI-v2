"""Immutable Step 4 endpoint, index, label, and split contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, cast

from metaboguard.config import load_config

OutcomeFamily = Literal["cancer", "diabetes"]
LabelState = Literal["positive", "eligible_negative", "censored", "competing_death", "excluded"]


def _int_value(value: object) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"Expected numeric configuration value, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class EndpointProtocol:
    """One endpoint's eligibility, washout, and horizon protocol."""

    endpoint_id: str
    outcome_family: OutcomeFamily
    cancer_site: str | None
    diabetes_type: Literal["type1", "type2", "gestational"] | None
    horizon_years: tuple[int, ...]
    minimum_age_years: int
    minimum_history_days: int
    minimum_preindex_encounters: int
    minimum_preindex_measurements: int
    rolling_index_interval_days: int
    prevalent_exclusion: bool
    washout_days: int
    use_competing_death_risk: bool
    enabled: bool
    definition_version: str = "1.0.0"

    @classmethod
    def from_mapping(
        cls, endpoint_id: str, mapping: dict[str, object], definition_version: str
    ) -> EndpointProtocol:
        """Build a typed protocol from the YAML registry."""
        return cls(
            endpoint_id=endpoint_id,
            outcome_family=cast(OutcomeFamily, mapping["outcome_family"]),
            cancer_site=cast(str | None, mapping.get("cancer_site")),
            diabetes_type=cast(
                Literal["type1", "type2", "gestational"] | None,
                mapping.get("diabetes_type"),
            ),
            horizon_years=tuple(
                _int_value(value) for value in cast(list[object], mapping["horizon_years"])
            ),
            minimum_age_years=_int_value(mapping["minimum_age_years"]),
            minimum_history_days=_int_value(mapping["minimum_history_days"]),
            minimum_preindex_encounters=_int_value(mapping["minimum_preindex_encounters"]),
            minimum_preindex_measurements=_int_value(mapping["minimum_preindex_measurements"]),
            rolling_index_interval_days=_int_value(mapping["rolling_index_interval_days"]),
            prevalent_exclusion=bool(mapping["prevalent_exclusion"]),
            washout_days=_int_value(mapping["washout_days"]),
            use_competing_death_risk=bool(mapping["use_competing_death_risk"]),
            enabled=bool(mapping["enabled"]),
            definition_version=definition_version,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible protocol mapping."""
        return {
            "endpoint_id": self.endpoint_id,
            "outcome_family": self.outcome_family,
            "cancer_site": self.cancer_site,
            "diabetes_type": self.diabetes_type,
            "horizon_years": list(self.horizon_years),
            "minimum_age_years": self.minimum_age_years,
            "minimum_history_days": self.minimum_history_days,
            "minimum_preindex_encounters": self.minimum_preindex_encounters,
            "minimum_preindex_measurements": self.minimum_preindex_measurements,
            "rolling_index_interval_days": self.rolling_index_interval_days,
            "prevalent_exclusion": self.prevalent_exclusion,
            "washout_days": self.washout_days,
            "use_competing_death_risk": self.use_competing_death_risk,
            "enabled": self.enabled,
            "definition_version": self.definition_version,
        }


@dataclass(frozen=True, slots=True)
class PatientIndex:
    """One immutable patient prediction cut-off and permitted history summary."""

    patient_id: str
    cohort_class: str
    endpoint_id: str
    index_date: date
    index_source: str
    index_sequence_number: int
    preindex_event_dates: tuple[date, ...]
    preindex_measurement_dates: tuple[date, ...]
    preindex_encounter_count: int
    exclusion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class HorizonLabel:
    """One endpoint-specific label at one horizon."""

    patient_id: str
    cohort_class: str
    endpoint_id: str
    index_date: date
    horizon_years: int
    label_state: LabelState
    event_date: date | None
    death_date: date | None
    censor_date: date | None
    days_to_event_or_censor: int | None
    endpoint_definition_version: str
    outcome_source_condition_code: str | None = None
    exclusion_reason: str | None = None


@dataclass(slots=True)
class ConstructedCohort:
    """Endpoint-specific immutable cohort records before feature engineering."""

    cohort_class: str
    endpoint: EndpointProtocol
    patient_indexes: list[PatientIndex]
    excluded_indexes: list[PatientIndex]
    labels: list[HorizonLabel]
    source_canonical_sha256: str
    source_generation_manifest_sha256: str = ""
    date_normalisation_audit_sha256: str = ""
    output_dir: object | None = None


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Patient-level development and temporal-holdout split settings."""

    root_seed: int = 1729
    train_fraction: float = 0.65
    validation_fraction: float = 0.15
    test_fraction: float = 0.20
    temporal_holdout_cap: float = 0.25
    fixed_date: date | None = None


@dataclass(slots=True)
class SplitManifest:
    """Deterministic split assignments and validation metadata."""

    cohort_class: str
    endpoint_id: str
    root_seed: int
    split_rule: str
    temporal_cutoff: date | None
    quantile_fallback_used: bool
    assignments: dict[str, str]
    patient_fingerprints: dict[str, str]
    counts: dict[str, object]
    cohort_hash: str
    protocol_hash: str
    simulation_only: bool = True
    split_status: str = "created"


def load_endpoint_registry() -> dict[str, EndpointProtocol]:
    """Load all configured endpoint protocols without enabling them implicitly."""
    config = load_config()["cohort"]
    version = str(config["endpoint_definition_version"])
    return {
        endpoint_id: EndpointProtocol.from_mapping(endpoint_id, mapping, version)
        for endpoint_id, mapping in cast(dict[str, dict[str, object]], config["endpoints"]).items()
    }


def add_years(value: date, years: int) -> date:
    """Add calendar years deterministically, handling February 29."""
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)
