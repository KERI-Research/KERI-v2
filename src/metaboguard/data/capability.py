"""Capability reporting for simulation cohorts without model fitting."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from metaboguard.data.canonical import CanonicalDataset
from metaboguard.data.manifests import CohortClass


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    """Counts and quality state for a generated simulation cohort."""

    cohort_class: CohortClass
    state: str
    simulation_only: bool
    patient_count: int
    adult_count: int
    diabetes_type_counts: dict[str, int]
    incident_cancer_count_by_site: dict[str, int]
    diabetes_onset_distribution: dict[str, float | int | None]
    repeated_measurements: dict[str, int]
    encounter_count_distribution: dict[str, float | int]
    horizon_eligible_event_counts: dict[str, int]
    deaths_before_horizon: dict[str, int]
    feature_missingness: dict[str, float]
    augmentation_provenance_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _adult(patient_birth_date: date, as_of: date) -> bool:
    return (as_of - patient_birth_date).days >= 18 * 365


def _iqr(values: list[int]) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = (
        float(ordered[midpoint])
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    lower = ordered[max(0, int(len(ordered) * 0.25) - 1)]
    upper = ordered[min(len(ordered) - 1, int(len(ordered) * 0.75))]
    return median, float(upper - lower), float(upper)


def build_capability_report(
    dataset: CanonicalDataset,
    cohort_class: CohortClass,
    horizons_years: tuple[int, ...] = (1, 3, 5),
    minimum_event_count: int = 1,
) -> CapabilityReport:
    """Calculate software-readiness counts without fitting or evaluating a model."""
    as_of = date.today()
    patient_by_id = {patient.patient_id: patient for patient in dataset.patients}
    diabetes_types = Counter({"type1": 0, "type2": 0, "gestational": 0, "none": 0})
    diabetes_onsets: list[int] = []
    for condition in dataset.conditions:
        if condition.category == "diabetes":
            diabetes_types[condition.diabetes_type or "none"] += 1
            patient = patient_by_id.get(condition.patient_id)
            if patient is not None:
                diabetes_onsets.append((condition.onset_date - patient.birth_date).days // 365)
    for patient_id in patient_by_id:
        if not any(
            condition.patient_id == patient_id and condition.category == "diabetes"
            for condition in dataset.conditions
        ):
            diabetes_types["none"] += 1
    cancer_counts = Counter(
        condition.cancer_site or "unspecified"
        for condition in dataset.conditions
        if condition.category == "cancer"
    )
    by_feature_patient: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_feature_totals: Counter[str] = Counter()
    for event in dataset.events:
        by_feature_totals[event.feature_name] += 1
        by_feature_patient[event.feature_name][event.patient_id] += 1
    repeated = {
        feature: sum(count >= 2 for count in patients.values())
        for feature, patients in by_feature_patient.items()
    }
    encounter_counts = Counter(event.patient_id for event in dataset.events)
    median, iqr, upper = _iqr(list(encounter_counts.values()))
    eligible: dict[str, int] = {}
    deaths: dict[str, int] = {}
    for horizon in horizons_years:
        eligible_count = 0
        for condition in dataset.conditions:
            patient = patient_by_id.get(condition.patient_id)
            if condition.category != "cancer" or patient is None:
                continue
            if patient.death_date is None:
                eligible_count += 1
                continue
            onset_age_days = (condition.onset_date - patient.birth_date).days
            death_age_days = (patient.death_date - patient.birth_date).days
            eligible_count += onset_age_days + horizon * 365 <= death_age_days
        eligible[str(horizon)] = eligible_count
        deaths[str(horizon)] = sum(
            patient.death_date is not None and (as_of - patient.death_date).days >= horizon * 365
            for patient in dataset.patients
        )
    missingness = {
        feature: sum(event.is_missing for event in dataset.events if event.feature_name == feature)
        / total
        for feature, total in by_feature_totals.items()
    }
    augmentation_counts = Counter(event.augmentation_module or "native" for event in dataset.events)
    state = (
        "ready_for_cohort_construction"
        if sum(cancer_counts.values()) >= minimum_event_count
        else "insufficient_event_count"
    )
    return CapabilityReport(
        cohort_class=cohort_class,
        state=state,
        simulation_only=True,
        patient_count=len(dataset.patients),
        adult_count=sum(_adult(patient.birth_date, as_of) for patient in dataset.patients),
        diabetes_type_counts=dict(sorted(diabetes_types.items())),
        incident_cancer_count_by_site=dict(sorted(cancer_counts.items())),
        diabetes_onset_distribution={
            "count": len(diabetes_onsets),
            "median_years": (
                sorted(diabetes_onsets)[len(diabetes_onsets) // 2] if diabetes_onsets else None
            ),
        },
        repeated_measurements=repeated,
        encounter_count_distribution={
            "median": median,
            "iqr": iqr,
            "upper_quartile": upper,
        },
        horizon_eligible_event_counts=eligible,
        deaths_before_horizon=deaths,
        feature_missingness=missingness,
        augmentation_provenance_counts=dict(sorted(augmentation_counts.items())),
    )
