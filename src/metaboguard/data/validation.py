"""Strict validation of canonical longitudinal datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

from metaboguard.config import load_config
from metaboguard.data.canonical import CanonicalDataset
from metaboguard.data.schema import SCHEMA_VERSION
from metaboguard.features.dictionary import (
    FEATURE_DICTIONARY,
    DeniedFeatureError,
    get_feature_definition,
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of one validation rule."""

    name: str
    level: str
    passed: bool
    status: str
    warning_count: int
    offending_row_count: int
    example_ids: tuple[str, ...]
    message: str = ""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Serializable validation outcome."""

    passed: bool
    dataset_sha256: str
    schema_version: str
    row_counts: dict[str, int]
    checks: tuple[CheckResult, ...]
    dropped_loinc_codes: dict[str, int]
    report_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["checks"] = [asdict(check) for check in self.checks]
        result["report_path"] = str(self.report_path) if self.report_path else None
        return result


def _config() -> dict[str, Any]:
    return cast(dict[str, Any], load_config()["validation"])


def _examples(ids: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(ids))[:10])


def _result(name: str, level: str, ids: list[str], message: str = "") -> CheckResult:
    warning = level == "warning"
    return CheckResult(
        name=name,
        level=level,
        passed=warning or not ids,
        status="warning" if warning and ids else ("failed" if ids else "passed"),
        warning_count=len(ids) if warning else 0,
        offending_row_count=len(ids),
        example_ids=_examples(ids),
        message=message,
    )


def _dataset_hash(dataset: CanonicalDataset) -> str:
    if dataset.dataset_sha256:
        return dataset.dataset_sha256
    payload = json.dumps(
        {
            "patients": [
                patient.model_dump(mode="json") for patient in dataset.patients
            ],
            "events": [event.model_dump(mode="json") for event in dataset.events],
            "conditions": [
                condition.model_dump(mode="json") for condition in dataset.conditions
            ],
            "outcomes": [
                outcome.model_dump(mode="json") for outcome in dataset.outcomes
            ],
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate(dataset: CanonicalDataset) -> ValidationReport:
    """Run all error and warning checks and write the validation report."""
    settings = _config()
    patient_ids = {patient.patient_id for patient in dataset.patients}
    patient_by_id = {patient.patient_id: patient for patient in dataset.patients}
    today = date.today()
    checks: list[CheckResult] = []
    checks.append(
        _result(
            "referential_integrity",
            "error",
            [
                item.patient_id
                for item in dataset.events
                if item.patient_id not in patient_ids
            ]
            + [
                item.patient_id
                for item in dataset.conditions
                if item.patient_id not in patient_ids
            ],
        )
    )
    chronology_ids = [
        event.patient_id
        for event in dataset.events
        if event.patient_id in patient_by_id
        and (
            event.event_date < patient_by_id[event.patient_id].birth_date
            or event.event_date > (patient_by_id[event.patient_id].death_date or today)
        )
    ]
    chronology_ids.extend(
        condition.patient_id
        for condition in dataset.conditions
        if condition.patient_id in patient_by_id
        and (
            condition.onset_date < patient_by_id[condition.patient_id].birth_date
            or condition.onset_date
            > (patient_by_id[condition.patient_id].death_date or today)
        )
    )
    checks.append(_result("chronology", "error", chronology_ids))
    after_death_ids: list[str] = []
    for event in dataset.events:
        patient = patient_by_id.get(event.patient_id)
        if (
            patient is not None
            and patient.death_date is not None
            and event.event_date > patient.death_date
        ):
            after_death_ids.append(event.patient_id)
    checks.append(_result("no_events_after_death", "error", after_death_ids))
    age_ids = []
    for event in dataset.events:
        patient = patient_by_id.get(event.patient_id)
        if patient is None:
            continue
        expected = (event.event_date - patient.birth_date).days / 365.2425
        if (
            abs(event.age_at_event - expected)
            > settings["age_tolerance_days"] / 365.2425
        ):
            age_ids.append(event.patient_id)
    checks.append(_result("age_at_event", "error", age_ids))
    range_ids = []
    for event in dataset.events:
        if event.is_missing or event.value is None:
            continue
        definition = FEATURE_DICTIONARY.get(event.feature_name)
        if (
            definition is None
            or not definition.plausible_min <= event.value <= definition.plausible_max
        ):
            range_ids.append(event.patient_id)
    checks.append(_result("plausible_ranges", "error", range_ids))
    missing_ids = [
        event.patient_id
        for event in dataset.events
        if event.is_missing != (event.value is None)
    ]
    checks.append(_result("missingness_consistency", "error", missing_ids))
    unit_ids = [
        event.patient_id
        for event in dataset.events
        if event.feature_name not in FEATURE_DICTIONARY
        or event.unit != FEATURE_DICTIONARY[event.feature_name].canonical_unit
    ]
    checks.append(_result("canonical_units", "error", unit_ids))
    denied_ids = []
    for event in dataset.events:
        try:
            get_feature_definition(event.feature_name)
        except DeniedFeatureError:
            denied_ids.append(event.patient_id)
    checks.append(_result("denylist", "error", denied_ids))
    module_ids = [
        event.patient_id
        for event in dataset.events
        if event.provenance == "augmented"
        and event.augmentation_module not in {"ca19_9", "c_peptide", "insulin"}
    ]
    checks.append(_result("augmentation_module", "error", module_ids))
    diabetes_onset_ids = [
        condition.patient_id
        for condition in dataset.conditions
        if condition.category == "diabetes"
        and condition.patient_id in patient_by_id
        and condition.onset_date < patient_by_id[condition.patient_id].birth_date
    ]
    checks.append(_result("diabetes_onset_after_birth", "error", diabetes_onset_ids))
    checks.append(
        _result(
            "cancer_site",
            "error",
            [
                condition.patient_id
                for condition in dataset.conditions
                if condition.category == "cancer" and condition.cancer_site is None
            ],
        )
    )
    patient_id_counts = Counter(patient.patient_id for patient in dataset.patients)
    duplicate_ids = [
        patient_id for patient_id, count in patient_id_counts.items() if count > 1
    ]
    checks.append(_result("unique_patient_ids", "error", duplicate_ids))

    encounter_counts: dict[str, int] = {}
    for event in dataset.events:
        encounter_counts[event.patient_id] = (
            encounter_counts.get(event.patient_id, 0) + 1
        )
    checks.append(
        _result(
            "minimum_encounters",
            "warning",
            [
                patient.patient_id
                for patient in dataset.patients
                if encounter_counts.get(patient.patient_id, 0)
                < settings["minimum_encounters"]
            ],
        )
    )
    by_feature: dict[str, list[bool]] = {}
    for event in dataset.events:
        by_feature.setdefault(event.feature_name, []).append(event.is_missing)
    checks.append(
        _result(
            "missingness_rate",
            "warning",
            [
                name
                for name, values in by_feature.items()
                if sum(values) / len(values) > settings["maximum_missingness"]
            ],
        )
    )
    checks.append(_result("population_range_review", "warning", []))

    errors_failed = any(check.level == "error" and not check.passed for check in checks)
    report = ValidationReport(
        passed=not errors_failed,
        dataset_sha256=_dataset_hash(dataset),
        schema_version=SCHEMA_VERSION,
        row_counts={
            "patients": len(dataset.patients),
            "events": len(dataset.events),
            "conditions": len(dataset.conditions),
            "outcomes": len(dataset.outcomes),
        },
        checks=tuple(checks),
        dropped_loinc_codes=dataset.dropped_loinc_codes,
    )
    report_path = Path(load_config()["paths"]["validation_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {**report.to_dict(), "report_path": report_path.as_posix()},
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return ValidationReport(
        passed=report.passed,
        dataset_sha256=report.dataset_sha256,
        schema_version=report.schema_version,
        row_counts=report.row_counts,
        checks=report.checks,
        dropped_loinc_codes=report.dropped_loinc_codes,
        report_path=report_path,
    )
