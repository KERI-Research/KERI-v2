"""Fail-closed validation for immutable cohort indexes and labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from metaboguard.cohort.censoring import follow_up_end_date
from metaboguard.cohort.eligibility import endpoint_conditions
from metaboguard.cohort.protocol import ConstructedCohort, EndpointProtocol, add_years
from metaboguard.config import load_config
from metaboguard.data.canonical import CanonicalDataset
from metaboguard.data.manifests import CohortClassMismatchError, assert_same_cohort_class


@dataclass(frozen=True, slots=True)
class CohortCheck:
    name: str
    level: str
    passed: bool
    status: str
    warning_count: int
    offending_row_count: int
    example_ids: tuple[str, ...]
    message: str = ""


@dataclass(frozen=True, slots=True)
class CohortValidationReport:
    passed: bool
    checks: tuple[CohortCheck, ...]
    warnings: tuple[CohortCheck, ...] = ()
    simulation_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["checks"] = [asdict(check) for check in self.checks]
        result["warnings"] = [asdict(check) for check in self.warnings]
        return result


def _check(name: str, level: str, ids: list[str], message: str = "") -> CohortCheck:
    warning = level == "warning"
    return CohortCheck(
        name=name,
        level=level,
        passed=warning or not ids,
        status="warning" if warning and ids else ("failed" if ids else "passed"),
        warning_count=len(ids) if warning else 0,
        offending_row_count=len(ids),
        example_ids=tuple(sorted(set(ids))[:10]),
        message=message,
    )


def validate_constructed_cohort(
    cohort: ConstructedCohort,
    dataset: CanonicalDataset,
    endpoint: EndpointProtocol,
) -> CohortValidationReport:
    """Validate temporal, state, protocol, and class-isolation invariants."""
    checks: list[CohortCheck] = []
    patient_by_id = {patient.patient_id: patient for patient in dataset.patients}
    endpoint_dates = {
        patient_id: min(
            condition.onset_date
            for condition in endpoint_conditions(dataset, endpoint)
            if condition.patient_id == patient_id
        )
        for patient_id in {record.patient_id for record in cohort.patient_indexes}
        if any(
            condition.patient_id == patient_id
            for condition in endpoint_conditions(dataset, endpoint)
        )
    }
    history_ids = [
        record.patient_id
        for record in cohort.patient_indexes
        if any(value > record.index_date for value in record.preindex_event_dates)
    ]
    checks.append(_check("history_on_or_before_index", "error", history_ids))
    index_ids = []
    for record in cohort.patient_indexes:
        patient = patient_by_id.get(record.patient_id)
        if patient is None or record.index_date < patient.birth_date:
            index_ids.append(record.patient_id)
        elif patient.death_date is not None and record.index_date > patient.death_date:
            index_ids.append(record.patient_id)
        elif (
            record.patient_id in endpoint_dates
            and record.index_date >= endpoint_dates[record.patient_id]
        ):
            index_ids.append(record.patient_id)
    checks.append(_check("index_temporal_bounds", "error", index_ids))
    positive_ids = [
        label.patient_id
        for label in cohort.labels
        if label.label_state == "positive"
        and (label.event_date is None or label.event_date <= label.index_date)
    ]
    checks.append(_check("positive_event_after_index", "error", positive_ids))
    negative_ids: list[str] = []
    censored_ids: list[str] = []
    competing_ids: list[str] = []
    protocol_ids = [
        label.patient_id
        for label in cohort.labels
        if label.endpoint_definition_version != endpoint.definition_version
    ]
    for label in cohort.labels:
        horizon_end = add_years(label.index_date, label.horizon_years)
        follow_up = follow_up_end_date(dataset, label.patient_id)
        if label.label_state == "eligible_negative" and follow_up < horizon_end:
            negative_ids.append(label.patient_id)
        if label.label_state == "censored" and follow_up >= horizon_end:
            censored_ids.append(label.patient_id)
        if label.label_state == "competing_death" and (
            label.death_date is None or label.death_date > horizon_end
        ):
            competing_ids.append(label.patient_id)
    checks.append(_check("negative_follow_up", "error", negative_ids))
    checks.append(_check("censored_follow_up", "error", censored_ids))
    checks.append(_check("competing_death_timing", "error", competing_ids))
    checks.append(_check("endpoint_protocol_version", "error", protocol_ids))
    try:
        assert_same_cohort_class([record.cohort_class for record in cohort.patient_indexes])
        class_ids: list[str] = []
    except CohortClassMismatchError:
        class_ids = [record.patient_id for record in cohort.patient_indexes]
    checks.append(_check("single_cohort_class", "error", class_ids))
    checks.append(
        _check("models_not_created", "error", [endpoint.endpoint_id] if endpoint.enabled else [])
    )
    warnings: list[CohortCheck] = []
    research_config = load_config()["research"]
    minimum_events = int(research_config["minimum_events"])
    minimum_nonevents = int(research_config["minimum_nonevents"])
    for horizon in endpoint.horizon_years:
        horizon_labels = [label for label in cohort.labels if label.horizon_years == horizon]
        positives = sum(label.label_state == "positive" for label in horizon_labels)
        negatives = sum(label.label_state == "eligible_negative" for label in horizon_labels)
        warnings.append(
            _check(
                f"horizon_{horizon}y_minimum_events",
                "warning",
                [endpoint.endpoint_id] if positives < minimum_events else [],
                f"positive_count={positives}; required={minimum_events}",
            )
        )
        warnings.append(
            _check(
                f"horizon_{horizon}y_minimum_nonevents",
                "warning",
                [endpoint.endpoint_id] if negatives < minimum_nonevents else [],
                f"eligible_negative_count={negatives}; required={minimum_nonevents}",
            )
        )
    return CohortValidationReport(
        passed=all(check.passed for check in checks if check.level == "error"),
        checks=tuple(checks),
        warnings=tuple(warnings),
    )
