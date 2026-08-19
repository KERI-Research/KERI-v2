"""Endpoint-specific patient-index eligibility rules."""

from __future__ import annotations

from datetime import date

from metaboguard.cohort.protocol import EndpointProtocol, PatientIndex
from metaboguard.data.canonical import CanonicalDataset
from metaboguard.data.manifests import CohortClassMismatchError, assert_same_cohort_class
from metaboguard.data.schema import ClinicalEvent, ConditionRecord, Patient


def build_eligible_patient_indexes(
    dataset: CanonicalDataset, endpoint: EndpointProtocol
) -> list[PatientIndex]:
    """Build eligible rolling patient-index records for one endpoint."""
    from metaboguard.cohort.index_dates import generate_rolling_index_dates

    return generate_rolling_index_dates(dataset, endpoint)


def _condition_matches(condition: ConditionRecord, endpoint: EndpointProtocol) -> bool:
    if endpoint.outcome_family == "diabetes":
        return (
            condition.category == "diabetes" and condition.diabetes_type == endpoint.diabetes_type
        )
    return condition.category == "cancer" and condition.cancer_site == endpoint.cancer_site


def endpoint_conditions(
    dataset: CanonicalDataset, endpoint: EndpointProtocol
) -> list[ConditionRecord]:
    """Return canonical conditions belonging to this endpoint only."""
    return [
        condition for condition in dataset.conditions if _condition_matches(condition, endpoint)
    ]


def _first_endpoint_date(
    dataset: CanonicalDataset, endpoint: EndpointProtocol, patient_id: str
) -> date | None:
    dates = [
        condition.onset_date
        for condition in endpoint_conditions(dataset, endpoint)
        if condition.patient_id == patient_id
    ]
    return min(dates) if dates else None


def _patient_final_date(dataset: CanonicalDataset, patient_id: str) -> date | None:
    """Return the last valid dated activity, bounded by death."""
    patient = next(patient for patient in dataset.patients if patient.patient_id == patient_id)
    dates = (
        [event.event_date for event in dataset.events if event.patient_id == patient_id]
        + [
            condition.onset_date
            for condition in dataset.conditions
            if condition.patient_id == patient_id
        ]
        + [outcome.outcome_date for outcome in dataset.outcomes if outcome.patient_id == patient_id]
    )
    if patient.death_date is not None:
        dates.append(patient.death_date)
    valid = [value for value in dates if patient.death_date is None or value <= patient.death_date]
    return max(valid) if valid else patient.death_date


def _age_on(patient_birth_date: date, value: date) -> int:
    years = value.year - patient_birth_date.year
    if (value.month, value.day) < (patient_birth_date.month, patient_birth_date.day):
        years -= 1
    return years


def _index_reason(
    dataset: CanonicalDataset,
    endpoint: EndpointProtocol,
    patient_id: str,
    index_date: date,
    *,
    patient: Patient | None = None,
    endpoint_date: date | None = None,
    history: list[ClinicalEvent] | None = None,
) -> str | None:
    patient = patient or next(
        patient for patient in dataset.patients if patient.patient_id == patient_id
    )
    if _age_on(patient.birth_date, index_date) < endpoint.minimum_age_years:
        return "under_minimum_age"
    if patient.death_date is not None and index_date > patient.death_date:
        return "index_after_death"
    endpoint_date = (
        _first_endpoint_date(dataset, endpoint, patient_id)
        if endpoint_date is None
        else endpoint_date
    )
    if endpoint.prevalent_exclusion and endpoint_date is not None and endpoint_date <= index_date:
        return "prevalent_endpoint"
    if endpoint_date is not None and endpoint_date <= index_date:
        return "index_after_endpoint_onset"
    if endpoint.washout_days and endpoint_date is not None:
        if endpoint_date <= index_date.fromordinal(index_date.toordinal() + endpoint.washout_days):
            return f"excluded_{endpoint.endpoint_id}_washout"
    history = history or [
        event
        for event in dataset.events
        if event.patient_id == patient_id and event.event_date <= index_date
    ]
    if not history:
        return "insufficient_history"
    first_date = min(event.event_date for event in history)
    if (index_date - first_date).days < endpoint.minimum_history_days:
        return "insufficient_history"
    if len({event.event_date for event in history}) < endpoint.minimum_preindex_encounters:
        return "insufficient_preindex_encounters"
    measurement_dates = {event.event_date for event in history if event.value is not None}
    if len(measurement_dates) < endpoint.minimum_preindex_measurements:
        return "insufficient_preindex_measurements"
    return None


def make_patient_index(
    dataset: CanonicalDataset,
    endpoint: EndpointProtocol,
    patient_id: str,
    index_date: date,
    sequence_number: int,
    index_source: str,
    *,
    history: list[ClinicalEvent] | None = None,
) -> PatientIndex:
    """Build an index record and retain only history at or before the cutoff."""
    history = history or [
        event
        for event in dataset.events
        if event.patient_id == patient_id and event.event_date <= index_date
    ]
    return PatientIndex(
        patient_id=patient_id,
        cohort_class=dataset.cohort_class or "",
        endpoint_id=endpoint.endpoint_id,
        index_date=index_date,
        index_source=index_source,
        index_sequence_number=sequence_number,
        preindex_event_dates=tuple(sorted(event.event_date for event in history)),
        preindex_measurement_dates=tuple(
            sorted({event.event_date for event in history if event.value is not None})
        ),
        preindex_encounter_count=len({event.event_date for event in history}),
    )


def classify_index_candidate(
    dataset: CanonicalDataset,
    endpoint: EndpointProtocol,
    patient_id: str,
    index_date: date,
) -> str | None:
    """Return the deterministic exclusion reason for a candidate date."""
    return _index_reason(dataset, endpoint, patient_id, index_date)


def assert_dataset_has_one_class(dataset: CanonicalDataset) -> str:
    """Require an attached cohort class before constructing indexes."""
    if not dataset.cohort_class:
        raise CohortClassMismatchError("Canonical dataset has no cohort class")
    return assert_same_cohort_class([dataset.cohort_class])
