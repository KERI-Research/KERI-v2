"""Deterministic rolling prediction index dates."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from metaboguard.cohort.eligibility import (
    _index_reason,
    assert_dataset_has_one_class,
    make_patient_index,
)
from metaboguard.cohort.protocol import EndpointProtocol, PatientIndex
from metaboguard.data.canonical import CanonicalDataset
from metaboguard.data.schema import ClinicalEvent, ConditionRecord, OutcomeRecord


def generate_rolling_index_dates(
    dataset: CanonicalDataset, endpoint: EndpointProtocol
) -> list[PatientIndex]:
    """Generate eligible rolling patient indexes at the configured interval."""
    assert_dataset_has_one_class(dataset)
    events_by_patient: dict[str, list[ClinicalEvent]] = defaultdict(list)
    for event in dataset.events:
        events_by_patient.setdefault(event.patient_id, []).append(event)
    conditions_by_patient: dict[str, list[ConditionRecord]] = defaultdict(list)
    for condition in dataset.conditions:
        conditions_by_patient.setdefault(condition.patient_id, []).append(condition)
    outcomes_by_patient: dict[str, list[OutcomeRecord]] = defaultdict(list)
    for outcome in dataset.outcomes:
        outcomes_by_patient[outcome.patient_id].append(outcome)
    records: list[PatientIndex] = []
    for patient in sorted(dataset.patients, key=lambda value: value.patient_id):
        patient_events = events_by_patient.get(patient.patient_id, [])
        patient_conditions = conditions_by_patient.get(patient.patient_id, [])
        patient_outcomes = outcomes_by_patient.get(patient.patient_id, [])
        activity_dates = (
            [event.event_date for event in patient_events]
            + [condition.onset_date for condition in patient_conditions]
            + [outcome.outcome_date for outcome in patient_outcomes]
            + ([patient.death_date] if patient.death_date is not None else [])
        )
        valid_dates = [
            value
            for value in activity_dates
            if patient.death_date is None or value <= patient.death_date
        ]
        final_date = max(valid_dates) if valid_dates else patient.death_date
        if final_date is None or not patient_events:
            continue
        first_candidate = min(event.event_date for event in patient_events) + timedelta(
            days=endpoint.minimum_history_days
        )
        endpoint_dates = [
            condition.onset_date
            for condition in patient_conditions
            if (
                (
                    endpoint.outcome_family == "diabetes"
                    and condition.category == "diabetes"
                    and condition.diabetes_type == endpoint.diabetes_type
                )
                or (
                    endpoint.outcome_family == "cancer"
                    and condition.category == "cancer"
                    and condition.cancer_site == endpoint.cancer_site
                )
            )
        ]
        endpoint_date = min(endpoint_dates) if endpoint_dates else None
        if endpoint_date is not None and endpoint_date < first_candidate:
            continue
        sequence = 0
        index_date = first_candidate
        while index_date <= final_date:
            if endpoint_date is not None and index_date >= endpoint_date:
                break
            history = [
                event for event in patient_events if event.event_date <= index_date
            ]
            reason = _index_reason(
                dataset,
                endpoint,
                patient.patient_id,
                index_date,
                patient=patient,
                endpoint_date=endpoint_date,
                history=history,
            )
            if reason is None:
                records.append(
                    make_patient_index(
                        dataset,
                        endpoint,
                        patient.patient_id,
                        index_date,
                        sequence,
                        f"rolling_{endpoint.rolling_index_interval_days}d",
                        history=history,
                    )
                )
                sequence += 1
            index_date += timedelta(days=endpoint.rolling_index_interval_days)
    return sorted(
        records,
        key=lambda record: (record.patient_id, record.index_date, record.endpoint_id),
    )
