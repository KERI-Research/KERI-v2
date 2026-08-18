"""Deterministic rolling prediction index dates."""

from __future__ import annotations

from datetime import timedelta

from metaboguard.cohort.eligibility import (
    _patient_final_date,
    assert_dataset_has_one_class,
    classify_index_candidate,
    make_patient_index,
)
from metaboguard.cohort.protocol import EndpointProtocol, PatientIndex
from metaboguard.data.canonical import CanonicalDataset


def generate_rolling_index_dates(
    dataset: CanonicalDataset, endpoint: EndpointProtocol
) -> list[PatientIndex]:
    """Generate eligible rolling patient indexes at the configured interval."""
    assert_dataset_has_one_class(dataset)
    records: list[PatientIndex] = []
    for patient in sorted(dataset.patients, key=lambda value: value.patient_id):
        final_date = _patient_final_date(dataset, patient.patient_id)
        patient_events = [
            event for event in dataset.events if event.patient_id == patient.patient_id
        ]
        if final_date is None or not patient_events:
            continue
        first_candidate = min(event.event_date for event in patient_events) + timedelta(
            days=endpoint.minimum_history_days
        )
        endpoint_dates = [
            condition.onset_date
            for condition in dataset.conditions
            if condition.patient_id == patient.patient_id
            and (
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
            reason = classify_index_candidate(dataset, endpoint, patient.patient_id, index_date)
            if reason is None:
                records.append(
                    make_patient_index(
                        dataset,
                        endpoint,
                        patient.patient_id,
                        index_date,
                        sequence,
                        f"rolling_{endpoint.rolling_index_interval_days}d",
                    )
                )
                sequence += 1
            index_date += timedelta(days=endpoint.rolling_index_interval_days)
    return sorted(
        records,
        key=lambda record: (record.patient_id, record.index_date, record.endpoint_id),
    )
