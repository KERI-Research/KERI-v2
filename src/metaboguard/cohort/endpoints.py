"""Endpoint-specific incident labels and independent horizon states."""

from __future__ import annotations

from typing import cast

from metaboguard.cohort.censoring import death_before, follow_up_end_date
from metaboguard.cohort.eligibility import endpoint_conditions
from metaboguard.cohort.protocol import (
    EndpointProtocol,
    HorizonLabel,
    LabelState,
    PatientIndex,
    add_years,
)
from metaboguard.data.canonical import CanonicalDataset
from metaboguard.data.schema import ConditionRecord


def _first_future_endpoint(
    dataset: CanonicalDataset, patient_index: PatientIndex, endpoint: EndpointProtocol
) -> ConditionRecord | None:
    candidates = [
        condition
        for condition in endpoint_conditions(dataset, endpoint)
        if condition.patient_id == patient_index.patient_id
        and condition.onset_date > patient_index.index_date
    ]
    return min(candidates, key=lambda condition: condition.onset_date) if candidates else None


def _label_one_horizon(
    patient_index: PatientIndex,
    dataset: CanonicalDataset,
    endpoint: EndpointProtocol,
    horizon_years: int,
) -> HorizonLabel:
    horizon_end = add_years(patient_index.index_date, horizon_years)
    patient = next(
        value for value in dataset.patients if value.patient_id == patient_index.patient_id
    )
    endpoint_condition = _first_future_endpoint(dataset, patient_index, endpoint)
    endpoint_date = endpoint_condition.onset_date if endpoint_condition else None
    follow_up = follow_up_end_date(dataset, patient_index.patient_id)
    death_date = patient.death_date
    if endpoint_date is not None and endpoint_date <= horizon_end:
        if death_before(death_date, endpoint_date, endpoint.use_competing_death_risk):
            state = "competing_death"
            event_date = None
            source_code = None
            censor_date = death_date
        else:
            state = "positive"
            event_date = endpoint_date
            source_code = endpoint_condition.condition_code if endpoint_condition else None
            censor_date = None
    elif death_before(death_date, horizon_end, endpoint.use_competing_death_risk):
        state = "competing_death"
        event_date = None
        source_code = None
        censor_date = death_date
    elif follow_up >= horizon_end:
        state = "eligible_negative"
        event_date = None
        source_code = None
        censor_date = None
    else:
        state = "censored"
        event_date = None
        source_code = None
        censor_date = follow_up
    terminal_date = event_date or censor_date
    return HorizonLabel(
        patient_id=patient_index.patient_id,
        cohort_class=patient_index.cohort_class,
        endpoint_id=endpoint.endpoint_id,
        index_date=patient_index.index_date,
        horizon_years=horizon_years,
        label_state=cast(LabelState, state),
        event_date=event_date,
        death_date=death_date,
        censor_date=censor_date,
        days_to_event_or_censor=(
            (terminal_date - patient_index.index_date).days if terminal_date is not None else None
        ),
        endpoint_definition_version=endpoint.definition_version,
        outcome_source_condition_code=source_code,
    )


def assign_outcomes(
    patient_indexes: list[PatientIndex],
    dataset: CanonicalDataset,
    endpoint: EndpointProtocol,
) -> list[HorizonLabel]:
    """Assign independent positive, negative, censored, or competing labels."""
    labels = [
        _label_one_horizon(patient_index, dataset, endpoint, horizon)
        for patient_index in patient_indexes
        for horizon in endpoint.horizon_years
    ]
    return sorted(
        labels,
        key=lambda label: (label.patient_id, label.index_date, label.horizon_years),
    )
