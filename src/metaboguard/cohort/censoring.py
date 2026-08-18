"""Deterministic patient follow-up and competing-death calculations."""

from __future__ import annotations

from datetime import date

from metaboguard.data.canonical import CanonicalDataset


def follow_up_end_date(dataset: CanonicalDataset, patient_id: str) -> date:
    """Return maximum valid observed activity, never extending beyond death."""
    patient = next(patient for patient in dataset.patients if patient.patient_id == patient_id)
    dates = [event.event_date for event in dataset.events if event.patient_id == patient_id]
    dates.extend(
        condition.onset_date
        for condition in dataset.conditions
        if condition.patient_id == patient_id
    )
    dates.extend(
        outcome.outcome_date for outcome in dataset.outcomes if outcome.patient_id == patient_id
    )
    if patient.death_date is not None:
        dates.append(patient.death_date)
    valid_dates = [
        value for value in dates if patient.death_date is None or value <= patient.death_date
    ]
    return max(valid_dates) if valid_dates else patient.birth_date


def death_before(death_date: date | None, boundary: date, use_competing_death_risk: bool) -> bool:
    """Return whether death precedes a future endpoint boundary."""
    return use_competing_death_risk and death_date is not None and death_date <= boundary
