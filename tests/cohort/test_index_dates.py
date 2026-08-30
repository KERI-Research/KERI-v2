from dataclasses import replace
from datetime import date

from metaboguard.cohort.index_dates import generate_rolling_index_dates


def test_rolling_indexes_are_deterministic_and_preindex(cohort_dataset, diabetes_endpoint) -> None:
    first = generate_rolling_index_dates(cohort_dataset, diabetes_endpoint)
    second = generate_rolling_index_dates(cohort_dataset, diabetes_endpoint)
    assert first == second
    assert first
    assert all(max(index.preindex_event_dates) <= index.index_date for index in first)
    assert all(index.index_source == "rolling_365d" for index in first)
    assert all(
        index.index_date != date(2019, 1, 1) for index in first if index.patient_id == "p-positive"
    )


def test_pancreatic_endpoint_stops_at_onset(cohort_dataset, pancreatic_endpoint) -> None:
    from metaboguard.cohort.index_dates import generate_rolling_index_dates

    indexes = generate_rolling_index_dates(cohort_dataset, pancreatic_endpoint)
    assert all(
        index.index_date < date(2020, 1, 1) for index in indexes if index.patient_id == "p-positive"
    )


def test_indexes_skip_patients_without_events_and_stop_after_death(
    cohort_dataset, diabetes_endpoint
) -> None:
    from metaboguard.data.schema import Patient

    empty_patient = Patient(
        patient_id="p-empty", birth_date=date(1980, 1, 1), sex="male", ethnicity="x"
    )
    changed = replace(cohort_dataset, patients=[*cohort_dataset.patients, empty_patient])
    indexes = generate_rolling_index_dates(changed, diabetes_endpoint)
    assert all(index.patient_id != "p-empty" for index in indexes)
    assert all(
        index.index_date <= date(2018, 6, 1)
        for index in indexes
        if index.patient_id == "p-competing"
    )
    early_death = replace(
        cohort_dataset,
        patients=[
            (
                patient.model_copy(update={"death_date": date(2016, 6, 1)})
                if patient.patient_id == "p-censored"
                else patient
            )
            for patient in cohort_dataset.patients
        ],
    )
    generate_rolling_index_dates(early_death, diabetes_endpoint)
    without_endpoint = replace(
        cohort_dataset,
        conditions=[
            condition
            for condition in cohort_dataset.conditions
            if condition.patient_id != "p-competing"
        ],
    )
    indexes_without_endpoint = generate_rolling_index_dates(without_endpoint, diabetes_endpoint)
    assert all(
        index.index_date <= date(2018, 6, 1)
        for index in indexes_without_endpoint
        if index.patient_id == "p-competing"
    )
