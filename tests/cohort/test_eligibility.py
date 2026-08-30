from dataclasses import replace
from datetime import date

import pytest

from metaboguard.cohort.eligibility import (
    _age_on,
    _patient_final_date,
    build_eligible_patient_indexes,
    classify_index_candidate,
    endpoint_conditions,
)
from metaboguard.data.canonical import CanonicalDataset
from metaboguard.data.manifests import CohortClassMismatchError
from metaboguard.data.schema import Patient


def test_eligibility_excludes_prevalent_patient(cohort_dataset, diabetes_endpoint) -> None:
    indexes = build_eligible_patient_indexes(cohort_dataset, diabetes_endpoint)
    assert indexes
    assert all(index.patient_id != "p-prevalent" for index in indexes)
    assert endpoint_conditions(cohort_dataset, diabetes_endpoint)


def test_missing_cohort_class_fails_closed(cohort_dataset, diabetes_endpoint) -> None:
    missing = CanonicalDataset(
        cohort_dataset.patients,
        cohort_dataset.events,
        cohort_dataset.conditions,
        cohort_dataset.outcomes,
    )
    with pytest.raises(CohortClassMismatchError):
        build_eligible_patient_indexes(missing, diabetes_endpoint)


def test_candidate_reasons(cohort_dataset, diabetes_endpoint) -> None:
    assert _age_on(date(1980, 6, 1), date(1980, 1, 1)) == -1
    assert (
        classify_index_candidate(cohort_dataset, diabetes_endpoint, "p-positive", date(1980, 1, 1))
        == "under_minimum_age"
    )
    assert (
        classify_index_candidate(cohort_dataset, diabetes_endpoint, "p-positive", date(2015, 1, 1))
        == "insufficient_history"
    )


def test_patient_final_date_is_bounded_by_death(cohort_dataset) -> None:
    with_post_death_activity = replace(
        cohort_dataset,
        events=[
            *cohort_dataset.events,
            cohort_dataset.events[0].model_copy(
                update={"patient_id": "p-competing", "event_date": date(2020, 1, 1)}
            ),
        ],
    )
    death_only = CanonicalDataset(
        [
            Patient(
                patient_id="p-death-only",
                birth_date=date(1980, 1, 1),
                sex="male",
                ethnicity="x",
                death_date=date(2022, 1, 1),
            )
        ],
        [],
        [],
        [],
    )
    no_activity = CanonicalDataset(
        [
            Patient(
                patient_id="p-no-activity",
                birth_date=date(1980, 1, 1),
                sex="male",
                ethnicity="x",
            )
        ],
        [],
        [],
        [],
    )

    assert _patient_final_date(with_post_death_activity, "p-competing") == date(2018, 6, 1)
    assert _patient_final_date(death_only, "p-death-only") == date(2022, 1, 1)
    assert _patient_final_date(no_activity, "p-no-activity") is None


def test_candidate_exclusion_reasons(
    cohort_dataset, diabetes_endpoint, pancreatic_endpoint
) -> None:
    assert (
        classify_index_candidate(
            cohort_dataset, diabetes_endpoint, "p-positive", date(1980, 12, 31)
        )
        == "under_minimum_age"
    )
    assert (
        classify_index_candidate(cohort_dataset, diabetes_endpoint, "p-positive", date(2020, 1, 1))
        == "prevalent_endpoint"
    )
    assert (
        classify_index_candidate(cohort_dataset, diabetes_endpoint, "p-competing", date(2019, 1, 1))
        == "index_after_death"
    )
    no_prevalent = replace(diabetes_endpoint, prevalent_exclusion=False)
    assert (
        classify_index_candidate(cohort_dataset, no_prevalent, "p-positive", date(2020, 1, 1))
        == "index_after_endpoint_onset"
    )
    assert (
        classify_index_candidate(
            cohort_dataset, pancreatic_endpoint, "p-positive", date(2019, 1, 1)
        )
        == "excluded_pancreatic_cancer_washout"
    )
    sparse = replace(
        cohort_dataset,
        events=[event for event in cohort_dataset.events if event.patient_id != "p-negative"],
    )
    assert (
        classify_index_candidate(sparse, diabetes_endpoint, "p-negative", date(2016, 1, 1))
        == "insufficient_history"
    )
    one_measurement = replace(
        cohort_dataset,
        events=[event for event in cohort_dataset.events if event.patient_id != "p-negative"]
        + [
            cohort_dataset.events[0].model_copy(
                update={"patient_id": "p-negative", "event_date": date(2015, 1, 1)}
            ),
            cohort_dataset.events[1].model_copy(
                update={
                    "patient_id": "p-negative",
                    "event_date": date(2016, 1, 1),
                    "value": None,
                    "is_missing": True,
                }
            ),
        ],
    )
    assert (
        classify_index_candidate(one_measurement, diabetes_endpoint, "p-negative", date(2016, 1, 1))
        == "insufficient_preindex_measurements"
    )
    few_encounters = replace(diabetes_endpoint, minimum_preindex_encounters=3)
    assert (
        classify_index_candidate(cohort_dataset, few_encounters, "p-positive", date(2016, 1, 1))
        == "insufficient_preindex_encounters"
    )
