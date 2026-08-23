from dataclasses import replace
from datetime import date

from metaboguard.cohort.endpoints import assign_outcomes
from metaboguard.cohort.protocol import ConstructedCohort, HorizonLabel, PatientIndex
from metaboguard.cohort.validation import validate_constructed_cohort


def _index(patient_id: str, index_date: date = date(2016, 1, 1)) -> PatientIndex:
    return PatientIndex(
        patient_id,
        "ordinary_incidence",
        "type2_diabetes",
        index_date,
        "test",
        0,
        (date(2015, 1, 1), index_date),
        (date(2015, 1, 1), index_date),
        2,
    )


def test_constructed_validation_passes(cohort_dataset, diabetes_endpoint) -> None:
    indexes = [_index("p-positive")]
    labels = assign_outcomes(indexes, cohort_dataset, diabetes_endpoint)
    cohort = ConstructedCohort(
        "ordinary_incidence", diabetes_endpoint, indexes, [], labels, "hash"
    )
    report = validate_constructed_cohort(cohort, cohort_dataset, diabetes_endpoint)
    assert report.passed
    assert report.warnings
    assert all(warning.passed is True for warning in report.warnings)
    assert all(warning.status == "warning" for warning in report.warnings)
    assert all(warning.warning_count == 1 for warning in report.warnings)


def test_constructed_validation_catches_future_history(
    cohort_dataset, diabetes_endpoint
) -> None:
    bad = replace(_index("p-positive"), preindex_event_dates=(date(2020, 1, 1),))
    cohort = ConstructedCohort(
        "ordinary_incidence", diabetes_endpoint, [bad], [], [], "hash"
    )
    report = validate_constructed_cohort(cohort, cohort_dataset, diabetes_endpoint)
    assert not report.passed
    assert (
        next(
            check
            for check in report.checks
            if check.name == "history_on_or_before_index"
        ).offending_row_count
        == 1
    )


def test_enabled_endpoint_is_rejected(cohort_dataset, diabetes_endpoint) -> None:
    enabled = replace(diabetes_endpoint, enabled=True)
    cohort = ConstructedCohort("ordinary_incidence", enabled, [], [], [], "hash")
    report = validate_constructed_cohort(cohort, cohort_dataset, enabled)
    assert not report.passed


def test_validation_catches_temporal_and_label_state_errors(
    cohort_dataset, diabetes_endpoint
) -> None:
    records = [
        _index("missing"),
        replace(_index("p-positive"), index_date=date(1970, 1, 1)),
        replace(_index("p-competing"), index_date=date(2019, 1, 1)),
        replace(_index("p-positive"), index_date=date(2020, 1, 1)),
        replace(_index("p-positive"), preindex_event_dates=(date(2020, 1, 1),)),
    ]
    bad_labels = [
        HorizonLabel(
            "p-positive",
            "ordinary_incidence",
            "type2_diabetes",
            date(2016, 1, 1),
            1,
            "positive",
            date(2015, 1, 1),
            None,
            None,
            -1,
            "1.0.0",
        ),
        HorizonLabel(
            "p-censored",
            "ordinary_incidence",
            "type2_diabetes",
            date(2016, 1, 1),
            5,
            "eligible_negative",
            None,
            None,
            None,
            None,
            "1.0.0",
        ),
        HorizonLabel(
            "p-negative",
            "ordinary_incidence",
            "type2_diabetes",
            date(2016, 1, 1),
            1,
            "censored",
            None,
            None,
            date(2020, 1, 1),
            100,
            "1.0.0",
        ),
        HorizonLabel(
            "p-competing",
            "ordinary_incidence",
            "type2_diabetes",
            date(2016, 1, 1),
            1,
            "competing_death",
            None,
            date(2022, 1, 1),
            date(2022, 1, 1),
            1,
            "wrong",
        ),
    ]
    mixed = replace(_index("p-positive"), cohort_class="enriched_pancreatic")
    cohort = ConstructedCohort(
        "ordinary_incidence",
        diabetes_endpoint,
        [*records, mixed],
        [],
        bad_labels,
        "hash",
    )
    report = validate_constructed_cohort(cohort, cohort_dataset, diabetes_endpoint)
    assert not report.passed
    assert any(not check.passed for check in report.checks)
