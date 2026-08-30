from datetime import date

from metaboguard.cohort.endpoints import assign_outcomes
from metaboguard.cohort.protocol import PatientIndex


def _index(patient_id: str) -> PatientIndex:
    return PatientIndex(
        patient_id=patient_id,
        cohort_class="ordinary_incidence",
        endpoint_id="type2_diabetes",
        index_date=date(2016, 1, 1),
        index_source="test",
        index_sequence_number=0,
        preindex_event_dates=(date(2015, 1, 1), date(2016, 1, 1)),
        preindex_measurement_dates=(date(2015, 1, 1), date(2016, 1, 1)),
        preindex_encounter_count=2,
    )


def test_independent_horizon_states(cohort_dataset, diabetes_endpoint) -> None:
    labels = assign_outcomes(
        [
            _index("p-positive"),
            _index("p-competing"),
            _index("p-censored"),
            _index("p-negative"),
        ],
        cohort_dataset,
        diabetes_endpoint,
    )
    by_key = {(label.patient_id, label.horizon_years): label for label in labels}
    assert by_key[("p-positive", 3)].label_state == "positive"
    assert by_key[("p-competing", 3)].label_state == "competing_death"
    assert by_key[("p-censored", 1)].label_state == "censored"
    assert by_key[("p-negative", 1)].label_state == "eligible_negative"
    assert by_key[("p-positive", 3)].event_date == date(2019, 1, 1)
    assert by_key[("p-positive", 3)].days_to_event_or_censor == 1096


def test_death_after_endpoint_keeps_positive(cohort_dataset, diabetes_endpoint) -> None:
    patient = next(value for value in cohort_dataset.patients if value.patient_id == "p-positive")
    from dataclasses import replace

    changed = replace(
        cohort_dataset,
        patients=[patient.model_copy(update={"death_date": date(2022, 1, 1)})],
    )
    label = assign_outcomes([_index("p-positive")], changed, diabetes_endpoint)
    assert next(item for item in label if item.horizon_years == 3).label_state == "positive"


def test_death_without_endpoint_is_competing(cohort_dataset, diabetes_endpoint) -> None:
    from dataclasses import replace

    changed = replace(
        cohort_dataset,
        conditions=[
            condition
            for condition in cohort_dataset.conditions
            if condition.patient_id != "p-competing"
        ],
    )
    labels = assign_outcomes([_index("p-competing")], changed, diabetes_endpoint)
    assert next(item for item in labels if item.horizon_years == 3).label_state == "competing_death"
