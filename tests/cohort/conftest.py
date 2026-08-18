"""Small deterministic canonical cohorts for Step 4 tests."""

from datetime import date

import pytest

from metaboguard.cohort.protocol import EndpointProtocol
from metaboguard.data.canonical import CanonicalDataset
from metaboguard.data.schema import ClinicalEvent, ConditionRecord, Patient


def _event(patient_id: str, event_date: date, value: float = 100.0) -> ClinicalEvent:
    return ClinicalEvent(
        patient_id=patient_id,
        event_date=event_date,
        age_at_event=float(event_date.year - 1980),
        encounter_type="ambulatory",
        feature_name="glucose",
        value=value,
        unit="mg/dL",
        is_missing=False,
        provenance="synthea_native",
    )


@pytest.fixture
def cohort_dataset() -> CanonicalDataset:
    patients = [
        Patient(patient_id="p-positive", birth_date=date(1980, 1, 1), sex="male", ethnicity="x"),
        Patient(
            patient_id="p-competing",
            birth_date=date(1980, 1, 1),
            sex="male",
            ethnicity="x",
            death_date=date(2018, 6, 1),
        ),
        Patient(patient_id="p-censored", birth_date=date(1980, 1, 1), sex="male", ethnicity="x"),
        Patient(patient_id="p-negative", birth_date=date(1980, 1, 1), sex="female", ethnicity="x"),
        Patient(patient_id="p-prevalent", birth_date=date(1980, 1, 1), sex="female", ethnicity="x"),
    ]
    events = []
    for patient_id in [patient.patient_id for patient in patients]:
        events.extend([_event(patient_id, date(2015, 1, 1)), _event(patient_id, date(2016, 1, 1))])
    events.extend([_event("p-positive", date(2019, 1, 1)), _event("p-negative", date(2020, 1, 1))])
    conditions = [
        ConditionRecord(
            patient_id="p-positive",
            condition_code="d1",
            condition_system="s",
            condition_display="Type 2 diabetes mellitus",
            onset_date=date(2019, 1, 1),
            category="diabetes",
            diabetes_type="type2",
        ),
        ConditionRecord(
            patient_id="p-positive",
            condition_code="c1",
            condition_system="s",
            condition_display="Pancreatic cancer",
            onset_date=date(2020, 1, 1),
            category="cancer",
            cancer_site="pancreas",
        ),
        ConditionRecord(
            patient_id="p-competing",
            condition_code="d3",
            condition_system="s",
            condition_display="Type 2 diabetes mellitus",
            onset_date=date(2018, 6, 1),
            category="diabetes",
            diabetes_type="type2",
        ),
        ConditionRecord(
            patient_id="p-prevalent",
            condition_code="d2",
            condition_system="s",
            condition_display="Type 2 diabetes mellitus",
            onset_date=date(2015, 6, 1),
            category="diabetes",
            diabetes_type="type2",
        ),
    ]
    return CanonicalDataset(
        patients,
        events,
        conditions,
        [],
        cohort_class="ordinary_incidence",
        dataset_sha256="dataset-hash",
    )


@pytest.fixture
def diabetes_endpoint() -> EndpointProtocol:
    return EndpointProtocol(
        endpoint_id="type2_diabetes",
        outcome_family="diabetes",
        cancer_site=None,
        diabetes_type="type2",
        horizon_years=(1, 3, 5),
        minimum_age_years=18,
        minimum_history_days=365,
        minimum_preindex_encounters=2,
        minimum_preindex_measurements=2,
        rolling_index_interval_days=365,
        prevalent_exclusion=True,
        washout_days=0,
        use_competing_death_risk=True,
        enabled=False,
    )


@pytest.fixture
def pancreatic_endpoint() -> EndpointProtocol:
    return EndpointProtocol(
        endpoint_id="pancreatic_cancer",
        outcome_family="cancer",
        cancer_site="pancreas",
        diabetes_type=None,
        horizon_years=(1, 3, 5),
        minimum_age_years=18,
        minimum_history_days=365,
        minimum_preindex_encounters=2,
        minimum_preindex_measurements=2,
        rolling_index_interval_days=365,
        prevalent_exclusion=True,
        washout_days=365,
        use_competing_death_risk=True,
        enabled=False,
    )
