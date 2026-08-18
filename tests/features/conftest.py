from datetime import date

import pytest

from metaboguard.cohort.protocol import PatientIndex
from metaboguard.data.canonical import CanonicalDataset
from metaboguard.data.schema import ClinicalEvent, Patient


@pytest.fixture
def feature_dataset() -> CanonicalDataset:
    patient = Patient(patient_id="p1", birth_date=date(1980, 1, 1), sex="male", ethnicity="x")
    events = [
        ClinicalEvent(
            patient_id="p1",
            event_date=date(2019, 1, 1),
            age_at_event=39.0,
            encounter_type="ambulatory",
            feature_name="glucose",
            value=90.0,
            unit="mg/dL",
            is_missing=False,
            provenance="synthea_native",
        ),
        ClinicalEvent(
            patient_id="p1",
            event_date=date(2019, 6, 1),
            age_at_event=39.4,
            encounter_type="ambulatory",
            feature_name="glucose",
            value=110.0,
            unit="mg/dL",
            is_missing=False,
            provenance="synthea_native",
        ),
        ClinicalEvent(
            patient_id="p1",
            event_date=date(2020, 1, 1),
            age_at_event=40.0,
            encounter_type="ambulatory",
            feature_name="glucose",
            value=130.0,
            unit="mg/dL",
            is_missing=False,
            provenance="synthea_native",
        ),
        ClinicalEvent(
            patient_id="p1",
            event_date=date(2021, 1, 1),
            age_at_event=41.0,
            encounter_type="ambulatory",
            feature_name="glucose",
            value=999.0,
            unit="mg/dL",
            is_missing=False,
            provenance="synthea_native",
        ),
    ]
    return CanonicalDataset(
        [patient], events, [], [], cohort_class="ordinary_incidence", dataset_sha256="canonical"
    )


@pytest.fixture
def feature_index() -> PatientIndex:
    return PatientIndex(
        "p1",
        "ordinary_incidence",
        "type2_diabetes",
        date(2020, 1, 1),
        "test",
        0,
        (date(2019, 1, 1), date(2019, 6, 1), date(2020, 1, 1)),
        (date(2019, 1, 1), date(2019, 6, 1), date(2020, 1, 1)),
        3,
    )
