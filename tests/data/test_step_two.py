"""Step-two schema, conversion, and validation tests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from metaboguard.data.canonical import (
    CanonicalDataset,
    _cancer_site,
    _condition_category,
    _diabetes_type,
    _encounter_type,
    _observation_value,
    _read_csv,
    _stable_frame,
    convert_unit,
    to_canonical,
)
from metaboguard.data.schema import (
    ClinicalEvent,
    ConditionRecord,
    Patient,
    export_json_schema,
)
from metaboguard.data.validation import _config, validate
from metaboguard.features.dictionary import (
    DeniedFeatureError,
    feature_names,
    get_feature_definition,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthea_one_patient"


def _boundary_fixture(
    tmp_path: Path, *, days_before: int = 1, code: str = "8302-2"
) -> Path:
    raw_dir = tmp_path / "synthea_boundary"
    shutil.copytree(FIXTURE, raw_dir)
    patients_path = raw_dir / "patients.csv"
    patients = list(csv.DictReader(patients_path.open(encoding="utf-8")))
    patients[0]["BIRTHDATE"] = "2015-01-16"
    with patients_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=patients[0].keys())
        writer.writeheader()
        writer.writerows(patients)
    observations_path = raw_dir / "observations.csv"
    observations = list(csv.DictReader(observations_path.open(encoding="utf-8")))
    observations[0].update(
        {
            "DATE": f"2015-01-{16 - days_before:02d}T23:45:47Z",
            "CODE": code,
            "DESCRIPTION": "Height" if code == "8302-2" else "CA 19-9",
            "VALUE": "170" if code == "8302-2" else "42",
            "UNITS": "cm" if code == "8302-2" else "U/mL",
        }
    )
    with observations_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=observations[0].keys())
        writer.writeheader()
        writer.writerows(observations)
    return raw_dir


def _dataset() -> CanonicalDataset:
    return to_canonical(FIXTURE)


def _event(**changes: object) -> ClinicalEvent:
    values: dict[str, object] = {
        "patient_id": "p-001",
        "event_date": date(2020, 1, 1),
        "age_at_event": 40.0,
        "encounter_type": "ambulatory",
        "feature_name": "glucose",
        "value": 100.0,
        "unit": "mg/dL",
        "is_missing": False,
        "provenance": "synthea_native",
        "augmentation_module": None,
    }
    values.update(changes)
    return ClinicalEvent.model_construct(**values)


def _invalid_dataset(**changes: object) -> CanonicalDataset:
    dataset = _dataset()
    return replace(dataset, **changes)


def _check_fails(dataset: CanonicalDataset, name: str) -> None:
    report = validate(dataset)
    result = next(check for check in report.checks if check.name == name)
    assert not result.passed
    assert result.offending_row_count > 0
    assert result.example_ids


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_fixture_conversion_is_clean_and_reports_dropped_code() -> None:
    dataset = _dataset()
    report = validate(dataset)
    assert report.passed
    assert report.dropped_loinc_codes == {"99999-9": 1}
    missingness_check = next(
        check for check in report.checks if check.name == "missingness_rate"
    )
    assert missingness_check.status == "warning"
    assert missingness_check.passed is True
    assert missingness_check.warning_count == 1
    assert len(dataset.patients) == 1
    assert len(dataset.events) == 7
    assert any(
        event.feature_name == "ca_19_9" and event.provenance == "augmented"
        for event in dataset.events
    )
    report_path = Path("artifacts/validation/dataset_validation_report.json")
    assert report_path.exists()
    report_json = json.loads(report_path.read_text(encoding="utf-8"))
    json_missingness = next(
        check for check in report_json["checks"] if check["name"] == "missingness_rate"
    )
    assert json_missingness["status"] == "warning"
    assert json_missingness["passed"] is True
    assert json_missingness["warning_count"] == 1
    assert (
        report_json["report_path"]
        == "artifacts/validation/dataset_validation_report.json"
    )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_conversion_is_byte_deterministic() -> None:
    first = _dataset()
    first_hash = hashlib.sha256((first.output_dir / "events.parquet").read_bytes()).hexdigest()  # type: ignore[union-attr]
    second = _dataset()
    second_hash = hashlib.sha256((second.output_dir / "events.parquet").read_bytes()).hexdigest()  # type: ignore[union-attr]
    assert first_hash == second_hash


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_synthea_birth_boundary_is_normalized_and_audited(tmp_path: Path) -> None:
    dataset = to_canonical(
        _boundary_fixture(tmp_path), source="synthea", source_version="3.3.0"
    )
    assert len(dataset.date_normalisation_audit) == 1
    audit = dataset.date_normalisation_audit[0]
    assert audit["original_event_date"] == "2015-01-15"
    assert audit["normalised_event_date"] == "2015-01-16"
    assert audit["source_version"] == "3.3.0"
    assert audit["source_code"] == "8302-2"
    assert validate(dataset).passed
    audit_path = dataset.output_dir / "date_normalisation_audit.json"  # type: ignore[union-attr]
    assert audit_path.exists()


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
@pytest.mark.parametrize(
    ("days_before", "source", "code"),
    [(2, "synthea", "8302-2"), (1, "other", "8302-2"), (1, "synthea", "aug-ca19-9")],
)
def test_nonqualifying_prebirth_events_remain_chronology_errors(
    tmp_path: Path, days_before: int, source: str, code: str
) -> None:
    dataset = to_canonical(
        _boundary_fixture(tmp_path, days_before=days_before, code=code),
        source=source,
        source_version="3.3.0",
    )
    report = validate(dataset)
    chronology = next(check for check in report.checks if check.name == "chronology")
    assert chronology.passed is False
    assert dataset.date_normalisation_audit == []


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_prebirth_condition_remains_chronology_error(tmp_path: Path) -> None:
    raw_dir = _boundary_fixture(tmp_path)
    conditions_path = raw_dir / "conditions.csv"
    conditions = list(csv.DictReader(conditions_path.open(encoding="utf-8")))
    conditions[0]["START"] = "2015-01-15"
    with conditions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=conditions[0].keys())
        writer.writeheader()
        writer.writerows(conditions)
    report = validate(to_canonical(raw_dir, source="synthea", source_version="3.3.0"))
    chronology = next(check for check in report.checks if check.name == "chronology")
    assert chronology.passed is False


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_birth_boundary_normalization_is_byte_deterministic(tmp_path: Path) -> None:
    first_dir = _boundary_fixture(tmp_path / "first")
    second_dir = _boundary_fixture(tmp_path / "second")
    first = to_canonical(first_dir, source="synthea", source_version="3.3.0")
    second = to_canonical(second_dir, source="synthea", source_version="3.3.0")
    assert (
        first.date_normalisation_audit_sha256 == second.date_normalisation_audit_sha256
    )
    assert (first.output_dir / "date_normalisation_audit.json").read_bytes() == (  # type: ignore[union-attr]
        second.output_dir / "date_normalisation_audit.json"  # type: ignore[union-attr]
    ).read_bytes()
    assert (first.output_dir / "events.parquet").read_bytes() == (  # type: ignore[union-attr]
        second.output_dir / "events.parquet"  # type: ignore[union-attr]
    ).read_bytes()


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_schema_round_trip_and_export(tmp_path: Path) -> None:
    path, digest = export_json_schema(tmp_path / "longitudinal_schema.json")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["x-schema-version"] == "2.0.0"
    assert digest == hashlib.sha256(Path(path).read_bytes()).hexdigest()
    patient = Patient.model_validate(
        {
            "patient_id": "p",
            "birth_date": date(1980, 1, 1),
            "sex": "male",
            "ethnicity": "x",
        }
    )
    assert patient.patient_id == "p"


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_schema_default_artifact_path_is_configured() -> None:
    path, _ = export_json_schema()
    assert Path(path).as_posix().endswith("artifacts/schema/longitudinal_schema.json")


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_schema_rejects_implicit_coercion() -> None:
    with pytest.raises(ValidationError):
        Patient.model_validate(
            {
                "patient_id": "p",
                "birth_date": "1980-01-01",
                "sex": "male",
                "ethnicity": "x",
            }
        )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_schema_rejects_cancer_without_site() -> None:
    with pytest.raises(ValidationError):
        ConditionRecord.model_validate(
            {
                "patient_id": "p",
                "condition_code": "c",
                "condition_system": "s",
                "condition_display": "c",
                "onset_date": date.today(),
                "category": "cancer",
            }
        )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_schema_rejects_diabetes_without_type() -> None:
    with pytest.raises(ValidationError):
        ConditionRecord.model_validate(
            {
                "patient_id": "p",
                "condition_code": "d",
                "condition_system": "s",
                "condition_display": "d",
                "onset_date": date.today(),
                "category": "diabetes",
            }
        )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_denylist_rejection() -> None:
    with pytest.raises(DeniedFeatureError):
        get_feature_definition("tumour_stage")
    with pytest.raises(DeniedFeatureError):
        get_feature_definition("insulin_use", target="diabetes_development")


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
@pytest.mark.parametrize(
    ("source", "canonical", "value", "expected"),
    [
        ("mmol/L", "mg/dL", 5.0, 90.0),
        ("mmol/mol", "%", 53.0, 7.0),
        ("lb", "kg", 220.0, 99.7903234),
        ("in", "cm", 10.0, 25.4),
        ("mm[Hg]", "mmHg", 120.0, 120.0),
        ("kg/m2", "kg/m^2", 25.0, 25.0),
        ("10*3/uL", "10^9/L", 250.0, 250.0),
    ],
)
def test_unit_conversions(
    source: str, canonical: str, value: float, expected: float
) -> None:
    feature = {
        ("mmol/L", "mg/dL"): "glucose",
        ("mmol/mol", "%"): "hba1c",
        ("lb", "kg"): "weight",
        ("in", "cm"): "height",
        ("mm[Hg]", "mmHg"): "systolic_bp",
        ("kg/m2", "kg/m^2"): "bmi",
        ("10*3/uL", "10^9/L"): "platelets",
    }[(source, canonical)]
    assert convert_unit(feature, value, source, canonical) == pytest.approx(
        expected, abs=0.001
    )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_unsupported_unit_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported unit mismatch"):
        convert_unit("glucose", 1.0, "stone", "mg/dL")
    assert convert_unit("glucose", 1.0, "mg/dL", "mg/dL") == 1.0


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_synthea_enzyme_international_unit_alias() -> None:
    assert convert_unit("alt", 42.0, "[iU]/L", "U/L") == 42.0
    assert convert_unit("alkaline_phosphatase", 88.0, "[iU]/L", "U/L") == 88.0
    with pytest.raises(ValueError, match="Unsupported unit mismatch"):
        convert_unit("creatinine", 1.0, "[iU]/L", "mg/dL")


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def _creatinine_fixture(tmp_path: Path, value: str = "97.1") -> Path:
    raw_dir = tmp_path / f"synthea_creatinine_{value}"
    shutil.copytree(FIXTURE, raw_dir)
    observations_path = raw_dir / "observations.csv"
    observations = list(csv.DictReader(observations_path.open(encoding="utf-8")))
    observations[0].update(
        {
            "CODE": "2160-0",
            "DESCRIPTION": "Creatinine [Mass/volume] in Serum or Plasma",
            "VALUE": value,
            "UNITS": "mg/dL",
        }
    )
    with observations_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=observations[0].keys())
        writer.writeheader()
        writer.writerows(observations)
    return raw_dir


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_synthea_micromolar_creatinine_is_normalised_and_audited(
    tmp_path: Path,
) -> None:
    dataset = to_canonical(
        _creatinine_fixture(tmp_path), source="synthea", source_version="3.3.0"
    )
    assert len(dataset.unit_normalisation_audit) == 1
    audit = dataset.unit_normalisation_audit[0]
    assert audit["original_value"] == 97.1
    assert audit["original_unit"] == "mg/dL"
    assert audit["normalised_value"] == pytest.approx(1.098, abs=0.01)
    assert audit["source_version"] == "3.3.0"
    assert validate(dataset).passed
    assert (dataset.output_dir / "unit_normalisation_audit.json").exists()  # type: ignore[union-attr]
    assert dataset.unit_normalisation_audit_sha256


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
@pytest.mark.parametrize(
    ("source", "value"),
    [("other", "97.1"), ("synthea", "1.1"), ("synthea", "10000")],
)
def test_creatinine_normalisation_is_narrowly_gated(
    tmp_path: Path, source: str, value: str
) -> None:
    dataset = to_canonical(
        _creatinine_fixture(tmp_path, value=value),
        source=source,
        source_version="3.3.0",
    )
    assert dataset.unit_normalisation_audit == []


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_synthea_coded_observation_mapping() -> None:
    assert _observation_value("smoking_status", "Never smoked tobacco", "") == 0.0
    assert _observation_value("smoking_status", "Current smoker", "") == 1.0
    assert _observation_value("glucose", "100", "mg/dL") == 100.0


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_canonical_mapping_helpers_cover_unknown_values() -> None:
    assert _encounter_type("OTHER") == "other"
    assert _condition_category("hypertension") == "other"
    assert _condition_category("Prediabetes") == "other"
    assert _condition_category("Benign neoplasm of lip") == "other"
    assert _condition_category("Malignant melanoma") == "cancer"
    assert _condition_category("malignant neoplasm") == "cancer"
    assert _diabetes_type("Type 1 diabetes") == "type1"
    assert _diabetes_type("gestational diabetes") == "gestational"
    assert _diabetes_type("unspecified diabetes") is None
    assert _cancer_site("lung cancer") == "lung"
    assert _cancer_site("cancer of unknown site") == "unspecified"
    assert _cancer_site("benign lesion") is None
    assert _stable_frame([], ["id"]).empty
    assert "hba1c" in feature_names()


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_missing_input_files_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FileNotFoundError):
        _read_csv(tmp_path, "patients")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        _config()


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_schema_event_governance_branches() -> None:
    with pytest.raises(ValidationError):
        ClinicalEvent(**_event(is_missing=False, value=None).model_dump())
    with pytest.raises(ValidationError):
        ClinicalEvent(
            **_event(provenance="augmented", augmentation_module=None).model_dump()
        )
    with pytest.raises(ValidationError):
        ClinicalEvent(
            **_event(
                provenance="synthea_native", augmentation_module="insulin"
            ).model_dump()
        )
    with pytest.raises(ValidationError):
        ClinicalEvent(
            **_event(
                feature_name="ca_19_9", unit="U/mL", provenance="synthea_native"
            ).model_dump()
        )
    with pytest.raises(ValidationError):
        ClinicalEvent(**_event(unit="mmol/L").model_dump())


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_referential_integrity() -> None:
    _check_fails(
        _invalid_dataset(events=[_event(patient_id="missing")]), "referential_integrity"
    )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_chronology() -> None:
    _check_fails(
        _invalid_dataset(events=[_event(event_date=date(1970, 1, 1))]), "chronology"
    )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_no_events_after_death() -> None:
    _check_fails(
        _invalid_dataset(events=[_event(event_date=date(2024, 1, 1))]),
        "no_events_after_death",
    )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_age_at_event() -> None:
    _check_fails(_invalid_dataset(events=[_event(age_at_event=99.0)]), "age_at_event")


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_plausible_ranges() -> None:
    _check_fails(_invalid_dataset(events=[_event(value=9999.0)]), "plausible_ranges")
    accepted = _invalid_dataset(
        events=[_event(feature_name="systolic_bp", value=20.0, unit="mmHg")]
    )
    report = validate(accepted)
    systolic = next(
        check for check in report.checks if check.name == "plausible_ranges"
    )
    assert systolic.passed
    _check_fails(
        _invalid_dataset(
            events=[_event(feature_name="systolic_bp", value=19.0, unit="mmHg")]
        ),
        "plausible_ranges",
    )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_missingness_consistency() -> None:
    _check_fails(
        _invalid_dataset(events=[_event(is_missing=True, value=1.0)]),
        "missingness_consistency",
    )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_units() -> None:
    _check_fails(_invalid_dataset(events=[_event(unit="mmol/L")]), "canonical_units")


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_validation_denylist() -> None:
    _check_fails(
        _invalid_dataset(
            events=[_event(feature_name="tumour_stage", unit="coded", value=0.0)]
        ),
        "denylist",
    )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_augmentation_module() -> None:
    _check_fails(
        _invalid_dataset(
            events=[_event(provenance="augmented", augmentation_module="unknown")]
        ),
        "augmentation_module",
    )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_diabetes_onset_after_birth() -> None:
    condition = (
        _dataset().conditions[0].model_copy(update={"onset_date": date(1900, 1, 1)})
    )
    _check_fails(_invalid_dataset(conditions=[condition]), "diabetes_onset_after_birth")


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_cancer_site() -> None:
    condition = (
        _dataset()
        .conditions[1]
        .model_construct(
            **{**_dataset().conditions[1].model_dump(), "cancer_site": None}
        )
    )
    _check_fails(_invalid_dataset(conditions=[condition]), "cancer_site")


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_duplicate_patient_ids() -> None:
    _check_fails(
        _invalid_dataset(patients=[_dataset().patients[0], _dataset().patients[0]]),
        "unique_patient_ids",
    )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_validation_computes_hash_when_dataset_hash_is_empty() -> None:
    report = validate(replace(_dataset(), dataset_sha256=""))
    assert report.dataset_sha256
