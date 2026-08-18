import hashlib
from pathlib import Path

import pandas as pd

from metaboguard.data.production_manifests import ProductionBatchRecord, new_production_manifest
from metaboguard.data.production_validation import (
    validate_cohort_table,
    validate_production_manifest,
    validate_recorded_hash,
    validation_payload,
)


def test_manifest_validation_rejects_failed_completed_run() -> None:
    manifest = new_production_manifest("ordinary_incidence", 10, 1, "baseline", "test", {"seed": 1})
    manifest.generated_patient_count = 11
    manifest.status = "completed"
    manifest.batch_records = [
        ProductionBatchRecord(
            batch_index=0,
            seed=1,
            target_size=10,
            source_output_path="raw",
            status="failed",
            started_at="now",
        )
    ]
    report = validate_production_manifest(manifest)
    assert not report.passed
    assert any(check.name == "population_target_not_overclaimed" for check in report.checks)


def test_cohort_table_and_hash_validation(tmp_path: Path) -> None:
    good = validate_cohort_table(
        pd.DataFrame({"cohort_class": ["enriched_incidence", "enriched_incidence"]}),
        "enriched_incidence",
    )
    mixed = validate_cohort_table(
        pd.DataFrame({"cohort_class": ["ordinary_incidence", "enriched_incidence"]}),
        "ordinary_incidence",
    )
    missing = validate_cohort_table(pd.DataFrame({"patient_id": ["p"]}), "ordinary_incidence")
    assert good.passed
    assert not mixed.passed
    assert not missing.passed
    path = tmp_path / "artifact.txt"
    path.write_text("ok", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert validate_recorded_hash(path, digest).passed
    assert not validate_recorded_hash(path, "0" * 64).passed
    assert not validate_recorded_hash(tmp_path / "missing", digest).passed
    invalid = new_production_manifest("ordinary_incidence", 1, 1, "baseline", "test", {})
    invalid.cohort_class = "invalid"  # type: ignore[assignment]
    report = validate_production_manifest(invalid)
    assert not report.passed
    assert validation_payload(report)["passed"] is False
