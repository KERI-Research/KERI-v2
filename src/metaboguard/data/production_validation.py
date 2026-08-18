"""Fail-closed validation for Step 7 production artifacts and separation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict

from metaboguard.data.manifests import CohortClass, assert_same_cohort_class
from metaboguard.data.production_manifests import ProductionRunManifest


class ProductionValidationCheck(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    level: str
    status: str
    passed: bool
    warning_count: int = 0
    offending_count: int = 0
    message: str = ""


class ProductionValidationReport(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    passed: bool
    checks: list[ProductionValidationCheck]
    simulation_only: bool = True


def _check(name: str, passed: bool, message: str = "") -> ProductionValidationCheck:
    return ProductionValidationCheck(
        name=name,
        level="error",
        status="passed" if passed else "failed",
        passed=passed,
        offending_count=0 if passed else 1,
        message=message,
    )


def validate_production_manifest(manifest: ProductionRunManifest) -> ProductionValidationReport:
    """Validate identity, stage ordering, failure accounting, and model prohibition."""
    checks = [
        _check("simulation_only", manifest.simulation_only and manifest.pipeline_rehearsal_only),
        _check("model_status", manifest.model_status == "not_created"),
        _check("cohort_class", _class_is_valid(manifest.cohort_class)),
        _check(
            "population_target_not_overclaimed",
            manifest.generated_patient_count <= manifest.population_target,
        ),
        _check(
            "failed_batches_are_not_complete",
            not any(batch.status == "failed" for batch in manifest.batch_records)
            or manifest.status != "completed",
        ),
        _check(
            "stage_order",
            not (manifest.readiness_status == "created" and manifest.feature_status != "created"),
        ),
    ]
    return ProductionValidationReport(passed=all(check.passed for check in checks), checks=checks)


def _class_is_valid(cohort_class: str) -> bool:
    try:
        assert_same_cohort_class([cohort_class])
    except ValueError:
        return False
    return True


def validate_cohort_table(
    frame: pd.DataFrame, expected_class: CohortClass
) -> ProductionValidationReport:
    """Reject missing, mixed, or misclassified cohort-class columns."""
    checks: list[ProductionValidationCheck] = []
    has_class = "cohort_class" in frame.columns
    checks.append(_check("cohort_class_column", has_class))
    values = set(str(value) for value in frame["cohort_class"].dropna()) if has_class else set()
    single_class = values == {expected_class}
    checks.append(_check("single_expected_cohort_class", single_class))
    return ProductionValidationReport(passed=all(check.passed for check in checks), checks=checks)


def validate_recorded_hash(path: Path, expected_sha256: str) -> ProductionValidationCheck:
    """Check one artifact hash without accepting a mismatch."""
    if not path.is_file():
        return _check("artifact_present", False, str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return _check("artifact_hash", digest == expected_sha256)


def validation_payload(report: ProductionValidationReport) -> dict[str, Any]:
    """Return a JSON-compatible validation report payload."""
    return report.model_dump()
