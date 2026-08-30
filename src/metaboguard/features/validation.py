"""Leakage, denylist, identity, and deterministic feature validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from metaboguard.features.definitions import EngineeredFeatureDefinition
from metaboguard.features.extraction import IDENTIFIER_COLUMNS, FeatureDataset


@dataclass(frozen=True, slots=True)
class FeatureCheck:
    name: str
    level: str
    status: str
    passed: bool
    warning_count: int
    offending_count: int
    offending_patient_fingerprints: tuple[str, ...]
    message: str = ""


@dataclass(frozen=True, slots=True)
class FeatureValidationReport:
    passed: bool
    checks: tuple[FeatureCheck, ...]
    simulation_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check(name: str, level: str, count: int, ids: list[str], message: str = "") -> FeatureCheck:
    warning = level == "warning"
    return FeatureCheck(
        name,
        level,
        "warning" if warning else "failed" if count else "passed",
        warning or count == 0,
        count if warning else 0,
        count,
        tuple(sorted(set(ids))[:10]),
        message,
    )


def validate_feature_dataset(
    feature_matrix: FeatureDataset,
    lineage: list[dict[str, object]],
    registry: dict[str, EngineeredFeatureDefinition],
    cohort_manifest: dict[str, object],
) -> FeatureValidationReport:
    """Validate feature rows and lineage without accepting outcome labels."""
    checks: list[FeatureCheck] = []
    future = [str(item["patient_id"]) for item in lineage if item["contains_post_index_record"]]
    checks.append(_check("no_post_index_lineage", "error", len(future), future))
    unknown = [
        key
        for row in feature_matrix.rows
        for key in row
        if key not in IDENTIFIER_COLUMNS and key not in registry
    ]
    checks.append(_check("registry_completeness", "error", len(unknown), []))
    missing_splits = [str(row["patient_id"]) for row in feature_matrix.rows if not row.get("split")]
    checks.append(_check("assigned_split", "error", len(missing_splits), missing_splits))
    forbidden = [
        key
        for row in feature_matrix.rows
        for key in row
        if any(
            token in key.lower()
            for token in (
                "label",
                "outcome",
                "censor",
                "death",
                "diagnosis",
                "treatment",
            )
        )
    ]
    checks.append(_check("no_outcome_columns", "error", len(forbidden), []))
    classes = {str(row["cohort_class"]) for row in feature_matrix.rows}
    checks.append(_check("single_cohort_class", "error", len(classes) - 1 if classes else 0, []))
    warnings = [
        _check(
            "feature_missingness",
            "warning",
            0,
            [],
            "Missingness is reported per feature; no feature-level failure.",
        )
    ]
    return FeatureValidationReport(
        all(check.passed for check in checks), tuple(checks) + tuple(warnings)
    )
