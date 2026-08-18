"""Read-only inventory and completeness classification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from metaboguard.readiness.contracts import ArtifactInventory, ArtifactRecord, FeatureBuildStatus


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_artifact_dir(run_path: Path, endpoint_id: str) -> Path:
    """Resolve the canonical endpoint feature path or explicit inspection fallback."""
    direct = run_path / "features" / endpoint_id
    if direct.is_dir():
        return direct
    inspection = run_path / "features" / f"{endpoint_id}_inspection"
    return inspection


def _record(run: Path, relative: str, required: bool) -> ArtifactRecord:
    path = run / relative
    present = path.is_file()
    size = path.stat().st_size if present else 0
    digest = _hash(path) if present else ""
    recorded = None
    if present and path.name.endswith("manifest.json"):
        try:
            recorded = json.loads(path.read_text(encoding="utf-8")).get("sha256")
        except (OSError, ValueError):
            recorded = None
    return ArtifactRecord(
        path=relative,
        required=required,
        present=present,
        byte_size=size,
        sha256=digest,
        recorded_sha256=recorded,
        hash_matches=None if recorded is None else recorded == digest,
    )


def inspect_artifact_inventory(run_path: Path, endpoint_id: str) -> ArtifactInventory:
    """Inspect one immutable run without mutating any input artifact."""
    cohort = run_path / "cohort" / endpoint_id
    features = feature_artifact_dir(run_path, endpoint_id)
    feature_prefix = features.relative_to(run_path).as_posix()
    required = [
        "manifest.json",
        "canonical/patients.parquet",
        "canonical/events.parquet",
        "canonical/conditions.parquet",
        "canonical/outcomes.parquet",
        f"cohort/{endpoint_id}/cohort_manifest.json",
        f"cohort/{endpoint_id}/endpoint_protocol.json",
        f"cohort/{endpoint_id}/eligible_indexes.parquet",
        f"cohort/{endpoint_id}/splits/split_assignments.parquet",
        f"cohort/{endpoint_id}/splits/split_manifest.json",
        f"{feature_prefix}/feature_matrix.parquet",
        f"{feature_prefix}/feature_lineage.parquet",
        f"{feature_prefix}/feature_manifest.json",
    ]
    optional = [
        f"cohort/{endpoint_id}/splits/split_validation_report.json",
        f"{feature_prefix}/feature_quality_report.json",
        f"{feature_prefix}/feature_validation_report.json",
    ]
    records = [_record(run_path, path, True) for path in required] + [
        _record(run_path, path, False) for path in optional
    ]
    missing_required = [record.path for record in records if record.required and not record.present]
    missing_optional = [
        record.path for record in records if not record.required and not record.present
    ]
    feature_row_count = 0
    expected_index_count = 0
    if (features / "feature_matrix.parquet").is_file():
        import pandas as pd  # type: ignore[import-untyped]

        feature_row_count = len(pd.read_parquet(features / "feature_matrix.parquet"))
    if (cohort / "eligible_indexes.parquet").is_file():
        import pandas as pd

        expected_index_count = len(pd.read_parquet(cohort / "eligible_indexes.parquet"))
    if not (features / "feature_matrix.parquet").exists():
        status = "not_started"
    elif expected_index_count and feature_row_count < expected_index_count:
        status = "partial"
    elif expected_index_count == feature_row_count:
        status = "complete"
    else:
        status = "unknown"
    run_manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    return ArtifactInventory(
        run_path=str(run_path),
        endpoint_id=endpoint_id,
        cohort_class=str(run_manifest["cohort_class"]),
        artifacts=records,
        missing_required=missing_required,
        missing_optional=missing_optional,
        feature_build_status=cast(FeatureBuildStatus, status),
        feature_row_count=feature_row_count,
        expected_eligible_index_count=expected_index_count,
        operational_warnings=(
            ["full_feature_build_runtime_termination_recorded; inspected feature set is partial"]
            if status == "partial"
            else []
        ),
        simulation_only=bool(run_manifest.get("simulation_only", False)),
    )
