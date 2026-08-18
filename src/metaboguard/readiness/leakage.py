"""Readiness-level lineage and feature leakage audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pandas as pd  # type: ignore[import-untyped]

from metaboguard.readiness.contracts import FeatureBuildStatus, LeakageReadinessReport
from metaboguard.readiness.inventory import feature_artifact_dir


def audit_feature_leakage(run_path: Path, endpoint_id: str) -> LeakageReadinessReport:
    features = feature_artifact_dir(run_path, endpoint_id)
    matrix = pd.read_parquet(features / "feature_matrix.parquet")
    lineage = pd.read_parquet(features / "feature_lineage.parquet")
    registry = json.loads(
        (features / "feature_definition_registry.json").read_text(encoding="utf-8")
    )
    registry_ids = {item["feature_id"] for item in registry}
    identifiers = {
        "patient_id",
        "cohort_class",
        "endpoint_id",
        "index_date",
        "split",
        "index_sequence_number",
        "feature_definition_version",
        "feature_window_config_sha256",
        "source_canonical_manifest_sha256",
        "source_cohort_manifest_sha256",
        "source_split_manifest_sha256",
        "simulation_only",
    }
    checks: list[dict[str, object]] = []
    forbidden_columns = [
        column
        for column in matrix.columns
        if column not in identifiers
        and any(
            token in column.lower()
            for token in ("label", "outcome", "censor", "death", "diagnosis", "treatment")
        )
    ]
    checks.append(
        {
            "name": "no_forbidden_columns",
            "level": "error",
            "passed": not forbidden_columns,
            "offending_count": len(forbidden_columns),
        }
    )
    future = (
        int(
            (
                pd.to_datetime(lineage["latest_source_date"])
                > pd.to_datetime(lineage["index_date"])
            ).sum()
        )
        if len(lineage)
        else 0
    )
    checks.append(
        {
            "name": "no_post_index_lineage",
            "level": "error",
            "passed": future == 0,
            "offending_count": future,
        }
    )
    flags = int(lineage["contains_post_index_record"].sum()) if len(lineage) else 0
    checks.append(
        {
            "name": "lineage_post_index_flag",
            "level": "error",
            "passed": flags == 0,
            "offending_count": flags,
        }
    )
    unknown = [
        column
        for column in matrix.columns
        if column not in identifiers and column not in registry_ids
    ]
    checks.append(
        {
            "name": "registry_columns",
            "level": "error",
            "passed": not unknown,
            "offending_count": len(unknown),
        }
    )
    manifest = json.loads((features / "feature_manifest.json").read_text(encoding="utf-8"))
    checks.append(
        {
            "name": "simulation_only",
            "level": "error",
            "passed": manifest.get("simulation_only") is True,
            "offending_count": int(manifest.get("simulation_only") is not True),
        }
    )
    eligible_indexes = pd.read_parquet(
        run_path / "cohort" / endpoint_id / "eligible_indexes.parquet"
    )
    feature_build_status = (
        "complete"
        if len(matrix) == len(eligible_indexes)
        else "partial"
        if len(matrix) < len(eligible_indexes)
        else "unknown"
    )
    return LeakageReadinessReport(
        passed=all(bool(check["passed"]) for check in checks),
        checks=checks,
        source_hashes={
            "feature_manifest": hashlib.sha256(
                (features / "feature_manifest.json").read_bytes()
            ).hexdigest()
        },
        manual_inspection_scope=(
            "bounded real artifact inspection; feature build completeness is separately gated"
        ),
        feature_build_status=cast(FeatureBuildStatus, feature_build_status),
        simulation_only=True,
    )
