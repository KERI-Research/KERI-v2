"""Feature completeness and availability summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from metaboguard.readiness.contracts import FeatureAvailabilityRecord
from metaboguard.readiness.inventory import feature_artifact_dir


def build_feature_availability(run_path: Path, endpoint_id: str) -> list[FeatureAvailabilityRecord]:
    features = feature_artifact_dir(run_path, endpoint_id)
    matrix = pd.read_parquet(features / "feature_matrix.parquet")
    registry_rows = pd.read_json(features / "feature_definition_registry.json")
    registry = {row["feature_id"]: row for row in registry_rows.to_dict(orient="records")}
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
    class_name = str(matrix["cohort_class"].iloc[0]) if len(matrix) else ""
    split_name = str(matrix["split"].iloc[0]) if len(matrix) else "unknown"
    records: list[FeatureAvailabilityRecord] = []
    for feature_id in matrix.columns:
        if feature_id in identifiers:
            continue
        definition = registry.get(feature_id)
        if definition is None:
            continue
        values = matrix[feature_id]
        numeric = pd.to_numeric(values, errors="coerce")
        nonnull = values.notna().sum()
        distinct = numeric.dropna().nunique()
        records.append(
            FeatureAvailabilityRecord(
                cohort_class=class_name,
                endpoint_id=endpoint_id,
                split=split_name,
                feature_id=feature_id,
                feature_family=str(definition["feature_family"]),
                source_feature_id=str(definition["source_feature_id"]),
                window=str(definition["window_days"]),
                value_type=str(definition["value_type"]),
                unit=str(definition["unit"]),
                row_count=len(matrix),
                available_count=int(nonnull),
                availability_fraction=float(nonnull / len(matrix)) if len(matrix) else 0.0,
                null_count=int(values.isna().sum()),
                null_fraction=float(values.isna().mean()) if len(matrix) else 1.0,
                all_null=bool(nonnull == 0),
                zero_variance=bool(distinct <= 1),
                simulation_only=True,
            )
        )
    return records
