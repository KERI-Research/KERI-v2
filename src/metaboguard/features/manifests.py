"""Feature artefact writing, hashes, lineage, and parent-status updates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from metaboguard.features.definitions import registry_hash
from metaboguard.features.extraction import FeatureDataset
from metaboguard.features.validation import validate_feature_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_feature_artifacts(
    feature_dataset: FeatureDataset,
    output_dir: Path,
    cohort_manifest_path: Path | None = None,
) -> dict[str, object]:
    """Write deterministic feature, lineage, quality, validation, and manifest files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.DataFrame(feature_dataset.rows)
    if not matrix.empty:
        matrix = matrix.sort_values(
            ["cohort_class", "endpoint_id", "patient_id", "index_date"], kind="mergesort"
        )
    matrix_path = output_dir / "feature_matrix.parquet"
    matrix.to_parquet(matrix_path, index=False, engine="pyarrow", compression="zstd")
    lineage = pd.DataFrame(feature_dataset.lineage)
    if not lineage.empty:
        lineage = lineage.sort_values(
            ["patient_id", "endpoint_id", "index_date", "feature_id"], kind="mergesort"
        )
    lineage_path = output_dir / "feature_lineage.parquet"
    lineage.to_parquet(lineage_path, index=False, engine="pyarrow", compression="zstd")
    registry_payload = [item.to_dict() for item in feature_dataset.registry.values()]
    registry_path = output_dir / "feature_definition_registry.json"
    registry_path.write_text(
        json.dumps(registry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    registry_sha = registry_hash(feature_dataset.registry)
    (output_dir / "feature_definition_registry.sha256").write_text(
        registry_sha + "\n", encoding="utf-8"
    )
    validation = validate_feature_dataset(
        feature_dataset, feature_dataset.lineage, feature_dataset.registry, {}
    )
    validation_path = output_dir / "feature_validation_report.json"
    validation_path.write_text(
        json.dumps(validation.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    quality = {
        "feature_count": len(feature_dataset.registry),
        "row_count": len(feature_dataset.rows),
        "unique_patient_count": len({row["patient_id"] for row in feature_dataset.rows}),
        "simulation_only": True,
        "warnings": [
            check.__dict__
            if hasattr(check, "__dict__")
            else {
                "name": check.name,
                "status": check.status,
                "passed": check.passed,
                "warning_count": check.warning_count,
            }
            for check in validation.checks
            if check.level == "warning"
        ],
    }
    quality_path = output_dir / "feature_quality_report.json"
    quality_path.write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "feature_matrix_sha256": _sha256(matrix_path),
        "lineage_sha256": _sha256(lineage_path),
        "registry_sha256": registry_sha,
        "validation_report_sha256": _sha256(validation_path),
        "quality_report_sha256": _sha256(quality_path),
        "row_count": len(feature_dataset.rows),
        "patient_count": len({row["patient_id"] for row in feature_dataset.rows}),
        "feature_count": len(feature_dataset.registry),
        "simulation_only": True,
        "split_status": "created",
        "feature_status": "created",
        "model_status": "not_created",
        "validation_passed": validation.passed,
    }
    manifest_path = output_dir / "feature_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if cohort_manifest_path is not None and validation.passed:
        parent = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
        parent.update(
            {"split_status": "created", "feature_status": "created", "model_status": "not_created"}
        )
        cohort_manifest_path.write_text(
            json.dumps(parent, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return manifest
