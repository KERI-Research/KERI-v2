"""Feature artefact writing, hashes, lineage, and parent-status updates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from metaboguard.features.definitions import registry_hash
from metaboguard.features.extraction import IDENTIFIER_COLUMNS, FeatureDataset
from metaboguard.features.validation import validate_feature_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_table(batch: FeatureDataset) -> pa.Table:
    fields = [
        pa.field(
            column,
            pa.bool_()
            if column == "simulation_only"
            else pa.date32()
            if column == "index_date"
            else pa.int64()
            if column == "index_sequence_number"
            else pa.string(),
        )
        for column in IDENTIFIER_COLUMNS
    ] + [
        pa.field(
            feature_id,
            pa.string()
            if definition.value_type == "categorical"
            else pa.int64()
            if definition.value_type == "integer"
            else pa.float64(),
        )
        for feature_id, definition in batch.registry.items()
    ]
    frame = pd.DataFrame(batch.rows)
    for column in IDENTIFIER_COLUMNS:
        if column not in frame:
            frame[column] = None
    frame = frame[list(IDENTIFIER_COLUMNS) + list(batch.registry)]
    frame["index_date"] = pd.to_datetime(frame["index_date"], errors="coerce").dt.date
    frame["simulation_only"] = frame["simulation_only"].astype("boolean")
    frame["index_sequence_number"] = pd.to_numeric(
        frame["index_sequence_number"], errors="coerce"
    ).astype("Int64")
    for feature_id, definition in batch.registry.items():
        if definition.value_type == "categorical":
            frame[feature_id] = frame[feature_id].astype("string")
        elif definition.value_type == "integer":
            frame[feature_id] = pd.to_numeric(frame[feature_id], errors="coerce").astype("Int64")
        else:
            frame[feature_id] = pd.to_numeric(frame[feature_id], errors="coerce").astype("float64")
    return pa.Table.from_pandas(frame, schema=pa.schema(fields), preserve_index=False)


def _lineage_table(batch: FeatureDataset) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("patient_id", pa.string()),
            pa.field("endpoint_id", pa.string()),
            pa.field("index_date", pa.date32()),
            pa.field("feature_id", pa.string()),
            pa.field("source_feature_id", pa.string()),
            pa.field("window_name", pa.string()),
            pa.field("window_start_date", pa.date32()),
            pa.field("window_end_date", pa.date32()),
            pa.field("source_record_count", pa.int64()),
            pa.field("earliest_source_date", pa.date32()),
            pa.field("latest_source_date", pa.date32()),
            pa.field("contains_post_index_record", pa.bool_()),
            pa.field("source_value_summary_sha256", pa.string()),
        ]
    )
    frame = pd.DataFrame(batch.lineage)
    for column in (
        "index_date",
        "window_start_date",
        "window_end_date",
        "earliest_source_date",
        "latest_source_date",
    ):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
    frame["source_record_count"] = pd.to_numeric(frame["source_record_count"]).astype("Int64")
    frame["contains_post_index_record"] = frame["contains_post_index_record"].astype("boolean")
    return pa.Table.from_pandas(frame, schema=schema, preserve_index=False)


class StreamingFeatureArtifactWriter:
    """Write feature artifacts without retaining all lineage rows in memory."""

    def __init__(self, output_dir: Path, cohort_manifest_path: Path | None = None) -> None:
        self.output_dir = output_dir
        self.cohort_manifest_path = cohort_manifest_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._matrix_writer: pq.ParquetWriter | None = None
        self._lineage_writer: pq.ParquetWriter | None = None
        self.registry: dict[str, Any] = {}
        self.row_count = 0
        self.patient_ids: set[object] = set()
        self.validation_passed = True
        self.validation_checks: list[dict[str, Any]] = []

    def write_batch(self, batch: FeatureDataset) -> None:
        if not batch.rows:
            return
        if not self.registry:
            self.registry = dict(batch.registry)
        matrix = _matrix_table(batch)
        lineage = _lineage_table(batch)
        if self._matrix_writer is None:
            self._matrix_writer = pq.ParquetWriter(
                self.output_dir / "feature_matrix.parquet", matrix.schema, compression="zstd"
            )
        if self._lineage_writer is None:
            self._lineage_writer = pq.ParquetWriter(
                self.output_dir / "feature_lineage.parquet", lineage.schema, compression="zstd"
            )
        self._matrix_writer.write_table(matrix)
        self._lineage_writer.write_table(lineage)
        self.row_count += len(batch.rows)
        self.patient_ids.update(row["patient_id"] for row in batch.rows)
        validation = validate_feature_dataset(batch, batch.lineage, batch.registry, {})
        self.validation_passed = self.validation_passed and validation.passed
        if not self.validation_checks:
            self.validation_checks = validation.to_dict()["checks"]

    def close(self) -> dict[str, object]:
        if self._matrix_writer is not None:
            self._matrix_writer.close()
        if self._lineage_writer is not None:
            self._lineage_writer.close()
        registry_payload = [item.to_dict() for item in self.registry.values()]
        registry_path = self.output_dir / "feature_definition_registry.json"
        registry_path.write_text(
            json.dumps(registry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        registry_sha = registry_hash(self.registry)
        (self.output_dir / "feature_definition_registry.sha256").write_text(
            registry_sha + "\n", encoding="utf-8"
        )
        validation_path = self.output_dir / "feature_validation_report.json"
        validation_path.write_text(
            json.dumps(
                {
                    "passed": self.validation_passed,
                    "checks": self.validation_checks,
                    "simulation_only": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        quality = {
            "feature_count": len(self.registry),
            "row_count": self.row_count,
            "unique_patient_count": len(self.patient_ids),
            "simulation_only": True,
            "warnings": [check for check in self.validation_checks if check["level"] == "warning"],
        }
        quality_path = self.output_dir / "feature_quality_report.json"
        quality_path.write_text(
            json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "feature_matrix_sha256": _sha256(self.output_dir / "feature_matrix.parquet"),
            "lineage_sha256": _sha256(self.output_dir / "feature_lineage.parquet"),
            "registry_sha256": registry_sha,
            "validation_report_sha256": _sha256(validation_path),
            "quality_report_sha256": _sha256(quality_path),
            "row_count": self.row_count,
            "patient_count": len(self.patient_ids),
            "feature_count": len(self.registry),
            "simulation_only": True,
            "split_status": "created",
            "feature_status": "created",
            "model_status": "not_created",
            "validation_passed": self.validation_passed,
        }
        (self.output_dir / "feature_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if self.cohort_manifest_path is not None and self.validation_passed:
            parent = json.loads(self.cohort_manifest_path.read_text(encoding="utf-8"))
            parent.update(
                {
                    "split_status": "created",
                    "feature_status": "created",
                    "model_status": "not_created",
                }
            )
            self.cohort_manifest_path.write_text(
                json.dumps(parent, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return manifest


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
