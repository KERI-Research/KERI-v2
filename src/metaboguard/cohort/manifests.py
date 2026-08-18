"""Cohort artifact writing and immutable construction orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from metaboguard.cohort.endpoints import assign_outcomes
from metaboguard.cohort.index_dates import generate_rolling_index_dates
from metaboguard.cohort.protocol import ConstructedCohort, EndpointProtocol
from metaboguard.cohort.validation import validate_constructed_cohort
from metaboguard.data.canonical import CanonicalDataset
from metaboguard.data.manifests import config_sha256


def _hash_parquet_files(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.parquet")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_rows(rows: Sequence[Any], path: Path) -> None:
    frame = pd.DataFrame([asdict(row) for row in rows])
    if frame.empty:
        frame = pd.DataFrame({"patient_id": pd.Series(dtype="string")})
    if not frame.empty:
        frame = frame.sort_values(list(frame.columns), kind="mergesort").reset_index(drop=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def construct_endpoint_cohort(
    dataset: CanonicalDataset,
    endpoint: EndpointProtocol,
    output_dir: Path,
    source_generation_manifest_sha256: str = "",
) -> ConstructedCohort:
    """Construct and persist one endpoint-specific cohort without features."""
    indexes = generate_rolling_index_dates(dataset, endpoint)
    labels = assign_outcomes(indexes, dataset, endpoint)
    cohort = ConstructedCohort(
        cohort_class=dataset.cohort_class or "",
        endpoint=endpoint,
        patient_indexes=indexes,
        excluded_indexes=[],
        labels=labels,
        source_canonical_sha256=dataset.dataset_sha256,
        source_generation_manifest_sha256=source_generation_manifest_sha256,
        date_normalisation_audit_sha256=dataset.date_normalisation_audit_sha256,
        output_dir=output_dir,
    )
    report = validate_constructed_cohort(cohort, dataset, endpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "endpoint_protocol.json").write_text(
        json.dumps(endpoint.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_rows(indexes, output_dir / "eligible_indexes.parquet")
    _write_rows([], output_dir / "excluded_indexes.parquet")
    for horizon in endpoint.horizon_years:
        _write_rows(
            [label for label in labels if label.horizon_years == horizon],
            output_dir / f"horizon_labels_{horizon}y.parquet",
        )
    state_counts: dict[str, dict[str, int]] = {}
    for label in labels:
        state_counts.setdefault(str(label.horizon_years), {})[label.label_state] = (
            state_counts.setdefault(str(label.horizon_years), {}).get(label.label_state, 0) + 1
        )
    summary = {
        "cohort_class": cohort.cohort_class,
        "endpoint_id": endpoint.endpoint_id,
        "patient_count": len({record.patient_id for record in indexes}),
        "index_count": len(indexes),
        "excluded_index_count": 0,
        "label_state_counts_by_horizon": state_counts,
        "validation_warnings": [asdict(warning) for warning in report.warnings],
        "simulation_only": True,
        "split_status": "not_created",
        "feature_status": "not_created",
        "model_status": "not_created",
    }
    (output_dir / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "washout_impact.json").write_text(
        json.dumps(
            {
                "washout_days": endpoint.washout_days,
                "excluded_indexes": 0,
                "excluded_endpoint_events": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "cohort_validation_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "source_generation_manifest_sha256": source_generation_manifest_sha256,
        "source_canonical_data_sha256": dataset.dataset_sha256,
        "date_normalisation_audit_sha256": dataset.date_normalisation_audit_sha256,
        "endpoint_protocol_sha256": config_sha256(endpoint.to_dict()),
        "endpoint_definition_version": endpoint.definition_version,
        "cohort_class": cohort.cohort_class,
        "patient_count": summary["patient_count"],
        "index_count": len(indexes),
        "excluded_index_count": 0,
        "label_state_counts_by_horizon": state_counts,
        "washout_impact": {"washout_days": endpoint.washout_days, "excluded_indexes": 0},
        "root_seed": 0,
        "configuration_hash": config_sha256(endpoint.to_dict()),
        "git_sha": "unavailable",
        "simulation_only": True,
        "split_status": "not_created",
        "feature_status": "not_created",
        "model_status": "not_created",
        "parquet_sha256": _hash_parquet_files(output_dir),
        "validation_passed": report.passed,
    }
    (output_dir / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return cohort
