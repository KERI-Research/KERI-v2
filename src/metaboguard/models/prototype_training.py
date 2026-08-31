"""Bounded training for professor-approved, simulation-only prototypes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from metaboguard.models.ssl_encoder import (
    RobustPCAEncoder,
    RobustPCAEncoderConfig,
    SyntheticPrototypeAuthorization,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _atomic_write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, delete=False
    ) as temporary:
        pickle.dump(payload, temporary)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _prototype_model_card(summary: dict[str, object]) -> str:
    return f"""# MetaboGuard Synthetic Prototype Model Card

## Mandatory Disclaimer

This model was trained exclusively on a synthetic Synthea-derived dataset. It is a
simulation-only pipeline rehearsal artifact and must not be used for diagnosis,
clinical decision-making, patient screening, patient risk assessment, or medical
advice. Its outputs are not evidence of clinical performance, calibration,
generalizability, or utility. External validation on an approved real longitudinal
cohort is required before any clinical research-model use is considered.

## Scope

- Model type: label-free robust-PCA representation encoder.
- Source run: `{summary["source_run_id"]}`.
- Endpoint artifact: `{summary["endpoint_id"]}`.
- Training rows: {summary["training_row_count"]} pre-index feature rows.
- Latent dimensions: {summary["latent_dimensions"]}.
- Approval reference: `{summary["approval_reference"]}`.
- Source provenance discrepancy: {summary["source_provenance_discrepancy"]}.

## Outputs

The artifact emits latent representations and metabolic-deviation measures only.
It does not emit diagnoses, disease probabilities, future-risk estimates, or
clinical recommendations. No outcome labels are used for fitting or reported as
performance results.
"""


def train_synthetic_prototype(
    run_path: Path,
    endpoint_id: str,
    approval_reference: str,
    artifact_root: Path = Path("artifacts/model_prototypes"),
    config: RobustPCAEncoderConfig | None = None,
) -> dict[str, object]:
    """Train and persist one explicitly approved, isolated synthetic prototype."""
    authorization = SyntheticPrototypeAuthorization(approval_reference)
    run_manifest_path = run_path / "manifest.json"
    generation_manifest_path = run_path / "generation_manifest.json"
    feature_dir = run_path / "features" / endpoint_id
    feature_matrix_path = feature_dir / "feature_matrix.parquet"
    feature_manifest_path = feature_dir / "feature_manifest.json"
    run_manifest = _load_json(run_manifest_path)
    generation_manifest = _load_json(generation_manifest_path)
    feature_manifest = _load_json(feature_manifest_path)
    synthea_evidence_complete = (
        generation_manifest.get("state") == "complete"
        and bool(generation_manifest.get("synthea_version"))
        and bool(generation_manifest.get("jar_sha256"))
        and bool(generation_manifest.get("canonical_dataset_sha256"))
    )
    run_is_synthetic = bool(run_manifest.get("simulation_only", False))
    if not run_is_synthetic and not synthea_evidence_complete:
        raise ValueError(
            "prototype training requires a simulation-only run manifest or "
            "completed Synthea generation evidence"
        )
    if run_manifest.get("status") not in {
        "completed",
        "completed_not_ready",
        "completed_prototype_ready",
    }:
        raise ValueError("prototype training requires a completed production run")
    if not bool(feature_manifest.get("validation_passed", False)):
        raise ValueError("prototype training requires a validated feature manifest")
    matrix = pd.read_parquet(feature_matrix_path)
    training = matrix[matrix["split"].eq("train")].copy()
    if training.empty:
        raise ValueError("prototype training requires at least one train row")
    encoder = RobustPCAEncoder(config).fit_synthetic_prototype(training, authorization)
    deviation = encoder.score(training)
    output_dir = artifact_root / str(run_manifest["run_id"]) / endpoint_id
    summary: dict[str, object] = {
        "artifact_version": "1.0.0",
        "artifact_kind": "simulation_only_unsupervised_prototype",
        "simulation_only": True,
        "pipeline_rehearsal_only": True,
        "clinical_use_prohibited": True,
        "source_run_id": str(run_manifest["run_id"]),
        "source_run_manifest_sha256": _sha256(run_manifest_path),
        "source_generation_manifest_sha256": _sha256(generation_manifest_path),
        "source_feature_matrix_sha256": _sha256(feature_matrix_path),
        "source_feature_manifest_sha256": _sha256(feature_manifest_path),
        "source_provenance_discrepancy": not run_is_synthetic
        or not bool(generation_manifest.get("simulation_only", False))
        or not bool(feature_manifest.get("simulation_only", False)),
        "endpoint_id": endpoint_id,
        "approval_reference": authorization.approval_reference,
        "training_row_count": len(training),
        "latent_dimensions": encoder.config.latent_dimensions,
        "feature_column_count": len(encoder._feature_columns),
        "mean_reconstruction_error": float(np.mean(deviation.reconstruction_error)),
        "mean_latent_distance": float(np.mean(deviation.latent_distance)),
    }
    _atomic_write_pickle(output_dir / "prototype_encoder.pkl", encoder)
    _atomic_write_text(
        output_dir / "training_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(output_dir / "MODEL_CARD.md", _prototype_model_card(summary))
    return summary


def main() -> int:
    """Train one approved synthetic prototype from a completed production run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", type=Path)
    parser.add_argument("endpoint_id")
    parser.add_argument("--approval-reference", required=True)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/model_prototypes")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            train_synthetic_prototype(
                args.run_path,
                args.endpoint_id,
                args.approval_reference,
                args.artifact_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0
