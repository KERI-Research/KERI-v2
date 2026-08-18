"""Readiness artifact writer and new readiness-only manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from metaboguard.readiness.capability import build_capability_decisions
from metaboguard.readiness.features import build_feature_availability
from metaboguard.readiness.inventory import inspect_artifact_inventory
from metaboguard.readiness.labels import build_label_feasibility
from metaboguard.readiness.leakage import audit_feature_leakage
from metaboguard.readiness.splits import build_split_readiness
from metaboguard.readiness.validation import validate_readiness_bundle


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_readiness_bundle(run_path: Path, endpoint_id: str) -> dict[str, object]:
    """Inspect and write one endpoint readiness bundle without mutating inputs."""
    out = run_path / "readiness" / endpoint_id
    out.mkdir(parents=True, exist_ok=True)
    inventory = inspect_artifact_inventory(run_path, endpoint_id)
    labels = build_label_feasibility(run_path, endpoint_id)
    features = build_feature_availability(run_path, endpoint_id)
    splits = build_split_readiness(run_path, endpoint_id)
    leakage = audit_feature_leakage(run_path, endpoint_id)
    decisions = build_capability_decisions(inventory, labels, features, splits, leakage)
    validation = validate_readiness_bundle(inventory, labels, features, splits, leakage, decisions)
    for name, rows in (
        ("label_feasibility", labels),
        ("feature_availability", features),
        ("split_readiness", splits),
    ):
        pd.DataFrame([row.model_dump() for row in rows]).to_parquet(
            out / f"{name}.parquet", index=False
        )
    inventory_path = out / "artifact_inventory.json"
    inventory_path.write_text(inventory.model_dump_json(indent=2) + "\n", encoding="utf-8")
    leakage_path = out / "leakage_readiness.json"
    leakage_path.write_text(leakage.model_dump_json(indent=2) + "\n", encoding="utf-8")
    report = {
        "report_version": "1.0.0",
        "generated_at": "2026-08-16",
        "cohort_class": inventory.cohort_class,
        "endpoint_id": endpoint_id,
        "simulation_only": inventory.simulation_only,
        "feature_build_status": inventory.feature_build_status,
        "overall_decision": "blocked"
        if any(item.decision == "blocked" for item in decisions)
        else "not_eligible",
        "overall_reasons": sorted(
            {reason for item in decisions for reason in item.decision_reasons}
        ),
        "pipeline_rehearsal_only": inventory.simulation_only,
        "horizon_decisions": [item.model_dump() for item in decisions],
        "readiness_validation": validation.model_dump(),
    }
    report_path = out / "capability_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation_path = out / "readiness_validation_report.json"
    validation_path.write_text(validation.model_dump_json(indent=2) + "\n", encoding="utf-8")
    manifest = {
        "report_version": "1.0.0",
        "cohort_class": inventory.cohort_class,
        "endpoint_id": endpoint_id,
        "readiness_status": "created",
        "readiness_decision": report["overall_decision"],
        "feature_build_status": inventory.feature_build_status,
        "artifact_inventory_sha256": _sha(inventory_path),
        "label_feasibility_sha256": _sha(out / "label_feasibility.parquet"),
        "feature_availability_sha256": _sha(out / "feature_availability.parquet"),
        "split_readiness_sha256": _sha(out / "split_readiness.parquet"),
        "leakage_readiness_sha256": _sha(leakage_path),
        "readiness_validation_report_sha256": _sha(validation_path),
        "simulation_only": True,
    }
    (out / "readiness_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
