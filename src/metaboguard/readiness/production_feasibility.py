"""Model-free endpoint and horizon feasibility reporting for Step 7."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict

from metaboguard.config import load_config
from metaboguard.readiness.inventory import inspect_artifact_inventory
from metaboguard.readiness.leakage import audit_feature_leakage
from metaboguard.readiness.splits import build_split_readiness


class ProductionFeasibilityRow(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    cohort_class: str
    run_id: str
    endpoint_id: str
    horizon_years: int
    simulation_only: bool
    pipeline_rehearsal_only: bool
    population_target: int
    generated_patient_count: int
    eligible_index_count: int
    feature_row_count: int
    feature_build_status: str
    positive_count: int
    eligible_negative_count: int
    censored_count: int
    competing_death_count: int
    excluded_count: int
    train_positive_count: int
    validation_positive_count: int
    test_positive_count: int
    temporal_holdout_positive_count: int
    minimum_events_required: int
    minimum_eligible_non_events_required: int
    minimum_partition_events_required: int
    event_count_gate_passed: bool
    eligible_non_event_gate_passed: bool
    partition_event_gate_passed: bool
    feature_completeness_gate_passed: bool
    leakage_gate_passed: bool
    split_integrity_gate_passed: bool
    readiness_decision: str
    decision_reasons: list[str]
    mechanical_pipeline_capacity: str


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _split_positive_counts(rows: list[dict[str, Any]], horizon: int) -> dict[str, int]:
    return {
        split: sum(
            int(row["positive_count"])
            for row in rows
            if int(row["horizon_years"]) == horizon and row["split"] == split
        )
        for split in ("train", "validation", "test", "temporal_holdout")
    }


def build_production_feasibility(
    run_path: Path,
    endpoint_id: str,
    population_target: int,
    run_id: str | None = None,
) -> list[ProductionFeasibilityRow]:
    """Build independent endpoint/horizon feasibility rows without model metrics."""
    manifest = _read_json(run_path / "manifest.json")
    capability = _read_json(
        run_path / "readiness" / endpoint_id / "capability_report.json"
    )
    inventory = inspect_artifact_inventory(run_path, endpoint_id)
    leakage = audit_feature_leakage(run_path, endpoint_id)
    split_rows = [
        row.model_dump() for row in build_split_readiness(run_path, endpoint_id)
    ]
    thresholds = load_config()["readiness"]
    minimum_events = int(thresholds["minimum_events_per_horizon"])
    minimum_nonevents = int(thresholds["minimum_eligible_non_events_per_horizon"])
    minimum_partition = int(thresholds["minimum_events_per_evaluation_partition"])
    simulation_only = bool(manifest.get("simulation_only", True))
    pipeline_rehearsal_only = bool(
        manifest.get("pipeline_rehearsal_only", simulation_only)
    )
    rows: list[ProductionFeasibilityRow] = []
    for decision in capability["horizon_decisions"]:
        horizon = int(decision["horizon_years"])
        split_positive = _split_positive_counts(split_rows, horizon)
        event_passed = int(decision["event_count"]) >= minimum_events
        nonevent_passed = int(decision["eligible_negative_count"]) >= minimum_nonevents
        partition_passed = all(
            split_positive[split] >= minimum_partition
            for split in ("test", "temporal_holdout")
        )
        feature_passed = inventory.feature_build_status == "complete"
        rows.append(
            ProductionFeasibilityRow(
                cohort_class=str(manifest["cohort_class"]),
                run_id=run_id or str(manifest.get("run_id", run_path.name)),
                endpoint_id=endpoint_id,
                horizon_years=horizon,
                simulation_only=simulation_only,
                pipeline_rehearsal_only=pipeline_rehearsal_only,
                population_target=population_target,
                generated_patient_count=int(manifest.get("generated_patient_count", 0)),
                eligible_index_count=inventory.expected_eligible_index_count,
                feature_row_count=inventory.feature_row_count,
                feature_build_status=inventory.feature_build_status,
                positive_count=int(decision["event_count"]),
                eligible_negative_count=int(decision["eligible_negative_count"]),
                censored_count=int(decision["censored_count"]),
                competing_death_count=int(decision["competing_death_count"]),
                excluded_count=0,
                train_positive_count=split_positive["train"],
                validation_positive_count=split_positive["validation"],
                test_positive_count=split_positive["test"],
                temporal_holdout_positive_count=split_positive["temporal_holdout"],
                minimum_events_required=minimum_events,
                minimum_eligible_non_events_required=minimum_nonevents,
                minimum_partition_events_required=minimum_partition,
                event_count_gate_passed=event_passed,
                eligible_non_event_gate_passed=nonevent_passed,
                partition_event_gate_passed=partition_passed,
                feature_completeness_gate_passed=feature_passed,
                leakage_gate_passed=leakage.passed,
                split_integrity_gate_passed=decision["split_integrity_status"]
                == "passed",
                readiness_decision=str(decision["decision"]),
                decision_reasons=list(decision["decision_reasons"]),
                mechanical_pipeline_capacity=(
                    "sufficient"
                    if event_passed
                    and nonevent_passed
                    and partition_passed
                    and feature_passed
                    else "insufficient"
                ),
            )
        )
    return rows


def write_production_feasibility(
    run_path: Path,
    rows: list[ProductionFeasibilityRow],
) -> None:
    """Write the required feasibility parquet and synthetic-only report."""
    output = run_path / "feasibility"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row.model_dump() for row in rows]).to_parquet(
        output / "endpoint_feasibility.parquet", index=False
    )
    report = {
        "report_version": "1.0.0",
        "cohort_class": rows[0].cohort_class if rows else "",
        "run_id": rows[0].run_id if rows else run_path.name,
        "simulation_only": rows[0].simulation_only if rows else True,
        "pipeline_rehearsal_only": rows[0].pipeline_rehearsal_only if rows else True,
        "prototype_modeling_authorized": rows[0].simulation_only if rows else True,
        "clinical_model_research_authorized": (
            not rows[0].simulation_only if rows else False
        ),
        "model_status": "not_created",
        "rows": [row.model_dump() for row in rows],
        "claim_limitation": (
            "These are synthetic prototype feasibility results. They do not measure clinical "
            "prevalence, predictive performance, calibration, discrimination, clinical utility, "
            "or patient-level risk."
            if rows and rows[0].simulation_only
            else "Clinical claims require approved data and external validation."
        ),
    }
    (output / "endpoint_feasibility_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
