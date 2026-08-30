"""Label-count inventory for independent horizons."""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from metaboguard.readiness.contracts import LabelFeasibilityRecord


def build_label_feasibility(run_path: Path, endpoint_id: str) -> list[LabelFeasibilityRecord]:
    """Count frozen labels without constructing targets or feature inputs."""
    cohort = run_path / "cohort" / endpoint_id
    split_manifest = pd.read_parquet(cohort / "splits" / "split_assignments.parquet")
    label_records: list[LabelFeasibilityRecord] = []
    for horizon in (1, 3, 5):
        path = cohort / f"horizon_labels_{horizon}y.parquet"
        if not path.exists():
            continue
        labels = pd.read_parquet(path)
        assignments = split_manifest.copy()
        if "patient_fingerprint" in assignments.columns:
            # Internal readiness counts use the frozen assignment manifest, not raw IDs in reports.
            assignments = assignments.iloc[0:0]
        split = "all"
        counts = labels["label_state"].value_counts().to_dict()
        positive = int(counts.get("positive", 0))
        negative = int(counts.get("eligible_negative", 0))
        label_records.append(
            LabelFeasibilityRecord(
                cohort_class=str(labels["cohort_class"].iloc[0]) if len(labels) else "",
                endpoint_id=endpoint_id,
                horizon_years=horizon,
                split=split,
                total_labelled_indexes=len(labels),
                unique_patient_count=(int(labels["patient_id"].nunique()) if len(labels) else 0),
                positive_count=positive,
                eligible_negative_count=negative,
                censored_count=int(counts.get("censored", 0)),
                competing_death_count=int(counts.get("competing_death", 0)),
                excluded_count=int(counts.get("excluded", 0)),
                evaluable_prevalence=(
                    positive / (positive + negative) if positive + negative else None
                ),
            )
        )
    return label_records
