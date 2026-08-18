"""Split-level readiness summaries and isolation checks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from metaboguard.readiness.contracts import SplitReadinessRecord
from metaboguard.readiness.inventory import feature_artifact_dir


def build_split_readiness(run_path: Path, endpoint_id: str) -> list[SplitReadinessRecord]:
    cohort = run_path / "cohort" / endpoint_id
    split_manifest = json.loads(
        (cohort / "splits" / "split_manifest.json").read_text(encoding="utf-8")
    )
    indexes = pd.read_parquet(cohort / "eligible_indexes.parquet")
    feature_count = len(
        pd.read_parquet(feature_artifact_dir(run_path, endpoint_id) / "feature_matrix.parquet")
    )
    records: list[SplitReadinessRecord] = []
    for horizon in (1, 3, 5):
        path = cohort / f"horizon_labels_{horizon}y.parquet"
        if not path.exists():
            continue
        labels = pd.read_parquet(path)
        for split in sorted(set(split_manifest["assignments"].values())):
            patient_ids = [
                pid
                for pid, assignment in split_manifest["assignments"].items()
                if assignment == split
            ]
            subset = labels[labels["patient_id"].isin(patient_ids)]
            dates = pd.to_datetime(indexes[indexes["patient_id"].isin(patient_ids)]["index_date"])
            records.append(
                SplitReadinessRecord(
                    cohort_class=str(labels["cohort_class"].iloc[0]) if len(labels) else "",
                    endpoint_id=endpoint_id,
                    horizon_years=horizon,
                    split=split,
                    patient_count=len(patient_ids),
                    index_count=len(subset),
                    positive_count=int((subset["label_state"] == "positive").sum()),
                    eligible_negative_count=int(
                        (subset["label_state"] == "eligible_negative").sum()
                    ),
                    censored_count=int((subset["label_state"] == "censored").sum()),
                    competing_death_count=int((subset["label_state"] == "competing_death").sum()),
                    first_index_date=dates.min().date() if len(dates) else None,
                    last_index_date=dates.max().date() if len(dates) else None,
                    feature_complete_fraction=float(feature_count / len(indexes))
                    if len(indexes)
                    else 0.0,
                    reaches_evaluation_event_threshold=int(
                        (subset["label_state"] == "positive").sum()
                    )
                    >= 30,
                )
            )
    return records
