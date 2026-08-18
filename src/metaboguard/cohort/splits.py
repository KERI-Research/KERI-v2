"""Patient-level and quantile-based temporal split construction."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from metaboguard.cohort.protocol import ConstructedCohort, SplitConfig, SplitManifest
from metaboguard.data.manifests import assert_same_cohort_class


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_splits(cohort: ConstructedCohort, split_config: SplitConfig) -> SplitManifest:
    """Assign all indexes for each patient to one deterministic split."""
    assert_same_cohort_class([cohort.cohort_class])
    patient_ids = sorted({record.patient_id for record in cohort.patient_indexes})
    randomizer = random.Random(split_config.root_seed)
    shuffled = patient_ids.copy()
    randomizer.shuffle(shuffled)
    train_end = int(len(shuffled) * split_config.train_fraction)
    validation_end = train_end + int(len(shuffled) * split_config.validation_fraction)
    assignments = {
        patient_id: (
            "train" if index < train_end else "validation" if index < validation_end else "test"
        )
        for index, patient_id in enumerate(shuffled)
    }
    dates = sorted(record.index_date for record in cohort.patient_indexes)
    quantile_cutoff = (
        dates[max(0, int(len(dates) * (1 - split_config.temporal_holdout_cap)) - 1)]
        if dates
        else None
    )
    cutoff = split_config.fixed_date
    fallback = False
    over_cap = False
    if dates and cutoff is not None:  # pragma: no branch
        over_cap = sum(value >= cutoff for value in dates) > int(
            len(patient_ids) * split_config.temporal_holdout_cap
        )
    if cutoff is None:
        cutoff = quantile_cutoff
    elif over_cap:  # pragma: no branch
        cutoff = quantile_cutoff
        fallback = True
    if cutoff is not None:  # pragma: no branch
        for patient_id in patient_ids:
            patient_dates = [
                record.index_date
                for record in cohort.patient_indexes
                if record.patient_id == patient_id
            ]
            if any(value >= cutoff for value in patient_dates):
                assignments[patient_id] = "temporal_holdout"
    label_counts: dict[str, dict[str, int]] = {}
    for label in cohort.labels:
        split = assignments.get(label.patient_id, "unassigned")
        label_counts.setdefault(split, {})[label.label_state] = (
            label_counts.setdefault(split, {}).get(label.label_state, 0) + 1
        )
    return SplitManifest(
        cohort_class=cohort.cohort_class,
        endpoint_id=cohort.endpoint.endpoint_id,
        root_seed=split_config.root_seed,
        split_rule="patient_level_65_15_20_with_quantile_temporal_holdout",
        temporal_cutoff=cutoff,
        quantile_fallback_used=fallback,
        assignments=assignments,
        patient_fingerprints={patient_id: _fingerprint(patient_id) for patient_id in patient_ids},
        counts={"patients": len(patient_ids), "labels_by_split": label_counts},
        cohort_hash=cohort.source_canonical_sha256,
        protocol_hash=hashlib.sha256(
            json.dumps(cohort.endpoint.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    )


def write_split_artifacts(split: SplitManifest, labels: list[object], output_dir: Path) -> None:
    """Write split assignments and a fingerprint-only manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"patient_fingerprint": split.patient_fingerprints[patient_id], "split": assignment}
        for patient_id, assignment in sorted(split.assignments.items())
    ]
    pd.DataFrame(rows).to_parquet(output_dir / "split_assignments.parquet", index=False)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(
            {**asdict(split), "temporal_cutoff": str(split.temporal_cutoff)},
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    cohort_manifest_path = output_dir.parent / "cohort_manifest.json"
    if cohort_manifest_path.exists():
        cohort_manifest = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
        cohort_manifest["split_status"] = "created"
        cohort_manifest["feature_status"] = "not_created"
        cohort_manifest["model_status"] = "not_created"
        cohort_manifest_path.write_text(
            json.dumps(cohort_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
