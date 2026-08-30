"""Label-blind deterministic feature extraction and lineage."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from metaboguard.cohort.protocol import PatientIndex
from metaboguard.config import load_config
from metaboguard.data.canonical import CanonicalDataset
from metaboguard.data.manifests import assert_same_cohort_class
from metaboguard.features.definitions import (
    EngineeredFeatureDefinition,
    build_feature_registry,
)
from metaboguard.features.missingness import missingness_flags
from metaboguard.features.trajectories import change, summaries, trajectory
from metaboguard.features.windows import select_preindex_events

IDENTIFIER_COLUMNS = (
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
)


@dataclass(frozen=True, slots=True)
class FeatureDataset:
    rows: list[dict[str, object]]
    lineage: list[dict[str, object]]
    registry: dict[str, EngineeredFeatureDefinition]
    feature_definition_version: str
    simulation_only: bool = True


def _feature_values(events: list[Any], feature: str) -> list[tuple[date, float]]:
    return sorted(
        [
            (event.event_date, float(event.value))
            for event in events
            if event.feature_name == feature and event.value is not None
        ],
        key=lambda item: item[0],
    )


def extract_features(
    dataset: CanonicalDataset,
    indexes: list[PatientIndex],
    split_assignments: dict[str, str],
    source_cohort_manifest_sha256: str = "",
    source_split_manifest_sha256: str = "",
    batch_size: int | None = None,
    batch_callback: Callable[[FeatureDataset], None] | None = None,
) -> FeatureDataset:
    """Build features, optionally flushing bounded batches to a callback.

    The default returns the complete in-memory dataset for compatibility. A callback
    bounds memory for large cohorts; flushed rows are not retained in the result.
    """
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    cohort_class = assert_same_cohort_class([record.cohort_class for record in indexes])
    registry = build_feature_registry()
    patients = {patient.patient_id: patient for patient in dataset.patients}
    windows = {
        definition.window_days
        for definition in registry.values()
        if definition.window_days is not None
    }
    window_names = {
        days: name for name, days in load_config()["features"]["feature_windows_days"].items()
    }
    rows: list[dict[str, object]] = []
    lineage: list[dict[str, object]] = []
    events_by_patient: dict[str, list[Any]] = {}
    for event in dataset.events:
        events_by_patient.setdefault(event.patient_id, []).append(event)
    for index in sorted(indexes, key=lambda record: (record.patient_id, record.index_date)):
        events = [
            event
            for event in events_by_patient.get(index.patient_id, [])
            if event.event_date <= index.index_date
        ]
        source_events_cache: dict[tuple[str, int | None], list[Any]] = {}
        values_cache: dict[tuple[str, int | None], list[tuple[date, float]]] = {}
        row: dict[str, object] = {
            "patient_id": index.patient_id,
            "cohort_class": cohort_class,
            "endpoint_id": index.endpoint_id,
            "index_date": index.index_date,
            "split": split_assignments[index.patient_id],
            "index_sequence_number": index.index_sequence_number,
            "feature_definition_version": "1.0.0",
            "feature_window_config_sha256": hashlib.sha256(
                str(sorted(windows)).encode()
            ).hexdigest(),
            "source_canonical_manifest_sha256": dataset.dataset_sha256,
            "source_cohort_manifest_sha256": source_cohort_manifest_sha256,
            "source_split_manifest_sha256": source_split_manifest_sha256,
            "simulation_only": True,
        }
        for definition in registry.values():
            cache_key = (definition.source_feature_id, definition.window_days)
            if cache_key not in source_events_cache:
                source_events_cache[cache_key] = select_preindex_events(
                    events,
                    index.index_date,
                    definition.window_days,
                    patients[index.patient_id].birth_date,
                )
                values_cache[cache_key] = _feature_values(
                    source_events_cache[cache_key], definition.source_feature_id
                )
            source_events = source_events_cache[cache_key]
            values = values_cache[cache_key]
            metric_part = definition.feature_id.split("__", 1)[1]
            window_suffix = (
                "lifetime"
                if definition.window_days is None
                else window_names[definition.window_days]
            )
            metric = metric_part.removesuffix(f"_{window_suffix}")
            value: object
            if definition.feature_family == "latest":
                latest = values[-1] if values else None
                if metric == "latest":
                    value = latest[1] if latest else None
                elif metric == "latest_date_offset_days":
                    value = (index.index_date - latest[0]).days if latest else None
                else:
                    value = (
                        (
                            "lifetime"
                            if definition.window_days is None
                            else str(definition.window_days)
                        )
                        if latest
                        else None
                    )
            elif definition.feature_family == "summary":
                summary = summaries(values)
                value = len(values) if metric == "count" else summary.get(metric)
            elif definition.feature_family == "change":
                value = change(values).get(metric)
            elif definition.feature_family == "trajectory":
                value = trajectory(values).get(metric)
            elif definition.feature_family == "missingness":
                value = missingness_flags(
                    events,
                    definition.source_feature_id,
                    index.index_date,
                    patients[index.patient_id].birth_date,
                ).get(metric)
            else:
                value = None  # pragma: no cover
            row[definition.feature_id] = value
            lineage.append(
                {
                    "patient_id": index.patient_id,
                    "endpoint_id": index.endpoint_id,
                    "index_date": index.index_date,
                    "feature_id": definition.feature_id,
                    "source_feature_id": definition.source_feature_id,
                    "window_name": (
                        "lifetime"
                        if definition.window_days is None
                        else str(definition.window_days)
                    ),
                    "window_start_date": min(
                        (event.event_date for event in source_events), default=None
                    ),
                    "window_end_date": index.index_date,
                    "source_record_count": len(source_events),
                    "earliest_source_date": min(
                        (event.event_date for event in source_events), default=None
                    ),
                    "latest_source_date": max(
                        (event.event_date for event in source_events), default=None
                    ),
                    "contains_post_index_record": any(
                        event.event_date > index.index_date for event in source_events
                    ),
                    "source_value_summary_sha256": hashlib.sha256(str(values).encode()).hexdigest(),
                }
            )
        rows.append(row)
        if batch_callback is not None and batch_size is not None and len(rows) >= batch_size:
            batch_callback(
                FeatureDataset(
                    rows,
                    lineage,
                    registry,
                    "1.0.0",
                    simulation_only=dataset.cohort_metadata.get("simulation_only", True)
                    is not False,
                )
            )
            rows = []
            lineage = []
    if batch_callback is not None and rows:
        batch_callback(
            FeatureDataset(
                rows,
                lineage,
                registry,
                "1.0.0",
                simulation_only=dataset.cohort_metadata.get("simulation_only", True) is not False,
            )
        )
        rows = []
        lineage = []
    return FeatureDataset(rows, lineage, registry, "1.0.0")
