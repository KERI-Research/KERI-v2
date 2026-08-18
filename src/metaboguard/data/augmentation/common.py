"""Shared deterministic augmentation mechanics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from metaboguard.data.canonical import CanonicalDataset
from metaboguard.data.schema import ClinicalEvent
from metaboguard.features.dictionary import get_feature_definition


@dataclass(frozen=True, slots=True)
class AugmentationResult:
    """Augmented dataset plus versioned assumptions metadata."""

    dataset: CanonicalDataset
    module_name: str
    module_version: str
    assumptions: dict[str, object]
    assumptions_sha256: str


def assumptions_digest(assumptions: dict[str, object]) -> str:
    payload = json.dumps(assumptions, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_assumptions(result: AugmentationResult, output_dir: Path) -> Path:
    """Write one machine-readable assumptions file and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.module_name}_assumptions.json"
    path.write_text(
        json.dumps(result.assumptions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def augment_events(
    dataset: CanonicalDataset,
    rng: np.random.Generator,
    module_name: str,
    module_version: str,
    feature_name: str,
    seed: int,
    value_function: Callable[[ClinicalEvent, float, np.random.Generator], float],
    assumptions: dict[str, object],
) -> AugmentationResult:
    """Create generated events from pre-existing event rows only."""
    definition = get_feature_definition(feature_name)
    source_events = sorted(
        (event for event in dataset.events if event.provenance == "synthea_native"),
        key=lambda event: (event.patient_id, event.event_date, event.feature_name),
    )
    generated: list[ClinicalEvent] = []
    for source in source_events:
        if source.value is None:
            continue
        value = value_function(source, source.value, rng)
        generated.append(
            ClinicalEvent(
                patient_id=source.patient_id,
                event_date=source.event_date,
                age_at_event=source.age_at_event,
                encounter_type=source.encounter_type,
                feature_name=feature_name,
                value=max(definition.plausible_min, min(definition.plausible_max, value)),
                unit=definition.canonical_unit,
                is_missing=False,
                provenance="augmented",
                augmentation_module=module_name,
            )
        )
    all_events = sorted(
        [*dataset.events, *generated],
        key=lambda event: (
            event.patient_id,
            event.event_date,
            event.feature_name,
            event.provenance,
        ),
    )
    assumptions = {
        "module_name": module_name,
        "module_version": module_version,
        "causal_assumptions": assumptions["causal_assumptions"],
        "input_features": assumptions["input_features"],
        "noise_distribution": assumptions["noise_distribution"],
        "clipping_range_policy": assumptions["clipping_range_policy"],
        "seed": seed,
        "known_limitations": assumptions["known_limitations"],
    }
    return AugmentationResult(
        dataset=CanonicalDataset(
            patients=dataset.patients,
            events=all_events,
            conditions=dataset.conditions,
            outcomes=dataset.outcomes,
            dropped_loinc_codes=dataset.dropped_loinc_codes,
            output_dir=dataset.output_dir,
            dataset_sha256="",
            cohort_class=dataset.cohort_class,
            cohort_metadata=dataset.cohort_metadata,
            date_normalisation_audit=dataset.date_normalisation_audit,
            date_normalisation_audit_sha256=dataset.date_normalisation_audit_sha256,
        ),
        module_name=module_name,
        module_version=module_version,
        assumptions=assumptions,
        assumptions_sha256=assumptions_digest(assumptions),
    )
