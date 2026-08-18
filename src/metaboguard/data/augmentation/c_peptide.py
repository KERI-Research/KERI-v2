"""Deterministic C-peptide augmentation from pre-existing metabolic events."""

from __future__ import annotations

import numpy as np

from metaboguard.data.augmentation.common import AugmentationResult, augment_events
from metaboguard.data.canonical import CanonicalDataset

MODULE_NAME = "c_peptide"
MODULE_VERSION = "1.0.0"


def augment_c_peptide(
    dataset: CanonicalDataset, rng: np.random.Generator, seed: int
) -> AugmentationResult:
    """Generate weakly correlated C-peptide values without outcome inputs."""
    return augment_events(
        dataset,
        rng,
        MODULE_NAME,
        MODULE_VERSION,
        "c_peptide",
        seed,
        lambda _event, value, generator: 0.002 * value + generator.normal(1.2, 0.35),
        {
            "causal_assumptions": "Weak association with the measured metabolic state only.",
            "input_features": ["source_event_value"],
            "noise_distribution": "Normal(mean=1.2, sd=0.35), seeded generator",
            "clipping_range_policy": "Clip to the canonical dictionary range [0, 50].",
            "known_limitations": "Not an observed measurement and not a physiological simulator.",
        },
    )
