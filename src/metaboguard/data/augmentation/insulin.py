"""Deterministic insulin augmentation from pre-existing metabolic events."""

from __future__ import annotations

import numpy as np

from metaboguard.data.augmentation.common import AugmentationResult, augment_events
from metaboguard.data.canonical import CanonicalDataset

MODULE_NAME = "insulin"
MODULE_VERSION = "1.0.0"


def augment_insulin(
    dataset: CanonicalDataset, rng: np.random.Generator, seed: int
) -> AugmentationResult:
    """Generate weakly correlated insulin values without outcome inputs."""
    return augment_events(
        dataset,
        rng,
        MODULE_NAME,
        MODULE_VERSION,
        "insulin",
        seed,
        lambda _event, value, generator: 0.08 * value + generator.normal(4.0, 2.0),
        {
            "causal_assumptions": "Weak association with the measured metabolic state only.",
            "input_features": ["source_event_value"],
            "noise_distribution": "Normal(mean=4, sd=2), seeded generator",
            "clipping_range_policy": "Clip to the canonical dictionary range [0, 1000].",
            "known_limitations": "Not an observed measurement and not a physiological simulator.",
        },
    )
