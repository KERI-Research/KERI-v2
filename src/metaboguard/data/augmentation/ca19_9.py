"""Deterministic CA 19-9 augmentation with overlapping weak distributions."""

from __future__ import annotations

import numpy as np

from metaboguard.data.augmentation.common import AugmentationResult, augment_events
from metaboguard.data.canonical import CanonicalDataset

MODULE_NAME = "ca19_9"
MODULE_VERSION = "1.0.0"


def augment_ca19_9(
    dataset: CanonicalDataset, rng: np.random.Generator, seed: int
) -> AugmentationResult:
    """Generate weakly variable CA 19-9 values without cancer labels or outcomes."""
    return augment_events(
        dataset,
        rng,
        MODULE_NAME,
        MODULE_VERSION,
        "ca_19_9",
        seed,
        lambda _event, value, generator: 18.0
        + 0.01 * value
        + generator.lognormal(0.0, 0.7),
        {
            "causal_assumptions": (
                "Overlapping weak distributions independent of cancer labels and sites."
            ),
            "input_features": ["source_event_value"],
            "noise_distribution": "Lognormal(meanlog=0, sdlog=0.7), seeded generator",
            "clipping_range_policy": "Clip to the canonical dictionary range [0, 10000].",
            "known_limitations": (
                "Synthetic augmentation only; cannot represent observed laboratory data."
            ),
        },
    )
