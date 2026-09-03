"""Cancer-type and diabetes-type classifier boundary.

Cancer site and diabetes type are matched deterministically from canonical
condition records during cohort construction (see
docs/Cohort Management/Cohort Construction.md); they are not predicted by a
model anywhere in this project today. Type 1 and gestational diabetes typing
remain explicitly disabled "future-development endpoints" per the same
document, independent of whether the source data is synthetic or real.

This module exists to give that restriction an explicit, load-bearing
boundary in code rather than leaving it undocumented: any attempt to fit a
typing head fails closed with a clear reason, for every disabled target,
under every authorization path (real or synthetic-prototype), until a
typing label pipeline and an explicit project-plan step authorize it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from metaboguard.models.ssl_encoder import SyntheticPrototypeAuthorization

MANDATORY_MODEL_CARD_RESTRICTION = (
    "Cancer-type and diabetes-type classification is not an authorized model "
    "capability in this project. Cancer site and diabetes type are matched "
    "deterministically from canonical condition records, not predicted. This "
    "boundary must not be bypassed to produce type predictions for diagnosis, "
    "patient screening, risk assessment, medical advice, treatment decisions, "
    "triage, or patient care."
)

#: Typing targets explicitly disabled as future-development endpoints,
#: regardless of data source, per docs/Cohort Management/Cohort Construction.md.
DISABLED_TYPING_TARGETS = frozenset(
    {"cancer_type", "diabetes_type_1", "diabetes_type_gestational"}
)


class TypingHeadNotAuthorizedError(ValueError):
    """Raised for every fit attempt: typing heads are not an authorized capability."""


class TypingHeadModel:
    """A hard-gated boundary for a not-yet-authorized typing classifier."""

    def __init__(self, typing_target: str) -> None:
        if typing_target not in DISABLED_TYPING_TARGETS:
            raise ValueError(f"unrecognized typing target: {typing_target}")
        self.typing_target = typing_target

    def fit(
        self, feature_matrix: pd.DataFrame, capability_report: dict[str, Any]
    ) -> TypingHeadModel:
        """Always fail closed: no typing target is currently authorized to fit."""
        _ = feature_matrix, capability_report
        raise TypingHeadNotAuthorizedError(
            f"typing target {self.typing_target!r} is a disabled future-development "
            "endpoint; cancer site and diabetes type are matched from canonical "
            "condition records, not fitted by a model"
        )

    def fit_synthetic_prototype(
        self,
        feature_matrix: pd.DataFrame,
        authorization: SyntheticPrototypeAuthorization,
    ) -> TypingHeadModel:
        """Always fail closed, even for an explicitly approved synthetic rehearsal."""
        _ = feature_matrix, authorization
        raise TypingHeadNotAuthorizedError(
            f"typing target {self.typing_target!r} is a disabled future-development "
            "endpoint; synthetic-prototype approval does not authorize typing-head "
            "fitting"
        )
