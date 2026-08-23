"""Backward-compatible import boundary for the canonical data schema."""

from metaboguard.data.schema import (
    ClinicalEvent,
    ConditionRecord,
    OutcomeRecord,
    Patient,
)

__all__ = ["ClinicalEvent", "ConditionRecord", "OutcomeRecord", "Patient"]
