"""Ordered capability decisions without model metrics."""

from __future__ import annotations

from typing import cast

from metaboguard.config import load_config
from metaboguard.readiness.contracts import (
    ArtifactInventory,
    CapabilityDecision,
    Decision,
    FeatureAvailabilityRecord,
    LabelFeasibilityRecord,
    LeakageReadinessReport,
    SplitReadinessRecord,
)


def build_capability_decisions(
    inventory: ArtifactInventory,
    labels: list[LabelFeasibilityRecord],
    features: list[FeatureAvailabilityRecord],
    splits: list[SplitReadinessRecord],
    leakage: LeakageReadinessReport,
) -> list[CapabilityDecision]:
    """Apply artifact, completeness, leakage, count, quality, and simulation gates."""
    config = load_config()["readiness"]
    decisions: list[CapabilityDecision] = []
    for record in labels:
        reasons: list[str] = []
        if inventory.missing_required:
            reasons.append("required_artifacts_missing")
        if (
            config["require_complete_feature_build"]
            and inventory.feature_build_status != "complete"
        ):
            reasons.append("feature_build_incomplete")
        if not leakage.passed:
            reasons.append("leakage_or_split_integrity_failure")
        if record.positive_count < int(config["minimum_events_per_horizon"]):
            reasons.append("insufficient_events")
        if record.eligible_negative_count < int(config["minimum_eligible_non_events_per_horizon"]):
            reasons.append("insufficient_eligible_non_events")
        if inventory.simulation_only:
            reasons.append("simulation_only_no_clinical_model_research")
        decision = (
            "blocked"
            if any(
                reason in reasons
                for reason in (
                    "required_artifacts_missing",
                    "feature_build_incomplete",
                    "leakage_or_split_integrity_failure",
                )
            )
            else "not_eligible"
            if reasons
            else "eligible_for_future_model_research"
        )
        split_rows = [item for item in splits if item.horizon_years == record.horizon_years]
        decisions.append(
            CapabilityDecision(
                cohort_class=inventory.cohort_class,
                endpoint_id=inventory.endpoint_id,
                horizon_years=record.horizon_years,
                decision=cast(Decision, decision),
                decision_reasons=reasons,
                pipeline_rehearsal_only=inventory.simulation_only,
                event_count=record.positive_count,
                eligible_negative_count=record.eligible_negative_count,
                censored_count=record.censored_count,
                competing_death_count=record.competing_death_count,
                train_event_count=sum(
                    item.positive_count for item in split_rows if item.split == "train"
                ),
                validation_event_count=sum(
                    item.positive_count for item in split_rows if item.split == "validation"
                ),
                test_event_count=sum(
                    item.positive_count for item in split_rows if item.split == "test"
                ),
                temporal_holdout_event_count=sum(
                    item.positive_count for item in split_rows if item.split == "temporal_holdout"
                ),
                feature_build_status=inventory.feature_build_status,
                feature_row_count=inventory.feature_row_count,
                feature_coverage_status="complete"
                if inventory.feature_build_status == "complete"
                else "partial",
                leakage_status="passed" if leakage.passed else "failed",
                split_integrity_status="unknown",
                simulation_only=inventory.simulation_only,
            )
        )
    return decisions
