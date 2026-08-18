"""Strict Step 6 readiness contracts."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


Decision = Literal["eligible_for_future_model_research", "not_eligible", "blocked"]
FeatureBuildStatus = Literal["not_started", "partial", "complete", "unknown"]


class ArtifactRecord(StrictModel):
    path: str
    required: bool
    present: bool
    byte_size: int = 0
    sha256: str = ""
    recorded_sha256: str | None = None
    hash_matches: bool | None = None


class ArtifactInventory(StrictModel):
    run_path: str
    endpoint_id: str
    cohort_class: str
    artifacts: list[ArtifactRecord]
    missing_required: list[str]
    missing_optional: list[str]
    feature_build_status: FeatureBuildStatus
    feature_row_count: int
    expected_eligible_index_count: int
    operational_warnings: list[str]
    simulation_only: bool


class LabelFeasibilityRecord(StrictModel):
    cohort_class: str
    endpoint_id: str
    horizon_years: int
    split: str
    total_labelled_indexes: int
    unique_patient_count: int
    positive_count: int
    eligible_negative_count: int
    censored_count: int
    competing_death_count: int
    excluded_count: int
    evaluable_prevalence: float | None
    follow_up_median_days: float | None = None


class FeatureAvailabilityRecord(StrictModel):
    cohort_class: str
    endpoint_id: str
    split: str
    feature_id: str
    feature_family: str
    source_feature_id: str
    window: str
    value_type: str
    unit: str
    row_count: int
    available_count: int
    availability_fraction: float
    null_count: int
    null_fraction: float
    all_null: bool
    zero_variance: bool
    simulation_only: bool


class SplitReadinessRecord(StrictModel):
    cohort_class: str
    endpoint_id: str
    horizon_years: int
    split: str
    patient_count: int
    index_count: int
    positive_count: int
    eligible_negative_count: int
    censored_count: int
    competing_death_count: int
    first_index_date: date | None
    last_index_date: date | None
    feature_complete_fraction: float
    reaches_evaluation_event_threshold: bool


class LeakageReadinessReport(StrictModel):
    passed: bool
    checks: list[dict[str, object]]
    source_hashes: dict[str, str]
    manual_inspection_scope: str
    feature_build_status: FeatureBuildStatus
    simulation_only: bool


class CapabilityDecision(StrictModel):
    cohort_class: str
    endpoint_id: str
    horizon_years: int
    decision: Decision
    decision_reasons: list[str]
    pipeline_rehearsal_only: bool
    event_count: int
    eligible_negative_count: int
    censored_count: int
    competing_death_count: int
    train_event_count: int
    validation_event_count: int
    test_event_count: int
    temporal_holdout_event_count: int
    feature_build_status: FeatureBuildStatus
    feature_row_count: int
    feature_coverage_status: str
    leakage_status: str
    split_integrity_status: str
    simulation_only: bool


class ReadinessCheck(StrictModel):
    name: str
    level: Literal["error", "warning"]
    status: Literal["passed", "failed", "warning"]
    passed: bool
    offending_count: int
    warning_count: int = 0
    message: str = ""


class ReadinessValidationReport(StrictModel):
    passed: bool
    checks: list[ReadinessCheck]
    simulation_only: bool


class ReadinessManifest(StrictModel):
    report_version: str
    cohort_class: str
    endpoint_id: str
    readiness_status: str
    readiness_decision: Decision
    feature_build_status: FeatureBuildStatus
    artifact_inventory_sha256: str
    label_feasibility_sha256: str
    feature_availability_sha256: str
    split_readiness_sha256: str
    leakage_readiness_sha256: str
    readiness_validation_report_sha256: str
    simulation_only: bool
