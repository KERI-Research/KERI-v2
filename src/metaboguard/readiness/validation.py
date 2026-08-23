"""Readiness bundle consistency validation."""

from __future__ import annotations

from metaboguard.readiness.contracts import (
    ArtifactInventory,
    CapabilityDecision,
    FeatureAvailabilityRecord,
    LabelFeasibilityRecord,
    LeakageReadinessReport,
    ReadinessCheck,
    ReadinessValidationReport,
    SplitReadinessRecord,
)


def validate_readiness_bundle(
    inventory: ArtifactInventory,
    labels: list[LabelFeasibilityRecord],
    features: list[FeatureAvailabilityRecord],
    splits: list[SplitReadinessRecord],
    leakage: LeakageReadinessReport,
    capability: list[CapabilityDecision],
) -> ReadinessValidationReport:
    """Validate readiness consistency without mutating source artifacts."""
    checks = [
        ReadinessCheck(
            name="required_artifacts",
            level="error",
            status="failed" if inventory.missing_required else "passed",
            passed=not inventory.missing_required,
            warning_count=0,
            offending_count=len(inventory.missing_required),
        ),
        ReadinessCheck(
            name="feature_build_status_truthful",
            level="error",
            status=(
                "failed" if inventory.feature_build_status == "unknown" else "passed"
            ),
            passed=inventory.feature_build_status != "unknown",
            warning_count=0,
            offending_count=int(inventory.feature_build_status == "unknown"),
        ),
        ReadinessCheck(
            name="leakage_audit",
            level="error",
            status="failed" if not leakage.passed else "passed",
            passed=leakage.passed,
            warning_count=0,
            offending_count=sum(not bool(item["passed"]) for item in leakage.checks),
        ),
        ReadinessCheck(
            name="simulation_only",
            level="error",
            status="passed" if inventory.simulation_only else "failed",
            passed=inventory.simulation_only,
            warning_count=0,
            offending_count=int(not inventory.simulation_only),
        ),
        ReadinessCheck(
            name="partial_feature_build",
            level="warning",
            status=(
                "warning" if inventory.feature_build_status == "partial" else "passed"
            ),
            passed=True,
            warning_count=int(inventory.feature_build_status == "partial"),
            offending_count=int(inventory.feature_build_status == "partial"),
            message="A bounded inspection is not a complete feature build.",
        ),
        ReadinessCheck(
            name="capability_decisions_present",
            level="error",
            status="passed" if capability else "failed",
            passed=bool(capability),
            warning_count=0,
            offending_count=int(not capability),
        ),
    ]
    return ReadinessValidationReport(
        passed=all(check.passed for check in checks if check.level == "error"),
        checks=checks,
        simulation_only=True,
    )
