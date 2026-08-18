from pathlib import Path

from metaboguard.readiness.manifests import build_readiness_bundle


def test_real_smoke_is_not_eligible_and_simulation_only() -> None:
    run = Path(
        "data/synthetic_longitudinal/smoke/ordinary_incidence/ordinary_incidence-20260815T153605Z-20260815-4b7a75b7"
    )
    report = build_readiness_bundle(run, "type2_diabetes")
    assert report["overall_decision"] == "not_eligible"
    assert "feature_build_incomplete" not in report["overall_reasons"]
    assert "simulation_only_no_clinical_model_research" in report["overall_reasons"]
