from pathlib import Path

from metaboguard.readiness.manifests import build_readiness_bundle


def test_real_smoke_is_not_eligible_and_simulation_only() -> None:
    # ! REVIEW
    # This test assumes a checked-in completed smoke run, but the referenced
    # directory currently contains only readiness/ and no manifest.json. The
    # fixture must be restored/generated or the test should use a valid fixture.
    # ----------
    run = Path(
        "data/synthetic_longitudinal/smoke/ordinary_incidence/ordinary_incidence-20260815T153605Z-20260815-4b7a75b7"
    )
    report = build_readiness_bundle(run, "type2_diabetes")
    assert report["overall_decision"] == "not_eligible"
    assert "feature_build_incomplete" not in report["overall_reasons"]
    assert "synthetic_data_only_prototype" in report["overall_reasons"]
