import os
import sys
from pathlib import Path

import pytest

from metaboguard.readiness.manifests import build_readiness_bundle


def _smoke_run_path() -> Path:
    run = Path(
        "data/synthetic_longitudinal/smoke/ordinary_incidence/ordinary_incidence-20260815T153605Z-20260815-4b7a75b7"
    )
    if not (run / "manifest.json").is_file():
        pytest.skip("Smoke run fixture is not present in this workspace")
    return run


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_real_smoke_is_not_eligible_and_simulation_only() -> None:
    run = _smoke_run_path()
    report = build_readiness_bundle(run, "type2_diabetes")
    assert report["overall_decision"] == "not_eligible"
    assert "feature_build_incomplete" not in report["overall_reasons"]
    assert "synthetic_data_only_prototype" in report["overall_reasons"]
