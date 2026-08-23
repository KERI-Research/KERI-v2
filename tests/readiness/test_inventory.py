import os
import sys
from pathlib import Path

import pytest

from metaboguard.readiness.inventory import inspect_artifact_inventory


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
def test_real_smoke_is_complete() -> None:
    run = _smoke_run_path()
    inventory = inspect_artifact_inventory(run, "type2_diabetes")
    assert inventory.feature_build_status == "complete"
    assert inventory.feature_row_count == 1019
    assert inventory.expected_eligible_index_count == 1019
