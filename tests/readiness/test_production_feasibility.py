import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from metaboguard.readiness.production_feasibility import (
    build_production_feasibility,
    write_production_feasibility,
)


def test_feasibility_is_horizon_specific_and_synthetic_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "ordinary_incidence" / "run-1"
    (run / "readiness" / "type2_diabetes").mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "cohort_class": "ordinary_incidence",
                "generated_patient_count": 100,
            }
        ),
        encoding="utf-8",
    )
    (run / "readiness" / "type2_diabetes" / "capability_report.json").write_text(
        json.dumps(
            {
                "horizon_decisions": [
                    {
                        "horizon_years": 1,
                        "event_count": 60,
                        "eligible_negative_count": 70,
                        "censored_count": 2,
                        "competing_death_count": 1,
                        "decision": "not_eligible",
                        "decision_reasons": ["simulation_only_no_clinical_model_research"],
                        "split_integrity_status": "passed",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "metaboguard.readiness.production_feasibility.inspect_artifact_inventory",
        lambda *_: SimpleNamespace(
            expected_eligible_index_count=100,
            feature_row_count=100,
            feature_build_status="complete",
        ),
    )
    monkeypatch.setattr(
        "metaboguard.readiness.production_feasibility.audit_feature_leakage",
        lambda *_: SimpleNamespace(passed=True),
    )

    class SplitRow:
        def model_dump(self) -> dict[str, object]:
            return {
                "horizon_years": 1,
                "split": "test",
                "positive_count": 30,
            }

    monkeypatch.setattr(
        "metaboguard.readiness.production_feasibility.build_split_readiness",
        lambda *_: [SplitRow()],
    )
    rows = build_production_feasibility(run, "type2_diabetes", 100)
    assert rows[0].event_count_gate_passed
    assert rows[0].eligible_non_event_gate_passed
    assert not rows[0].partition_event_gate_passed
    assert rows[0].readiness_decision == "not_eligible"
    write_production_feasibility(run, rows)
    report = json.loads((run / "feasibility" / "endpoint_feasibility_report.json").read_text())
    assert report["pipeline_rehearsal_only"] is True
    assert report["model_status"] == "not_created"
    assert "do not measure clinical prevalence" in report["claim_limitation"]
