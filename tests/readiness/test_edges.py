import json
from pathlib import Path

import pandas as pd

from metaboguard.readiness.capability import build_capability_decisions
from metaboguard.readiness.contracts import (
    ArtifactInventory,
    LabelFeasibilityRecord,
    LeakageReadinessReport,
)
from metaboguard.readiness.features import build_feature_availability
from metaboguard.readiness.inventory import (
    feature_artifact_dir,
    inspect_artifact_inventory,
)
from metaboguard.readiness.labels import build_label_feasibility
from metaboguard.readiness.splits import build_split_readiness


def test_inventory_not_started_and_direct_path(tmp_path: Path) -> None:
    run = tmp_path
    (run / "manifest.json").write_text(
        json.dumps({"cohort_class": "ordinary_incidence", "simulation_only": True})
    )
    (run / "features" / "e").mkdir(parents=True)
    assert feature_artifact_dir(run, "e") == run / "features" / "e"
    inventory = inspect_artifact_inventory(run, "e")
    assert inventory.feature_build_status == "not_started"
    assert inventory.missing_required


def test_inventory_unknown_scope(tmp_path: Path) -> None:
    run = tmp_path
    (run / "manifest.json").write_text(
        json.dumps({"cohort_class": "ordinary_incidence", "simulation_only": True})
    )
    (run / "features" / "e").mkdir(parents=True)
    (run / "cohort" / "e").mkdir(parents=True)
    pd.DataFrame({"patient_id": ["p"]}).to_parquet(
        run / "features" / "e" / "feature_matrix.parquet"
    )
    pd.DataFrame({"patient_id": ["p", "q"]}).to_parquet(
        run / "cohort" / "e" / "eligible_indexes.parquet"
    )
    inventory = inspect_artifact_inventory(run, "e")
    assert inventory.feature_build_status == "partial"


def test_inventory_unknown_without_expected_indexes(tmp_path: Path) -> None:
    run = tmp_path
    (run / "manifest.json").write_text(
        json.dumps({"cohort_class": "ordinary_incidence", "simulation_only": True})
    )
    features = run / "features" / "e"
    features.mkdir(parents=True)
    pd.DataFrame({"patient_id": ["p"]}).to_parquet(features / "feature_matrix.parquet")
    (run / "cohort" / "e").mkdir(parents=True)
    inventory = inspect_artifact_inventory(run, "e")
    assert inventory.feature_build_status == "unknown"


def test_inventory_handles_malformed_artifact_manifest(tmp_path: Path) -> None:
    run = tmp_path
    (run / "manifest.json").write_text(
        json.dumps({"cohort_class": "ordinary_incidence", "simulation_only": True})
    )
    (run / "cohort" / "e").mkdir(parents=True)
    (run / "cohort" / "e" / "cohort_manifest.json").write_text("{")
    inventory = inspect_artifact_inventory(run, "e")
    record = next(
        item for item in inventory.artifacts if item.path.endswith("cohort_manifest.json")
    )
    assert record.recorded_sha256 is None


def test_inventory_complete_feature_build(tmp_path: Path) -> None:
    run = tmp_path
    (run / "manifest.json").write_text(
        json.dumps({"cohort_class": "ordinary_incidence", "simulation_only": False})
    )
    features = run / "features" / "e"
    features.mkdir(parents=True)
    (run / "cohort" / "e").mkdir(parents=True)
    pd.DataFrame({"patient_id": ["p"]}).to_parquet(features / "feature_matrix.parquet")
    pd.DataFrame({"patient_id": ["p"]}).to_parquet(
        run / "cohort" / "e" / "eligible_indexes.parquet"
    )
    inventory = inspect_artifact_inventory(run, "e")
    assert inventory.feature_build_status == "complete"


def test_feature_availability_skips_unregistered(tmp_path: Path) -> None:
    features = tmp_path / "features" / "e"
    features.mkdir(parents=True)
    pd.DataFrame(
        {
            "patient_id": ["p"],
            "cohort_class": ["ordinary_incidence"],
            "endpoint_id": ["e"],
            "index_date": ["2020-01-01"],
            "split": ["train"],
            "unknown": [1],
        }
    ).to_parquet(features / "feature_matrix.parquet")
    (features / "feature_definition_registry.json").write_text("[]")
    assert build_feature_availability(tmp_path, "e") == []


def test_labels_and_splits_missing_horizon(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort" / "e" / "splits"
    cohort.mkdir(parents=True)
    pd.DataFrame({"patient_fingerprint": ["x"], "split": ["train"]}).to_parquet(
        cohort / "split_assignments.parquet"
    )
    (cohort / "split_manifest.json").write_text(json.dumps({"assignments": {"p": "train"}}))
    pd.DataFrame({"patient_id": ["p"]}).to_parquet(cohort.parent / "eligible_indexes.parquet")
    features = tmp_path / "features" / "e"
    features.mkdir(parents=True)
    pd.DataFrame({"patient_id": ["p"]}).to_parquet(features / "feature_matrix.parquet")
    assert build_label_feasibility(tmp_path, "e") == []
    assert build_split_readiness(tmp_path, "e") == []


def test_labels_without_fingerprint_assignments(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort" / "e" / "splits"
    cohort.mkdir(parents=True)
    pd.DataFrame({"split": ["train"]}).to_parquet(cohort / "split_assignments.parquet")
    pd.DataFrame(
        {
            "patient_id": ["p"],
            "cohort_class": ["ordinary_incidence"],
            "label_state": ["positive"],
        }
    ).to_parquet(cohort.parent / "horizon_labels_1y.parquet")
    records = build_label_feasibility(tmp_path, "e")
    assert records[0].positive_count == 1


def test_capability_decision_blocked_not_eligible_and_eligible() -> None:
    inventory = ArtifactInventory(
        run_path="r",
        endpoint_id="e",
        cohort_class="ordinary_incidence",
        artifacts=[],
        missing_required=[],
        missing_optional=[],
        feature_build_status="complete",
        feature_row_count=1,
        expected_eligible_index_count=1,
        operational_warnings=[],
        simulation_only=False,
    )
    label = LabelFeasibilityRecord(
        cohort_class="ordinary_incidence",
        endpoint_id="e",
        horizon_years=1,
        split="all",
        total_labelled_indexes=100,
        unique_patient_count=1,
        positive_count=60,
        eligible_negative_count=60,
        censored_count=0,
        competing_death_count=0,
        excluded_count=0,
        evaluable_prevalence=0.5,
    )
    leakage = LeakageReadinessReport(
        passed=True,
        checks=[],
        source_hashes={},
        manual_inspection_scope="test",
        feature_build_status="complete",
        simulation_only=False,
    )
    decisions = build_capability_decisions(inventory, [label], [], [], leakage)
    assert decisions[0].decision == "eligible_for_future_model_research"
    blocked = inventory.model_copy(update={"feature_build_status": "partial"})
    assert build_capability_decisions(blocked, [label], [], [], leakage)[0].decision == "blocked"
    insufficient = label.model_copy(update={"positive_count": 1})
    assert (
        build_capability_decisions(inventory, [insufficient], [], [], leakage)[0].decision
        == "not_eligible"
    )
    missing = inventory.model_copy(update={"missing_required": ["manifest.json"]})
    assert build_capability_decisions(missing, [label], [], [], leakage)[0].decision == "blocked"
    leakage_failure = leakage.model_copy(update={"passed": False})
    assert (
        build_capability_decisions(inventory, [label], [], [], leakage_failure)[0].decision
        == "blocked"
    )
    non_events = label.model_copy(update={"eligible_negative_count": 1})
    assert (
        build_capability_decisions(inventory, [non_events], [], [], leakage)[0].decision
        == "not_eligible"
    )
    synthetic = inventory.model_copy(update={"simulation_only": True})
    synthetic_decision = build_capability_decisions(synthetic, [label], [], [], leakage)[0]
    assert synthetic_decision.decision == "prototype_ready"
    assert synthetic_decision.decision_reasons == ["synthetic_data_only_prototype"]
