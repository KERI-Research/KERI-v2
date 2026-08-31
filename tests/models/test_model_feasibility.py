from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from metaboguard.models.model_feasibility import (
    MANDATORY_MODEL_CARD_RESTRICTION,
    SyntheticFeasibilityAuthorization,
    run_model_feasibility,
)
from metaboguard.models.model_feasibility_cli import main as feasibility_cli_main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(patient_id: str) -> str:
    return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()


def _build_run(
    tmp_path: Path,
    *,
    contradictory_feature_flag: bool = False,
    single_class_test: bool = False,
    add_denied_feature: bool = False,
    break_isolation: bool = False,
) -> Path:
    run_path = tmp_path / "ordinary_incidence" / "run-001"
    endpoint_id = "pancreatic_cancer"
    horizon = 3
    cohort_dir = run_path / "cohort" / endpoint_id
    split_dir = cohort_dir / "splits"
    feature_dir = run_path / "features" / endpoint_id
    split_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    (run_path / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-001",
                "cohort_class": "ordinary_incidence",
                "status": "completed",
                "simulation_only": True,
                "pipeline_rehearsal_only": True,
                "configuration_sha256": "a" * 64,
                "artifact_sha256": {},
            }
        ),
        encoding="utf-8",
    )
    (run_path / "generation_manifest.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "synthea_version": "4.0.0",
                "jar_sha256": "jar",
                "canonical_dataset_sha256": "canonical",
                "simulation_only": True,
            }
        ),
        encoding="utf-8",
    )
    (cohort_dir / "endpoint_protocol.json").write_text(
        json.dumps(
            {
                "endpoint_id": endpoint_id,
                "horizon_years": [1, 3, 5],
                "definition_version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )
    (cohort_dir / "cohort_manifest.json").write_text(
        json.dumps(
            {
                "cohort_class": "ordinary_incidence",
                "simulation_only": True,
                "validation_passed": True,
            }
        ),
        encoding="utf-8",
    )

    assignments = {
        "p1": "train",
        "p2": "train",
        "p3": "train",
        "p4": "validation",
        "p5": "validation",
        "p6": "test",
        "p7": "test",
        "p8": "temporal_holdout",
        "p9": "temporal_holdout",
    }
    split_manifest = {
        "cohort_class": "ordinary_incidence",
        "endpoint_id": endpoint_id,
        "assignments": assignments,
        "patient_fingerprints": {
            patient_id: _fingerprint(patient_id) for patient_id in assignments
        },
        "simulation_only": True,
    }
    split_manifest_path = split_dir / "split_manifest.json"
    split_manifest_path.write_text(
        json.dumps(split_manifest, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "patient_fingerprint": [
                _fingerprint(patient_id) for patient_id in assignments
            ],
            "split": [assignments[patient_id] for patient_id in assignments],
        }
    ).to_parquet(split_dir / "split_assignments.parquet", index=False)

    cohort_manifest_sha = _sha256(cohort_dir / "cohort_manifest.json")
    split_manifest_sha = _sha256(split_manifest_path)
    rows = []
    labels = []
    states = {
        "p1": "positive",
        "p2": "eligible_negative",
        "p3": "censored",
        "p4": "positive",
        "p5": "eligible_negative",
        "p6": "positive",
        "p7": "eligible_negative" if not single_class_test else "positive",
        "p8": "positive",
        "p9": "eligible_negative",
    }
    for index, patient_id in enumerate(assignments, start=1):
        rows.append(
            {
                "patient_id": patient_id,
                "cohort_class": "ordinary_incidence",
                "endpoint_id": endpoint_id,
                "index_date": f"2020-01-{index:02d}",
                "split": assignments[patient_id],
                "index_sequence_number": index,
                "feature_definition_version": "1.0.0",
                "feature_window_config_sha256": "windows",
                "source_canonical_manifest_sha256": "canonical",
                "source_cohort_manifest_sha256": cohort_manifest_sha,
                "source_split_manifest_sha256": split_manifest_sha,
                "simulation_only": True,
                "glucose__latest_lifetime": 100.0 + index,
                "hba1c__latest_lifetime": 5.0 + index / 20.0,
                "smoking_status__latest_window": "never" if index % 2 else pd.NA,
                "all_null_feature": pd.NA,
                "zero_variance_feature": 1,
                **({"diagnosis_shortcut": 1.0} if add_denied_feature else {}),
            }
        )
        labels.append(
            {
                "patient_id": patient_id,
                "cohort_class": "ordinary_incidence",
                "endpoint_id": endpoint_id,
                "index_date": f"2020-01-{index:02d}",
                "horizon_years": horizon,
                "label_state": states[patient_id],
            }
        )
    if break_isolation:
        duplicate = dict(rows[1])
        duplicate["index_date"] = "2020-02-01"
        duplicate["split"] = "validation"
        rows.append(duplicate)
    pd.DataFrame(rows).to_parquet(feature_dir / "feature_matrix.parquet", index=False)
    pd.DataFrame(labels).to_parquet(
        cohort_dir / f"horizon_labels_{horizon}y.parquet", index=False
    )

    registry = [
        {"feature_id": "glucose__latest_lifetime"},
        {"feature_id": "hba1c__latest_lifetime"},
        {"feature_id": "smoking_status__latest_window"},
        {"feature_id": "all_null_feature"},
        {"feature_id": "zero_variance_feature"},
    ]
    if add_denied_feature:
        registry.append({"feature_id": "diagnosis_shortcut"})
    (feature_dir / "feature_definition_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "patient_id": [row["patient_id"] for row in rows],
            "endpoint_id": endpoint_id,
            "index_date": [row["index_date"] for row in rows],
            "feature_id": "glucose__latest_lifetime",
            "latest_source_date": [row["index_date"] for row in rows],
            "contains_post_index_record": False,
        }
    ).to_parquet(feature_dir / "feature_lineage.parquet", index=False)

    feature_manifest = {
        "validation_passed": True,
        "simulation_only": False if contradictory_feature_flag else True,
        "feature_matrix_sha256": _sha256(feature_dir / "feature_matrix.parquet"),
    }
    (feature_dir / "feature_manifest.json").write_text(
        json.dumps(feature_manifest), encoding="utf-8"
    )
    return run_path


def _auth() -> SyntheticFeasibilityAuthorization:
    return SyntheticFeasibilityAuthorization(
        synthetic_feasibility=True,
        approval_reference="professor-approval-2026-08-30",
    )


def test_authorization_requires_explicit_flag_and_reference() -> None:
    with pytest.raises(ValueError, match="synthetic-feasibility"):
        SyntheticFeasibilityAuthorization(False, "ok")
    with pytest.raises(ValueError, match="approval-reference"):
        SyntheticFeasibilityAuthorization(True, "")


def test_preview_mode_only_returns_plan(tmp_path: Path) -> None:
    run_path = _build_run(tmp_path)
    result = run_model_feasibility(
        run_path,
        "pancreatic_cancer",
        3,
        _auth(),
        execute=False,
        seed=1729,
        artifact_root=tmp_path / "artifacts",
    )
    assert result["execute"] is False
    assert "plan" in result


def test_missing_artifact_fails_closed(tmp_path: Path) -> None:
    run_path = _build_run(tmp_path)
    (run_path / "features" / "pancreatic_cancer" / "feature_manifest.json").unlink()
    with pytest.raises(ValueError, match="Missing required feasibility input artifact"):
        run_model_feasibility(
            run_path,
            "pancreatic_cancer",
            3,
            _auth(),
            execute=False,
            artifact_root=tmp_path / "artifacts",
        )


def test_cli_refuses_missing_required_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_path = _build_run(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "metaboguard-model-feasibility",
            str(run_path),
            "pancreatic_cancer",
            "--horizon",
            "3",
            "--approval-reference",
            "professor-approval-2026-08-30",
        ],
    )
    with pytest.raises(ValueError, match="synthetic-feasibility"):
        feasibility_cli_main()


def test_execute_writes_required_artifacts_and_flags(tmp_path: Path) -> None:
    run_path = _build_run(tmp_path)
    result = run_model_feasibility(
        run_path,
        "pancreatic_cancer",
        3,
        _auth(),
        execute=True,
        artifact_root=tmp_path / "artifacts",
    )
    output_dir = Path(str(result["output_dir"]))
    for path in (
        output_dir / "experiment_manifest.json",
        output_dir / "authorization.json",
        output_dir / "input_artifact_hashes.json",
        output_dir / "preprocessing_manifest.json",
        output_dir / "feature_selection_manifest.json",
        output_dir / "split_audit.json",
        output_dir / "leakage_audit.json",
        output_dir / "shortcut_checks.json",
        output_dir / "baseline" / "model.pkl",
        output_dir / "baseline" / "predictions_test.parquet",
        output_dir / "representation_head" / "encoder.pkl",
        output_dir / "representation_head" / "head.pkl",
        output_dir / "MODEL_CARD.md",
        output_dir / "SYNTHETIC_FEASIBILITY_REPORT.md",
    ):
        assert path.exists()
    manifest = json.loads((output_dir / "experiment_manifest.json").read_text())
    assert manifest["clinical_use_restrictions"]["simulation_only"] is True
    assert manifest["clinical_use_restrictions"]["clinical_use_prohibited"] is True
    assert MANDATORY_MODEL_CARD_RESTRICTION in (output_dir / "MODEL_CARD.md").read_text(
        encoding="utf-8"
    )
    predictions = pd.read_parquet(output_dir / "baseline" / "predictions_test.parquet")
    assert "patient_id" not in predictions.columns
    assert "label_state" not in predictions.columns


def test_provenance_discrepancy_is_persisted_everywhere(tmp_path: Path) -> None:
    run_path = _build_run(tmp_path, contradictory_feature_flag=True)
    result = run_model_feasibility(
        run_path,
        "pancreatic_cancer",
        3,
        _auth(),
        execute=True,
        artifact_root=tmp_path / "artifacts",
    )
    output_dir = Path(str(result["output_dir"]))
    manifest = json.loads((output_dir / "experiment_manifest.json").read_text())
    baseline_eval = json.loads(
        (output_dir / "baseline" / "evaluation_report.json").read_text()
    )
    assert manifest["source_provenance_discrepancy"] is True
    assert baseline_eval["source_provenance_discrepancy"] is True
    assert manifest["source_provenance_discrepancy_details"]


def test_train_only_preprocess_records_all_null_and_zero_variance(
    tmp_path: Path,
) -> None:
    run_path = _build_run(tmp_path)
    result = run_model_feasibility(
        run_path,
        "pancreatic_cancer",
        3,
        _auth(),
        execute=True,
        artifact_root=tmp_path / "artifacts",
    )
    output_dir = Path(str(result["output_dir"]))
    preprocess = json.loads((output_dir / "preprocessing_manifest.json").read_text())
    baseline_eval = json.loads(
        (output_dir / "baseline" / "evaluation_report.json").read_text()
    )
    assert preprocess["train_only_fit"] is True
    assert (
        "all_null_feature"
        in baseline_eval["preprocessing"]["all_null_training_columns"]
    )
    assert (
        "zero_variance_feature"
        in baseline_eval["preprocessing"]["zero_variance_training_columns"]
    )


def test_denylisted_feature_fails_leakage_audit(tmp_path: Path) -> None:
    run_path = _build_run(tmp_path, add_denied_feature=True)
    with pytest.raises(ValueError, match="Feature leakage audit failed"):
        run_model_feasibility(
            run_path,
            "pancreatic_cancer",
            3,
            _auth(),
            execute=True,
            artifact_root=tmp_path / "artifacts",
        )


def test_single_class_partition_marks_not_evaluable(tmp_path: Path) -> None:
    run_path = _build_run(tmp_path, single_class_test=True)
    result = run_model_feasibility(
        run_path,
        "pancreatic_cancer",
        3,
        _auth(),
        execute=True,
        artifact_root=tmp_path / "artifacts",
    )
    output_dir = Path(str(result["output_dir"]))
    report = json.loads(
        (output_dir / "baseline" / "evaluation_report.json").read_text()
    )
    test_partition = next(
        item for item in report["partition_reports"] if item["partition"] == "test"
    )
    assert test_partition["status"] == "not_evaluable"


def test_resume_reuses_hash_matched_experiment(tmp_path: Path) -> None:
    run_path = _build_run(tmp_path)
    first = run_model_feasibility(
        run_path,
        "pancreatic_cancer",
        3,
        _auth(),
        execute=True,
        artifact_root=tmp_path / "artifacts",
    )
    second = run_model_feasibility(
        run_path,
        "pancreatic_cancer",
        3,
        _auth(),
        execute=True,
        artifact_root=tmp_path / "artifacts",
    )
    assert first["experiment_id"] == second["experiment_id"]
    assert second["resumed"] is True


def test_patient_isolation_violation_fails_closed(tmp_path: Path) -> None:
    run_path = _build_run(tmp_path, break_isolation=True)
    with pytest.raises(ValueError, match="split assignment mismatch"):
        run_model_feasibility(
            run_path,
            "pancreatic_cancer",
            3,
            _auth(),
            execute=True,
            artifact_root=tmp_path / "artifacts",
        )


@pytest.mark.parametrize(
    "forbidden_root",
    ["artifacts/model_prototypes", "src/metaboguard/serving"],
)
def test_artifact_root_cannot_target_prototype_or_serving_paths(
    tmp_path: Path, forbidden_root: str
) -> None:
    run_path = _build_run(tmp_path)
    with pytest.raises(ValueError, match="serving or generic prototype"):
        run_model_feasibility(
            run_path,
            "pancreatic_cancer",
            3,
            _auth(),
            execute=False,
            artifact_root=tmp_path / forbidden_root,
        )


def test_artifact_root_cannot_overlap_source_run(tmp_path: Path) -> None:
    run_path = _build_run(tmp_path)
    with pytest.raises(ValueError, match="overlap the source run directory"):
        run_model_feasibility(
            run_path,
            "pancreatic_cancer",
            3,
            _auth(),
            execute=False,
            artifact_root=run_path / "artifacts",
        )


def test_non_synthetic_source_is_rejected_regardless_of_flags(tmp_path: Path) -> None:
    run_path = _build_run(tmp_path)
    (run_path / "generation_manifest.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "synthea_version": "",
                "jar_sha256": "",
                "canonical_dataset_sha256": "",
                "simulation_only": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Durable Synthea provenance evidence"):
        run_model_feasibility(
            run_path,
            "pancreatic_cancer",
            3,
            _auth(),
            execute=False,
            artifact_root=tmp_path / "artifacts",
        )


def test_cli_requires_non_empty_approval_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_path = _build_run(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "metaboguard-model-feasibility",
            str(run_path),
            "pancreatic_cancer",
            "--horizon",
            "3",
            "--synthetic-feasibility",
        ],
    )
    with pytest.raises(SystemExit):
        feasibility_cli_main()


def test_shortcut_checks_retain_permutation_flag_and_both_ablations(
    tmp_path: Path,
) -> None:
    run_path = _build_run(tmp_path)
    result = run_model_feasibility(
        run_path,
        "pancreatic_cancer",
        3,
        _auth(),
        execute=True,
        artifact_root=tmp_path / "artifacts",
    )
    output_dir = Path(str(result["output_dir"]))
    shortcuts = json.loads((output_dir / "shortcut_checks.json").read_text())
    assert "suspicious_similarity_flag" in shortcuts["label_permutation_sanity"]
    ablation_names = {row["ablation"] for row in shortcuts["feature_ablation"]}
    assert ablation_names == {
        "all_allowed_retained_features",
        "compact_metabolic_laboratory_subset",
    }


def test_forced_write_failure_leaves_no_partial_experiment_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_path = _build_run(tmp_path)

    import metaboguard.models.model_feasibility as feasibility_module

    original_write_json = feasibility_module._atomic_write_json

    def failing_write_json(path: Path, payload: dict) -> None:
        if path.name == "shortcut_checks.json":
            raise RuntimeError("forced failure")
        original_write_json(path, payload)

    monkeypatch.setattr(feasibility_module, "_atomic_write_json", failing_write_json)
    with pytest.raises(RuntimeError, match="forced failure"):
        run_model_feasibility(
            run_path,
            "pancreatic_cancer",
            3,
            _auth(),
            execute=True,
            artifact_root=tmp_path / "artifacts",
        )
    output_dir = tmp_path / "artifacts"
    manifests = list(output_dir.rglob("experiment_manifest.json"))
    assert manifests == []
    temp_leftovers = list(output_dir.rglob("tmp*")) + [
        path for path in output_dir.rglob("*") if path.suffix == "" and path.is_file()
    ]
    assert temp_leftovers == []


def test_no_raw_patient_id_in_manifest_or_reports(tmp_path: Path) -> None:
    run_path = _build_run(tmp_path)
    result = run_model_feasibility(
        run_path,
        "pancreatic_cancer",
        3,
        _auth(),
        execute=True,
        artifact_root=tmp_path / "artifacts",
    )
    output_dir = Path(str(result["output_dir"]))
    for json_path in output_dir.rglob("*.json"):
        text = json_path.read_text(encoding="utf-8")
        for patient_id in ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"):
            assert (
                patient_id not in text.split()
            ), f"raw patient id token found in {json_path}"
