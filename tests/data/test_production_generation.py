import json
import runpy
import shutil
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from metaboguard.data.canonical import to_canonical
from metaboguard.data.production_cli import _selected_plan
from metaboguard.data.production_cli import main as production_cli_main
from metaboguard.data.production_generation import (
    ProductionCohortPlan,
    ProductionGenerationConfig,
    _generation_config,
    _load_batch_records,
    _run_frozen_pipeline,
    generate_configured_production_runs,
    generate_production_run,
    load_production_config,
    reconcile_production_manifest,
)
from metaboguard.data.synthea_runner import SyntheaGenerationConfig

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthea_one_patient"


def _mapping() -> dict[str, object]:
    return {
        "enabled": True,
        "simulation_only": True,
        "cohort_classes": {
            "ordinary_incidence": {
                "enabled": True,
                "population_sizes": [1],
                "seeds": [1],
                "augmentation_profile": "baseline",
            },
            "enriched_incidence": {
                "enabled": True,
                "population_sizes": [1],
                "seeds": [2],
                "augmentation_profile": "endpoint_enriched",
            },
        },
        "endpoints": ["type2_diabetes"],
        "horizons_years": [1, 3, 5],
        "retry": {"max_attempts": 2, "retry_only_failed_batches": True},
        "execution": {
            "batch_population_size": 1,
            "continue_on_independent_batch_failure": True,
            "require_canonical_validation_before_cohort_build": True,
            "require_complete_feature_build_before_readiness": True,
        },
        "reporting": {
            "write_per_run_reports": True,
            "write_cohort_class_feasibility_summary": True,
            "write_cross_run_manifest": True,
        },
    }


def _fake_runner(config: SyntheaGenerationConfig) -> object:
    source = config.output_root / "_fixture_raw"
    if source.exists():
        shutil.rmtree(source)
    shutil.copytree(FIXTURE, source)
    dataset = to_canonical(source)
    run = config.output_root / config.cohort_class / config.run_id
    shutil.copytree(dataset.output_dir, run / "canonical", dirs_exist_ok=True)
    (run / config.manifest_filename).write_text(
        json.dumps({"state": "complete", "cohort_class": config.cohort_class}),
        encoding="utf-8",
    )
    shutil.rmtree(source)
    return object()


def test_config_parsing_keeps_classes_separate() -> None:
    config = ProductionGenerationConfig.from_mapping(_mapping())
    assert len(load_production_config().plans) == 2
    assert [plan.cohort_class for plan in config.plans] == [
        "ordinary_incidence",
        "enriched_incidence",
    ]
    broken = _mapping()
    broken["cohort_classes"]["ordinary_incidence"]["seeds"] = [1, 2]  # type: ignore[index]
    with pytest.raises(ValueError, match="counts differ"):
        ProductionGenerationConfig.from_mapping(broken)
    for key, value in (
        ("cohort_classes", []),
        ("retry", []),
        ("execution", []),
        ("reporting", []),
        ("endpoints", "type2_diabetes"),
    ):
        invalid = _mapping()
        if key == "cohort_classes":
            invalid[key] = value
        else:
            invalid[key] = value
        with pytest.raises(TypeError):
            ProductionGenerationConfig.from_mapping(invalid)
    disabled = deepcopy(_mapping())
    disabled["cohort_classes"]["ordinary_incidence"]["enabled"] = False  # type: ignore[index]
    assert len(ProductionGenerationConfig.from_mapping(disabled).plans) == 1
    malformed = _mapping()
    malformed["cohort_classes"]["ordinary_incidence"] = []  # type: ignore[index]
    with pytest.raises(TypeError, match="mapping"):
        ProductionGenerationConfig.from_mapping(malformed)
    malformed_lists = _mapping()
    malformed_lists["cohort_classes"]["ordinary_incidence"]["seeds"] = "1"  # type: ignore[index]
    with pytest.raises(TypeError, match="lists"):
        ProductionGenerationConfig.from_mapping(malformed_lists)


def test_production_run_writes_distinct_manifest_and_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = ProductionCohortPlan("ordinary_incidence", 1, 1, "baseline")
    monkeypatch.setattr(
        "metaboguard.data.production_generation.load_production_config",
        lambda: ProductionGenerationConfig.from_mapping(_mapping()),
    )
    manifest = generate_production_run(
        plan, tmp_path / "production", generation_runner=_fake_runner
    )
    run = tmp_path / "production" / "ordinary_incidence" / manifest.run_id
    assert manifest.status in {"completed", "completed_not_ready"}
    assert manifest.model_status == "not_created"
    assert (run / "generation_manifest.json").exists()
    assert json.loads((run / "manifest.json").read_text())["model_status"] == "not_created"
    assert (run / "readiness").exists()
    assert (run / "feasibility" / "endpoint_feasibility_report.json").exists()
    no_pipeline = generate_production_run(
        plan, tmp_path / "production", run_pipeline=False, generation_runner=_fake_runner
    )
    assert no_pipeline.cohort_status == "created"


def test_configured_runs_delegate_independently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = ProductionGenerationConfig.from_mapping(_mapping())
    calls: list[str] = []
    monkeypatch.setattr(
        "metaboguard.data.production_generation.load_production_config", lambda: config
    )
    monkeypatch.setattr(
        "metaboguard.data.production_generation.generate_production_run",
        lambda plan, root, run_pipeline: calls.append(plan.cohort_class) or object(),
    )
    runs = generate_configured_production_runs(tmp_path, run_pipeline=False)
    assert len(runs) == 2
    assert calls == ["ordinary_incidence", "enriched_incidence"]


def test_failed_generation_is_persisted_and_disabled_config_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = ProductionCohortPlan("ordinary_incidence", 1, 1, "baseline")

    def failed_runner(_config: SyntheaGenerationConfig) -> object:
        raise RuntimeError("generator failed")

    with pytest.raises(RuntimeError, match="generator failed"):
        generate_production_run(plan, tmp_path / "production", generation_runner=failed_runner)
    run = next((tmp_path / "production" / "ordinary_incidence").iterdir())
    assert json.loads((run / "manifest.json").read_text())["status"] == "failed"
    disabled = ProductionGenerationConfig.from_mapping(_mapping())
    monkeypatch.setattr(
        "metaboguard.data.production_generation.load_production_config",
        lambda: replace(disabled, enabled=False),
    )
    with pytest.raises(ValueError, match="simulation-only"):
        generate_configured_production_runs(tmp_path)


def test_batch_records_and_retry_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch_root = tmp_path / "batch_manifests"
    batch_root.mkdir()
    (batch_root / "batch_00000.json").write_text(
        json.dumps({"batch_index": 0, "seed": 1, "state": "complete", "raw_sha256": "r"}),
        encoding="utf-8",
    )
    (batch_root / "batch_00001.json").write_text(
        json.dumps({"batch_index": 1, "seed": 2, "state": "failed"}), encoding="utf-8"
    )
    records = _load_batch_records(tmp_path, "now", 5)
    assert [record.status for record in records] == ["completed", "failed"]
    plan = ProductionCohortPlan("ordinary_incidence", 1, 3, "baseline")
    attempts = [0]

    def flaky_runner(config: SyntheaGenerationConfig) -> object:
        attempts[0] += 1
        if attempts[0] == 1:
            raise RuntimeError("retryable")
        return _fake_runner(config)

    monkeypatch.setattr(
        "metaboguard.data.production_generation.load_production_config",
        lambda: ProductionGenerationConfig.from_mapping(_mapping()),
    )
    manifest = generate_production_run(
        plan, tmp_path / "retry-production", run_pipeline=False, generation_runner=flaky_runner
    )
    assert attempts[0] == 2
    assert manifest.status == "completed"


def test_manifest_reconciliation_is_idempotent_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = ProductionCohortPlan("ordinary_incidence", 1, 1, "baseline")
    monkeypatch.setattr(
        "metaboguard.data.production_generation.load_production_config",
        lambda: ProductionGenerationConfig.from_mapping(_mapping()),
    )
    manifest = generate_production_run(
        plan, tmp_path / "production", run_pipeline=False, generation_runner=_fake_runner
    )
    run = tmp_path / "production" / "ordinary_incidence" / manifest.run_id
    first = reconcile_production_manifest(run).model_dump_json()
    second = reconcile_production_manifest(run).model_dump_json()
    assert first == second
    sys.argv = ["metaboguard-production", "ordinary_incidence", "--reconcile", str(run)]
    assert production_cli_main() == 0
    (run / "canonical" / "patients.parquet").unlink()
    incomplete = reconcile_production_manifest(run)
    assert incomplete.status == "partial"
    assert incomplete.canonical_status == "not_created"
    (run / "generation_manifest.json").unlink()
    incomplete_generation = reconcile_production_manifest(run)
    assert incomplete_generation.failure_summary == "required production artifacts are incomplete"


def test_frozen_pipeline_writes_feature_artifacts_when_streaming_has_no_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_path = tmp_path / "ordinary_incidence" / "run"
    run_path.mkdir(parents=True)
    (run_path / "generation_manifest.json").write_text(
        json.dumps({"cohort_class": "ordinary_incidence"}), encoding="utf-8"
    )
    dataset = SimpleNamespace(patients=[object()], cohort_class="")
    cohort = SimpleNamespace(patient_indexes=[], labels=[])
    split = SimpleNamespace(assignments={})
    calls: list[Path] = []

    def construct_cohort(_dataset: object, _endpoint: object, path: Path, _sha: str) -> object:
        path.mkdir(parents=True)
        (path / "cohort_manifest.json").write_text(
            json.dumps({"feature_status": "not_created"}), encoding="utf-8"
        )
        return cohort

    def capture_artifacts(
        _feature_dataset: object, output_dir: Path, _manifest: Path
    ) -> dict[str, object]:
        calls.append(output_dir)
        return {"row_count": 0}

    monkeypatch.setattr(
        "metaboguard.data.production_generation._read_canonical_dir", lambda _path: dataset
    )
    monkeypatch.setattr(
        "metaboguard.data.production_generation.load_endpoint_registry",
        lambda: {"type2_diabetes": object()},
    )
    monkeypatch.setattr(
        "metaboguard.data.production_generation.construct_endpoint_cohort", construct_cohort
    )
    monkeypatch.setattr(
        "metaboguard.data.production_generation.build_splits",
        lambda _cohort, _config: split,
    )
    monkeypatch.setattr(
        "metaboguard.data.production_generation.write_split_artifacts", lambda *_args: None
    )
    monkeypatch.setattr("metaboguard.data.production_generation.file_sha256", lambda _path: "sha")
    monkeypatch.setattr(
        "metaboguard.data.production_generation.extract_features",
        lambda *_args, **_kwargs: SimpleNamespace(rows=[], lineage=[], registry={}),
    )
    monkeypatch.setattr(
        "metaboguard.data.production_generation.write_feature_artifacts", capture_artifacts
    )
    monkeypatch.setattr(
        "metaboguard.data.production_generation.build_readiness_bundle", lambda *_args: None
    )
    monkeypatch.setattr(
        "metaboguard.data.production_generation.build_production_feasibility",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "metaboguard.data.production_generation.write_production_feasibility",
        lambda *_args: None,
    )

    _run_frozen_pipeline(run_path, ("type2_diabetes",), "generation-sha")

    assert calls == [run_path / "features" / "type2_diabetes"]


def test_generation_config_rejects_malformed_execution_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "metaboguard.data.production_generation.load_config",
        lambda: {"synthea": {}, "production_generation": {"execution": []}},
    )
    with pytest.raises(TypeError, match="execution configuration"):
        _generation_config(
            ProductionCohortPlan("ordinary_incidence", 1, 1, "baseline"), tmp_path, "run"
        )


def test_production_cli_previews_or_executes_one_class(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config = ProductionGenerationConfig.from_mapping(_mapping())
    monkeypatch.setattr("metaboguard.data.production_cli.load_production_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["metaboguard-production", "ordinary_incidence"])
    assert production_cli_main() == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["execute"] is False
    assert preview["model_status"] == "not_created"
    assert preview["population_target"] == 500
    calls: list[str] = []
    monkeypatch.setattr(
        "metaboguard.data.production_cli.generate_production_run",
        lambda plan, root: calls.append(str(root))
        or SimpleNamespace(
            run_id="run",
            cohort_class=plan.cohort_class,
            status="completed",
            population_target=plan.population_target,
            generated_patient_count=1,
            canonical_status="created",
            cohort_status="created",
            split_status="created",
            feature_status="created",
            readiness_status="created",
            simulation_only=True,
            model_status="not_created",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "metaboguard-production",
            "ordinary_incidence",
            "--output-root",
            str(tmp_path),
            "--execute",
        ],
    )
    assert production_cli_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert calls == [str(tmp_path)]
    assert result["cohort_class"] == "ordinary_incidence"
    assert result["model_status"] == "not_created"


def test_production_cli_plan_overrides_are_explicit() -> None:
    plan = ProductionCohortPlan("ordinary_incidence", 25000, 17, "baseline")
    selected = _selected_plan(plan, "5000", 23, 1000)
    assert selected.population_target == 5000
    assert selected.root_seed == 23
    assert selected.batch_population_size == 1000
    assert _selected_plan(plan, "all", None, None) == plan
    with pytest.raises(SystemExit, match="batch-size"):
        _selected_plan(plan, "500", None, 0)


def test_production_cli_rejects_disabled_class_and_module_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProductionGenerationConfig.from_mapping(_mapping())
    enriched_only = replace(config, plans=(config.plans[1],))
    monkeypatch.setattr(
        "metaboguard.data.production_cli.load_production_config", lambda: enriched_only
    )
    monkeypatch.setattr(sys, "argv", ["metaboguard-production", "ordinary_incidence"])
    with pytest.raises(SystemExit, match="No enabled production plan"):
        production_cli_main()
    monkeypatch.setattr(sys, "argv", ["metaboguard-production", "ordinary_incidence"])
    monkeypatch.delitem(sys.modules, "metaboguard.data.production_cli", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("metaboguard.data.production_cli", run_name="__main__")
    assert exit_info.value.code == 0
