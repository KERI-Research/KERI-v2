"""Step-three generation, augmentation, manifest, and capability tests."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from metaboguard.data import manifests as manifest_module
from metaboguard.data import synthea_runner as runner_module
from metaboguard.data.__main__ import main as cli_main
from metaboguard.data.augmentation import (
    augment_c_peptide,
    augment_ca19_9,
    augment_insulin,
)
from metaboguard.data.canonical import CanonicalDataset, to_canonical
from metaboguard.data.capability import _iqr, build_capability_report
from metaboguard.data.manifests import (
    CohortClassMismatchError,
    GenerationManifest,
    assert_dataset_class,
    assert_same_cohort_class,
    config_sha256,
    derive_batch_seed,
    file_sha256,
    git_sha,
    new_run_id,
    runtime_metadata,
)
from metaboguard.data.synthea_runner import (
    SyntheaGenerationConfig,
    SyntheaGenerationError,
    _canonical_dir_hash,
    _complete_batch,
    _merge_datasets,
    _run_java_batch,
    _validate_raw_export,
    _verify_jar,
    _write_models,
    generate_synthea_cohort,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthea_one_patient"


def _jar(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "synthea-with-dependencies.jar"
    path.write_bytes(b"fake jar for unit tests")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_generator(raw_dir: Path, _batch_size: int, _seed: int) -> list[str]:
    for source in ("patients", "encounters", "observations", "conditions"):
        shutil.copy2(FIXTURE / f"{source}.csv", raw_dir / f"{source}.csv")
    (raw_dir / "medications.csv").write_text("", encoding="utf-8")
    (raw_dir / "procedures.csv").write_text("", encoding="utf-8")
    return ["fake-synthea"]


def _config(tmp_path: Path, **changes: object) -> SyntheaGenerationConfig:
    jar_path, jar_hash = _jar(tmp_path)
    values: dict[str, object] = {
        "cohort_class": "ordinary_incidence",
        "root_seed": 1729,
        "target_patients": 1,
        "batch_size": 1,
        "jar_path": jar_path,
        "jar_sha256": jar_hash,
        "synthea_version": "test",
        "geography": "test",
        "min_age": 18,
        "max_age": 100,
        "output_root": tmp_path / "output",
        "run_id": "test-run",
        "batch_generator": _fake_generator,
    }
    values.update(changes)
    return SyntheaGenerationConfig(**values)


def test_batch_seeds_are_deterministic_and_distinct() -> None:
    assert derive_batch_seed(1729, 0) == derive_batch_seed(1729, 0)
    assert derive_batch_seed(1729, 0) != derive_batch_seed(1729, 1)


def test_mixed_cohort_classes_fail_closed() -> None:
    with pytest.raises(CohortClassMismatchError):
        assert_same_cohort_class([])
    with pytest.raises(CohortClassMismatchError):
        assert_same_cohort_class(["ordinary_incidence", "enriched_pancreatic"])
    with pytest.raises(CohortClassMismatchError):
        assert_same_cohort_class(["unknown"])


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_augmentation_is_seeded_and_preserves_provenance() -> None:
    dataset = to_canonical(FIXTURE)
    for augmenter in (augment_insulin, augment_c_peptide, augment_ca19_9):
        first = augmenter(dataset, np.random.default_rng(17), 17)
        second = augmenter(dataset, np.random.default_rng(17), 17)
        assert [event.model_dump() for event in first.dataset.events] == [
            event.model_dump() for event in second.dataset.events
        ]
        generated = [
            event for event in first.dataset.events if event.provenance == "augmented"
        ]
        assert generated
        assert all(event.augmentation_module for event in generated)
        assert first.assumptions["module_version"] == "1.0.0"


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_augmentation_ignores_labels_and_future_events() -> None:
    dataset = to_canonical(FIXTURE)
    baseline = augment_ca19_9(dataset, np.random.default_rng(23), 23)
    shuffled = replace(
        dataset, conditions=list(reversed(dataset.conditions)), outcomes=[]
    )
    without_future = replace(
        dataset,
        events=[event for event in dataset.events if event.event_date.year < 2022],
    )
    shuffled_result = augment_ca19_9(shuffled, np.random.default_rng(23), 23)
    earlier_result = augment_ca19_9(without_future, np.random.default_rng(23), 23)
    baseline_earlier = [
        event.model_dump()
        for event in baseline.dataset.events
        if event.event_date.year < 2022
    ]
    assert baseline_earlier == [
        event.model_dump()
        for event in earlier_result.dataset.events
        if event.event_date.year < 2022
    ]
    assert [
        event.model_dump()
        for event in baseline.dataset.events
        if event.provenance == "augmented"
    ] == [
        event.model_dump()
        for event in shuffled_result.dataset.events
        if event.provenance == "augmented"
    ]


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_changed_seed_changes_stochastic_values_with_same_schema() -> None:
    dataset = to_canonical(FIXTURE)
    first = augment_ca19_9(dataset, np.random.default_rng(1), 1).dataset
    second = augment_ca19_9(dataset, np.random.default_rng(2), 2).dataset
    first_values = [
        event.value for event in first.events if event.provenance == "augmented"
    ]
    second_values = [
        event.value for event in second.events if event.provenance == "augmented"
    ]
    assert first_values != second_values
    assert [event.feature_name for event in first.events] == [
        event.feature_name for event in second.events
    ]


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_smoke_generation_writes_contract_and_deletes_raw(tmp_path: Path) -> None:
    manifest = generate_synthea_cohort(_config(tmp_path))
    run_dir = tmp_path / "output" / "ordinary_incidence" / "test-run"
    assert manifest.simulation_only is True
    assert manifest.split_status == "not_created"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "canonical" / "events.parquet").exists()
    assert (run_dir / "augmentation" / "ca19_9_assumptions.json").exists()
    assert (run_dir / "canonical" / "cohort_metadata.json").exists()
    assert (run_dir / "canonical" / "date_normalisation_audit.json").exists()
    assert (run_dir / "reports" / "date_normalisation_audit.json").exists()
    assert not (run_dir / "raw").exists()
    assert (
        json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["state"]
        == "complete"
    )
    assert manifest.sampling_stratum == "ordinary_incidence"
    assert manifest.inverse_probability_weight_policy
    assert all("module_sha256" in module for module in manifest.augmentation_modules)
    assert manifest.date_normalisation_audit_sha256


def test_jar_hash_mismatch_fails_before_generator(tmp_path: Path) -> None:
    called = False

    def generator(_path: Path, _size: int, _seed: int) -> None:
        nonlocal called
        called = True

    config = _config(tmp_path, jar_sha256="0" * 64, batch_generator=generator)
    with pytest.raises(SyntheaGenerationError, match="hash mismatch"):
        generate_synthea_cohort(config)
    assert called is False


def test_jar_preconditions_and_raw_export_fail_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(SyntheaGenerationError, match="SHA-256"):
        _verify_jar(replace(config, jar_sha256=""))
    with pytest.raises(SyntheaGenerationError, match="does not exist"):
        _verify_jar(replace(config, jar_path=tmp_path / "missing.jar"))
    with pytest.raises(SyntheaGenerationError, match="missing CSV"):
        _validate_raw_export(tmp_path / "missing-raw")


def test_config_validation_and_java_version_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        _config(tmp_path, target_patients=0)
    with pytest.raises(ValueError, match="age range"):
        _config(tmp_path, min_age=101, max_age=100)

    class Result:
        stderr = 'openjdk version "21"'
        stdout = ""

    monkeypatch.setattr(
        runner_module.subprocess, "run", lambda *_args, **_kwargs: Result()
    )
    assert runner_module._java_version("java").startswith("openjdk")
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")),
    )
    with pytest.raises(SyntheaGenerationError, match="Java runtime"):
        runner_module._java_version("java")

    class SuccessfulResult:
        stderr = ""
        stdout = "java ok"

    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SuccessfulResult(),
    )
    command = _run_java_batch(config, tmp_path / "raw", 7)
    assert command[0] == "java"
    assert "--exporter.csv.export=true" in command
    assert "--exporter.fhir.export=false" in command


def test_java_batch_flattens_synthea_csv_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    raw_dir = tmp_path / "raw"
    exported_dir = raw_dir / "csv"
    exported_dir.mkdir(parents=True)
    (exported_dir / "patients.csv").write_text("Id\n", encoding="utf-8")
    monkeypatch.setattr(runner_module.subprocess, "run", lambda *_args, **_kwargs: None)
    _run_java_batch(config, raw_dir, 7)
    assert (raw_dir / "patients.csv").exists()
    assert not exported_dir.exists()


def test_java_failure_is_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("java unavailable")

    monkeypatch.setattr("metaboguard.data.synthea_runner.subprocess.run", fail)
    with pytest.raises(SyntheaGenerationError, match="Synthea batch failed"):
        _run_java_batch(config, tmp_path / "raw", 1)


def test_java_fatal_error_retries_with_stable_gc_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path,
        jvm_options=("-Xmx6g", "-XX:+UseSerialGC", "-XX:ActiveProcessorCount=4"),
    )
    raw_dir = tmp_path / "raw"
    commands: list[list[str]] = []

    def crash_then_succeed(command: list[str], **_kwargs: object) -> None:
        commands.append(command)
        if len(commands) == 1:
            (raw_dir / "hs_err_pid123.log").write_text("fatal JVM error")
            raise runner_module.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runner_module.subprocess, "run", crash_then_succeed)

    command = _run_java_batch(config, raw_dir, 7)

    assert len(commands) == 2
    assert "-XX:+UseSerialGC" not in commands[1]
    assert "-XX:+UseG1GC" in commands[1]
    assert "-Xmx4g" in commands[1]
    assert command == commands[1]


def test_java_fatal_error_retries_without_jit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    raw_dir = tmp_path / "raw"
    commands: list[list[str]] = []

    def crash_then_succeed(command: list[str], **_kwargs: object) -> None:
        commands.append(command)
        if len(commands) == 1:
            (raw_dir / "hs_err_pid123.log").write_text("fatal JVM error")
            raise runner_module.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runner_module.subprocess, "run", crash_then_succeed)

    command = _run_java_batch(config, raw_dir, 7)

    assert len(commands) == 2
    assert "-Xint" in commands[1]
    assert command == commands[1]


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_generation_resumes_complete_batch(tmp_path: Path) -> None:
    first = generate_synthea_cohort(_config(tmp_path))
    calls = 0

    def should_not_run(_path: Path, _size: int, _seed: int) -> None:
        nonlocal calls
        calls += 1

    resumed = generate_synthea_cohort(_config(tmp_path, batch_generator=should_not_run))
    assert first.canonical_dataset_sha256 == resumed.canonical_dataset_sha256
    assert calls == 0


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_corrupted_batch_canonical_is_not_accepted(tmp_path: Path) -> None:
    generate_synthea_cohort(_config(tmp_path))
    batch_events = (
        tmp_path
        / "output"
        / "ordinary_incidence"
        / "test-run"
        / "canonical"
        / "batches"
        / "batch_00000"
        / "events.parquet"
    )
    batch_events.write_bytes(b"corrupt")
    calls = 0

    def regenerate(path: Path, size: int, seed: int) -> list[str]:
        nonlocal calls
        calls += 1
        return _fake_generator(path, size, seed)

    generate_synthea_cohort(_config(tmp_path, batch_generator=regenerate))
    assert calls == 1


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_capability_report_is_simulation_only() -> None:
    dataset = to_canonical(FIXTURE)
    report = build_capability_report(dataset, "ordinary_incidence")
    assert report.simulation_only is True
    assert report.state == "ready_for_cohort_construction"
    assert report.incident_cancer_count_by_site["pancreas"] == 1
    assert (
        report.horizon_eligible_event_counts["1"]
        >= report.horizon_eligible_event_counts["5"]
    )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_capability_empty_and_no_death_edges(tmp_path: Path) -> None:
    empty = to_canonical(FIXTURE)
    empty = replace(empty, patients=[], events=[], conditions=[], outcomes=[])
    report = build_capability_report(empty, "ordinary_incidence")
    assert report.state == "insufficient_event_count"
    assert _iqr([]) == (0.0, 0.0, 0.0)
    dataset = to_canonical(FIXTURE)
    no_death = replace(
        dataset,
        patients=[dataset.patients[0].model_copy(update={"death_date": None})],
    )
    no_death_report = build_capability_report(no_death, "ordinary_incidence")
    assert no_death_report.horizon_eligible_event_counts["1"] == 1
    no_death_report.write(tmp_path / "capability.json")
    assert (tmp_path / "capability.json").exists()
    unknown_condition = dataset.conditions[1].model_copy(
        update={"patient_id": "unknown"}
    )
    unknown_report = build_capability_report(
        replace(dataset, conditions=[unknown_condition]), "ordinary_incidence"
    )
    assert unknown_report.diabetes_type_counts["none"] == 1
    unknown_diabetes = dataset.conditions[0].model_copy(
        update={"patient_id": "unknown"}
    )
    build_capability_report(
        replace(dataset, conditions=[unknown_diabetes]), "ordinary_incidence"
    )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_manifest_helpers_and_dataset_class(tmp_path: Path) -> None:
    dataset = to_canonical(FIXTURE)
    dataset.cohort_class = "ordinary_incidence"
    assert_dataset_class(dataset, "ordinary_incidence")
    with pytest.raises(CohortClassMismatchError):
        assert_dataset_class(dataset, "enriched_pancreatic")
    assert config_sha256({"seed": 1}) == config_sha256({"seed": 1})
    assert file_sha256(FIXTURE / "patients.csv")
    assert git_sha()
    assert new_run_id("ordinary_incidence", 1).startswith("ordinary_incidence-")
    assert runtime_metadata("java")["java_version"] == "java"
    manifest = GenerationManifest(
        cohort_class="ordinary_incidence",
        run_id="r",
        root_seed=1,
        batch_seeds=[1],
        config_hash="c",
        git_sha="g",
        synthea_version="s",
        jar_sha256="j",
        java_version="j",
        operating_system="o",
        python_version="p",
        geography="g",
        age_range=(18, 100),
        population_settings={},
        sampling_stratum="ordinary_incidence",
        inverse_probability_weight_policy="none",
        generation_commands=[],
    )
    manifest.write(tmp_path / "manifest.json")
    assert (tmp_path / "manifest.json").exists()


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_merge_rejects_duplicate_patients(tmp_path: Path) -> None:
    dataset = to_canonical(FIXTURE)
    with pytest.raises(SyntheaGenerationError, match="Duplicate patient IDs"):
        _merge_datasets([dataset, dataset], tmp_path / "canonical")


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_complete_batch_requires_matching_hash(tmp_path: Path) -> None:
    manifest = generate_synthea_cohort(_config(tmp_path))
    run_dir = tmp_path / "output" / "ordinary_incidence" / manifest.run_id
    batch_dir = run_dir / "canonical" / "batches" / "batch_00000"
    batch_manifest = run_dir / "batch_manifests" / "batch_00000.json"
    completed = _complete_batch(batch_manifest, batch_dir)
    assert completed is not None
    (batch_dir / "events.parquet").write_bytes(b"corrupt")
    assert _complete_batch(batch_manifest, batch_dir) is None
    invalid_manifest = tmp_path / "invalid.json"
    invalid_manifest.write_text("not json", encoding="utf-8")
    assert _complete_batch(invalid_manifest, batch_dir) is None


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_canonical_hash_supports_legacy_batch_without_audit(tmp_path: Path) -> None:
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()
    (canonical_dir / "patients.parquet").write_bytes(b"empty")
    assert _canonical_dir_hash(canonical_dir)


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_manifest_git_failure_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        manifest_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("git missing")),
    )
    assert manifest_module.git_sha() == "unavailable"

    class GitResult:
        stdout = "abc123\n"

    monkeypatch.setattr(
        manifest_module.subprocess,
        "run",
        lambda *_args, **_kwargs: GitResult(),
    )
    assert manifest_module.git_sha() == "abc123"


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_empty_table_writer(tmp_path: Path) -> None:
    _write_models(CanonicalDataset([], [], [], []), tmp_path / "canonical")
    assert (tmp_path / "canonical" / "patients.parquet").exists()


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_batch_validation_failure_keeps_generation_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner_module, "validate", lambda _dataset: SimpleNamespace(passed=False)
    )
    with pytest.raises(SyntheaGenerationError, match="batch 0"):
        generate_synthea_cohort(_config(tmp_path))


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_missing_converter_output_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = to_canonical(FIXTURE)
    monkeypatch.setattr(
        runner_module,
        "to_canonical",
        lambda _path, **_kwargs: replace(dataset, output_dir=None),
    )
    with pytest.raises(SyntheaGenerationError, match="output directory"):
        generate_synthea_cohort(_config(tmp_path))


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_final_validation_failure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = iter([SimpleNamespace(passed=True), SimpleNamespace(passed=False)])
    monkeypatch.setattr(runner_module, "validate", lambda _dataset: next(calls))
    with pytest.raises(SyntheaGenerationError, match="after augmentation"):
        generate_synthea_cohort(_config(tmp_path))


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_failed_raw_validation_keeps_raw_export(tmp_path: Path) -> None:
    def incomplete(raw_dir: Path, _size: int, _seed: int) -> None:
        (raw_dir / "patients.csv").write_text("bad", encoding="utf-8")

    with pytest.raises(SyntheaGenerationError, match="missing CSV"):
        generate_synthea_cohort(_config(tmp_path, batch_generator=incomplete))
    run = tmp_path / "output" / "ordinary_incidence" / "test-run"
    raw_dirs = list((run / "raw").glob("*"))
    assert raw_dirs
    batch_manifest = json.loads(
        (run / "batch_manifests" / "batch_00000.json").read_text()
    )
    generation_manifest = json.loads((run / "manifest.json").read_text())
    assert batch_manifest["state"] == "failed"
    assert batch_manifest["failure_stage"] == "step_2_canonicalization"
    assert "missing CSV" in batch_manifest["diagnostic"]
    assert generation_manifest["state"] == "failed"
    assert generation_manifest["failure_stage"] == "step_2_canonicalization"


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_unexpected_batch_error_is_logged_and_persisted(tmp_path: Path) -> None:
    def exploding(_raw_dir: Path, _size: int, _seed: int) -> None:
        raise ValueError("generator setup failed")

    with pytest.raises(SyntheaGenerationError, match="generator setup failed"):
        generate_synthea_cohort(_config(tmp_path, batch_generator=exploding))

    run = tmp_path / "output" / "ordinary_incidence" / "test-run"
    batch_manifest = json.loads(
        (run / "batch_manifests" / "batch_00000.json").read_text()
    )
    generation_manifest = json.loads((run / "manifest.json").read_text())
    assert batch_manifest["failure_stage"] == "step_1_raw_generation"
    assert batch_manifest["diagnostic"] == "ValueError: generator setup failed"
    assert generation_manifest["failure_stage"] == "step_1_raw_generation"
    assert generation_manifest["diagnostic"] == "ValueError: generator setup failed"


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_generation_cli_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthea_runner",
            "--cohort-class",
            "ordinary_incidence",
            "--batches",
            "1",
            "--patients-per-batch",
            "100",
            "--seed",
            "20260815",
            "--output-root",
            "data/synthetic_longitudinal/smoke",
            "--dry-run",
            "true",
        ],
    )
    assert cli_main() == 0


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_cli_module_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthea_runner",
            "--cohort-class",
            "ordinary_incidence",
            "--batches",
            "1",
            "--patients-per-batch",
            "1",
            "--seed",
            "1",
            "--output-root",
            "out",
            "--dry-run",
            "true",
        ],
    )
    with pytest.raises(SystemExit):
        runpy.run_path(
            str(
                Path(__file__).parents[2]
                / "src"
                / "metaboguard"
                / "data"
                / "__main__.py"
            ),
            run_name="__main__",
        )


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_generation_cli_runs_configured_generator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Manifest:
        run_id = "cli-run"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthea_runner",
            "--cohort-class",
            "ordinary_incidence",
            "--batches",
            "1",
            "--patients-per-batch",
            "1",
            "--seed",
            "1",
            "--output-root",
            str(tmp_path),
            "--dry-run",
            "false",
            "--jar-path",
            str(tmp_path / "jar"),
            "--jar-sha256",
            "hash",
        ],
    )
    monkeypatch.setattr(
        "metaboguard.data.__main__.generate_synthea_cohort",
        lambda _config: Manifest(),
    )
    assert cli_main() == 0


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_runner_module_cli_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["synthea_runner", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_path(str(Path(runner_module.__file__ or "")), run_name="__main__")


@pytest.mark.skipif(
    sys.platform == "linux" and "GITHUB_ACTIONS" in os.environ,
    reason="Skip on GitHub Actions",
)
def test_generation_cli_rejects_nonpositive_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthea_runner",
            "--cohort-class",
            "ordinary_incidence",
            "--batches",
            "0",
            "--patients-per-batch",
            "100",
            "--seed",
            "1",
            "--output-root",
            "out",
            "--dry-run",
            "false",
        ],
    )
    with pytest.raises(SystemExit, match="positive"):
        cli_main()
