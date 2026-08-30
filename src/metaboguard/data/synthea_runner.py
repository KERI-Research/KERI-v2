"""Hash-pinned, batched Synthea execution boundary."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from metaboguard.cli.progress import ProgressReporter
from metaboguard.data.augmentation import (
    augment_c_peptide,
    augment_ca19_9,
    augment_insulin,
)
from metaboguard.data.augmentation.common import write_assumptions
from metaboguard.data.canonical import CanonicalDataset, to_canonical
from metaboguard.data.capability import build_capability_report
from metaboguard.data.manifests import (
    CohortClass,
    GenerationManifest,
    assert_same_cohort_class,
    config_sha256,
    derive_batch_seed,
    file_sha256,
    git_sha,
    new_run_id,
    runtime_metadata,
)
from metaboguard.data.schema import (
    ClinicalEvent,
    ConditionRecord,
    OutcomeRecord,
    Patient,
)
from metaboguard.data.validation import validate

RawBatchGenerator = Callable[[Path, int, int], Sequence[str] | None]

logger = logging.getLogger(__name__)


class SyntheaGenerationError(RuntimeError):
    """Raised when a generation precondition or batch contract fails."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "unknown",
        batch_index: int | None = None,
        diagnostic: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.batch_index = batch_index
        self.diagnostic = diagnostic or message


def _failure_diagnostic(error: Exception) -> str:
    if isinstance(error, SyntheaGenerationError):
        return error.diagnostic
    return f"{type(error).__name__}: {error}"


def _write_failed_batch_manifest(
    path: Path,
    config: SyntheaGenerationConfig,
    batch_index: int,
    seed: int,
    raw_dir: Path,
    stage: str,
    error: Exception,
    command: list[str],
) -> None:
    cause = error.__cause__
    return_code = getattr(cause, "returncode", None)
    path.write_text(
        json.dumps(
            {
                "batch_index": batch_index,
                "seed": seed,
                "target_size": config.batch_size,
                "state": "failed",
                "failure_stage": stage,
                "diagnostic": _failure_diagnostic(error),
                "generation_command": command,
                "log_path": str(raw_dir / "synthea.log"),
                "return_code": return_code,
                "java_executable": config.java_executable,
                "jvm_options": list(config.jvm_options),
                "exporter_base_directory": raw_dir.resolve().as_posix(),
                "simulation_only": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


@dataclass(slots=True)
class SyntheaGenerationConfig:
    """Configuration for one reproducible cohort generation run."""

    cohort_class: CohortClass
    root_seed: int
    target_patients: int
    batch_size: int
    jar_path: Path
    jar_sha256: str
    synthea_version: str
    geography: str
    min_age: int
    max_age: int
    output_root: Path
    java_executable: str = "java"
    jvm_options: tuple[str, ...] = ()
    run_id: str | None = None
    manifest_filename: str = "manifest.json"
    population_settings: dict[str, object] = field(default_factory=dict)
    enabled_cancer_sites: tuple[str, ...] = (
        "pancreas",
        "colorectal",
        "breast",
        "prostate",
        "lung",
    )
    batch_generator: RawBatchGenerator | None = None

    def __post_init__(self) -> None:
        if self.target_patients < 1 or self.batch_size < 1:
            raise ValueError("target_patients and batch_size must be positive")
        if self.min_age < 0 or self.max_age < self.min_age:
            raise ValueError("Invalid age range")
        assert_same_cohort_class([self.cohort_class])

    @property
    def batch_count(self) -> int:
        return (self.target_patients + self.batch_size - 1) // self.batch_size

    def as_dict(self) -> dict[str, object]:
        return {
            "cohort_class": self.cohort_class,
            "root_seed": self.root_seed,
            "target_patients": self.target_patients,
            "batch_size": self.batch_size,
            "jar_path": str(self.jar_path),
            "jar_sha256": self.jar_sha256,
            "synthea_version": self.synthea_version,
            "geography": self.geography,
            "min_age": self.min_age,
            "max_age": self.max_age,
            "output_root": str(self.output_root),
            "java_executable": self.java_executable,
            "jvm_options": self.jvm_options,
            "manifest_filename": self.manifest_filename,
            "population_settings": self.population_settings,
            "enabled_cancer_sites": self.enabled_cancer_sites,
        }


def _verify_jar(config: SyntheaGenerationConfig) -> str:
    if not config.jar_sha256:
        raise SyntheaGenerationError("A configured Synthea JAR SHA-256 is required")
    if not config.jar_path.is_file():
        raise SyntheaGenerationError(f"Synthea JAR does not exist: {config.jar_path}")
    actual = file_sha256(config.jar_path)
    if actual != config.jar_sha256:
        raise SyntheaGenerationError(
            f"Synthea JAR hash mismatch: expected {config.jar_sha256}, received {actual}"
        )
    return actual


def _java_version(executable: str) -> str:
    try:
        result = subprocess.run(
            [executable, "-version"], capture_output=True, check=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SyntheaGenerationError("A working Java runtime is required") from error
    output = (result.stderr or result.stdout).splitlines()
    return output[0] if output else "unknown"


def _java_retry_options(config: SyntheaGenerationConfig) -> tuple[str, ...]:
    """Use a conservative JVM profile after a fatal JVM crash.

    The default production profile can drive Java 17 on Windows into a fatal VM
    crash with SerialGC and a large heap. When the VM dies, retry with a smaller
    heap and a more stable collector rather than reusing the crash-prone flags.
    """
    filtered: list[str] = []
    for option in config.jvm_options:
        lowered = option.lower()
        if lowered.startswith("-xmx") or lowered.startswith("-xms"):
            continue
        if lowered.startswith("-xx:+useserialgc"):
            continue
        if lowered.startswith("-xx:activeprocessorcount="):
            continue
        filtered.append(option)

    fallback = ("-Xmx4g", "-XX:+UseG1GC")
    return (*fallback, *filtered, "-Xint")


def _run_java_batch(
    config: SyntheaGenerationConfig, raw_dir: Path, seed: int
) -> list[str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = [
        config.java_executable,
        *config.jvm_options,
        "-XX:ErrorFile=" + (raw_dir / "hs_err_pid%p.log").resolve().as_posix(),
        "-jar",
        str(config.jar_path),
        "-p",
        str(config.batch_size),
        "-s",
        str(seed),
        "--exporter.csv.export=true",
        "--exporter.fhir.export=false",
        "--exporter.baseDirectory=" + raw_dir.resolve().as_posix(),
        "--exporter.years_of_history=10",
    ]
    log_path = raw_dir / "synthea.log"
    previous_crash_logs = set(raw_dir.glob("hs_err_pid*.log"))
    try:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command, check=True, stdout=log, stderr=subprocess.STDOUT, text=True
            )
    except (OSError, subprocess.CalledProcessError) as error:
        crash_logs = set(raw_dir.glob("hs_err_pid*.log")) - previous_crash_logs
        if isinstance(error, subprocess.CalledProcessError) and crash_logs:
            safe_command = [
                config.java_executable,
                *_java_retry_options(config),
                "-XX:ErrorFile="
                + (raw_dir / "hs_err_pid%p.retry.log").resolve().as_posix(),
                "-jar",
                str(config.jar_path),
                "-p",
                str(config.batch_size),
                "-s",
                str(seed),
                "--exporter.csv.export=true",
                "--exporter.fhir.export=false",
                "--exporter.baseDirectory=" + raw_dir.resolve().as_posix(),
                "--exporter.years_of_history=10",
            ]
            logger.warning(
                "Synthea JVM crashed; retrying batch %s with a more stable JVM profile",
                seed,
            )
            try:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(
                        "\nRetrying after JVM fatal error with a stable GC profile\n"
                    )
                    subprocess.run(
                        safe_command,
                        check=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                command = safe_command
            except (OSError, subprocess.CalledProcessError) as retry_error:
                raise SyntheaGenerationError(
                    f"Synthea batch failed: {' '.join(safe_command)}"
                ) from retry_error
        else:
            raise SyntheaGenerationError(
                f"Synthea batch failed: {' '.join(command)}"
            ) from error
    exported_dir = raw_dir / "csv"
    if exported_dir.is_dir():
        for csv_path in exported_dir.glob("*.csv"):
            shutil.move(str(csv_path), raw_dir / csv_path.name)
        exported_dir.rmdir()
    return command


def _validate_raw_export(raw_dir: Path) -> None:
    required = {
        "patients",
        "encounters",
        "observations",
        "conditions",
        "medications",
        "procedures",
    }
    missing = [
        name for name in sorted(required) if not (raw_dir / f"{name}.csv").is_file()
    ]
    if missing:
        raise SyntheaGenerationError(
            f"Raw Synthea export is missing CSV files: {missing}"
        )


def _read_canonical_dir(path: Path) -> CanonicalDataset:
    def read_models(name: str, model: type[object]) -> list[object]:
        frame = pd.read_parquet(path / f"{name}.parquet")
        records = []
        for row in frame.to_dict(orient="records"):
            normalized = {
                key: (
                    None
                    if value is None or (isinstance(value, float) and np.isnan(value))
                    else value
                )
                for key, value in row.items()
            }
            records.append(model.model_validate(normalized))  # type: ignore[attr-defined]
        return records

    return CanonicalDataset(
        patients=read_models("patients", Patient),  # type: ignore[arg-type]
        events=read_models("events", ClinicalEvent),  # type: ignore[arg-type]
        conditions=read_models("conditions", ConditionRecord),  # type: ignore[arg-type]
        outcomes=read_models("outcomes", OutcomeRecord),  # type: ignore[arg-type]
        output_dir=path,
        date_normalisation_audit=(
            json.loads(
                (path / "date_normalisation_audit.json").read_text(encoding="utf-8")
            )
            if (path / "date_normalisation_audit.json").exists()
            else []
        ),
        date_normalisation_audit_sha256=(
            file_sha256(path / "date_normalisation_audit.json")
            if (path / "date_normalisation_audit.json").exists()
            else ""
        ),
        unit_normalisation_audit=(
            json.loads(
                (path / "unit_normalisation_audit.json").read_text(encoding="utf-8")
            )
            if (path / "unit_normalisation_audit.json").exists()
            else []
        ),
        unit_normalisation_audit_sha256=(
            file_sha256(path / "unit_normalisation_audit.json")
            if (path / "unit_normalisation_audit.json").exists()
            else ""
        ),
    )


def _write_models(dataset: CanonicalDataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, tuple[Sequence[object], list[str]]] = {
        "patients": (
            cast(Sequence[object], dataset.patients),
            ["patient_id", "birth_date", "sex", "ethnicity", "death_date"],
        ),
        "events": (
            cast(Sequence[object], dataset.events),
            [
                "patient_id",
                "event_date",
                "age_at_event",
                "encounter_type",
                "feature_name",
                "value",
                "unit",
                "is_missing",
                "provenance",
                "augmentation_module",
            ],
        ),
        "conditions": (
            cast(Sequence[object], dataset.conditions),
            [
                "patient_id",
                "condition_code",
                "condition_system",
                "condition_display",
                "onset_date",
                "resolved_date",
                "category",
                "diabetes_type",
                "cancer_site",
            ],
        ),
        "outcomes": (
            cast(Sequence[object], dataset.outcomes),
            ["patient_id", "outcome_type", "outcome_date", "source_condition_code"],
        ),
    }
    for name, (rows, columns) in tables.items():
        frame = pd.DataFrame(
            [{column: getattr(row, column) for column in columns} for row in rows],
            columns=columns,
        )
        if not frame.empty:
            frame = frame.sort_values(columns, kind="mergesort").reset_index(drop=True)
        frame.to_parquet(
            output_dir / f"{name}.parquet",
            index=False,
            engine="pyarrow",
            compression="zstd",
        )


def _merge_datasets(
    datasets: list[CanonicalDataset], output_dir: Path
) -> CanonicalDataset:
    patients = [patient for dataset in datasets for patient in dataset.patients]
    patient_ids = [patient.patient_id for patient in patients]
    if len(patient_ids) != len(set(patient_ids)):
        raise SyntheaGenerationError(
            "Duplicate patient IDs detected across Synthea batches"
        )
    merged = CanonicalDataset(
        patients=patients,
        events=[event for dataset in datasets for event in dataset.events],
        conditions=[
            condition for dataset in datasets for condition in dataset.conditions
        ],
        outcomes=[outcome for dataset in datasets for outcome in dataset.outcomes],
        dropped_loinc_codes={
            code: sum(dataset.dropped_loinc_codes.get(code, 0) for dataset in datasets)
            for code in {
                code for dataset in datasets for code in dataset.dropped_loinc_codes
            }
        },
        output_dir=output_dir,
        date_normalisation_audit=[
            entry for dataset in datasets for entry in dataset.date_normalisation_audit
        ],
        unit_normalisation_audit=[
            entry for dataset in datasets for entry in dataset.unit_normalisation_audit
        ],
    )
    _write_models(merged, output_dir)
    audit_path = output_dir / "date_normalisation_audit.json"
    audit_path.write_text(
        json.dumps(merged.date_normalisation_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    merged.date_normalisation_audit_sha256 = file_sha256(audit_path)
    unit_audit_path = output_dir / "unit_normalisation_audit.json"
    unit_audit_path.write_text(
        json.dumps(merged.unit_normalisation_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    merged.unit_normalisation_audit_sha256 = file_sha256(unit_audit_path)
    digest = hashlib.sha256()
    for path in sorted(output_dir.glob("*.parquet")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(audit_path.name.encode("utf-8"))
    digest.update(audit_path.read_bytes())
    digest.update(unit_audit_path.name.encode("utf-8"))
    digest.update(unit_audit_path.read_bytes())
    merged.dataset_sha256 = digest.hexdigest()
    return merged


def _raw_hash(raw_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(raw_dir.glob("*.csv")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _canonical_dir_hash(canonical_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(canonical_dir.glob("*.parquet")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    audit_path = canonical_dir / "date_normalisation_audit.json"
    if audit_path.exists():
        digest.update(audit_path.name.encode("utf-8"))
        digest.update(audit_path.read_bytes())
    unit_audit_path = canonical_dir / "unit_normalisation_audit.json"
    if unit_audit_path.exists():
        digest.update(unit_audit_path.name.encode("utf-8"))
        digest.update(unit_audit_path.read_bytes())
    return digest.hexdigest()


def _complete_batch(
    manifest_path: Path, canonical_dir: Path
) -> tuple[CanonicalDataset, str] | None:
    if not manifest_path.is_file() or not canonical_dir.is_dir():
        return None
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hash = metadata["canonical_sha256"]
        if (
            metadata.get("state") != "complete"
            or _canonical_dir_hash(canonical_dir) != expected_hash
        ):
            return None
        return _read_canonical_dir(canonical_dir), str(metadata["raw_sha256"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _augment_dataset(
    dataset: CanonicalDataset, seed: int, output_dir: Path
) -> tuple[CanonicalDataset, list[dict[str, str]]]:
    current = dataset
    records: list[dict[str, str]] = []
    for offset, augmenter in enumerate(
        (augment_insulin, augment_c_peptide, augment_ca19_9)
    ):
        module_seed = derive_batch_seed(seed, offset)
        result = augmenter(current, np.random.default_rng(module_seed), module_seed)
        assumptions_path = write_assumptions(result, output_dir)
        current = result.dataset
        records.append(
            {
                "module_name": result.module_name,
                "module_version": result.module_version,
                "module_sha256": file_sha256(
                    Path(inspect.getsourcefile(augmenter) or "")
                ),
                "assumptions_sha256": file_sha256(assumptions_path),
            }
        )
    return current, records


def _generate_synthea_cohort(config: SyntheaGenerationConfig) -> GenerationManifest:
    """Generate, canonicalize, validate, and manifest one simulation cohort."""
    jar_hash = _verify_jar(config)
    java_version = (
        "injected-generator"
        if config.batch_generator
        else _java_version(config.java_executable)
    )
    run_id = config.run_id or new_run_id(config.cohort_class, config.root_seed)
    run_dir = config.output_root / config.cohort_class / run_id
    batch_root = run_dir / "batch_manifests"
    batch_canonical_root = run_dir / "canonical" / "batches"
    augmentation_root = run_dir / "augmentation"
    final_canonical_root = run_dir / "canonical"
    batch_root.mkdir(parents=True, exist_ok=True)
    batch_canonical_root.mkdir(parents=True, exist_ok=True)
    manifest = GenerationManifest(
        cohort_class=config.cohort_class,
        run_id=run_id,
        root_seed=config.root_seed,
        batch_seeds=[
            derive_batch_seed(config.root_seed, index)
            for index in range(config.batch_count)
        ],
        config_hash=config_sha256(config.as_dict()),
        git_sha=git_sha(),
        synthea_version=config.synthea_version,
        jar_sha256=jar_hash,
        java_version=java_version,
        operating_system=runtime_metadata(java_version)["operating_system"],
        python_version=runtime_metadata(java_version)["python_version"],
        geography=config.geography,
        age_range=(config.min_age, config.max_age),
        population_settings=config.population_settings,
        sampling_stratum=config.cohort_class,
        inverse_probability_weight_policy=(
            "configured_sampling_weight_required_for_enriched_classes"
        ),
        generation_commands=[],
    )
    manifest.write(run_dir / config.manifest_filename)
    batch_datasets: list[CanonicalDataset] = []
    raw_hashes: list[str] = []
    progress = ProgressReporter(
        config.batch_count, label=f"{config.cohort_class} batches"
    )
    for batch_index, seed in enumerate(manifest.batch_seeds):
        raw_dir = run_dir / "raw" / f"batch_{batch_index:05d}"
        canonical_dir = batch_canonical_root / f"batch_{batch_index:05d}"
        batch_manifest_path = batch_root / f"batch_{batch_index:05d}.json"
        completed = _complete_batch(batch_manifest_path, canonical_dir)
        if completed is not None:
            batch_dataset, previous_raw_hash = completed
            batch_datasets.append(batch_dataset)
            raw_hashes.append(previous_raw_hash)
            manifest.generation_commands.append(["resumed", str(batch_index)])
            progress.update(batch_index + 1, suffix="resumed")
            continue
        raw_dir.mkdir(parents=True, exist_ok=True)
        command: list[str] = []
        stage = "step_1_raw_generation"
        try:
            command = (
                _run_java_batch(config, raw_dir, seed)
                if config.batch_generator is None
                else list(
                    config.batch_generator(raw_dir, config.batch_size, seed) or []
                )
            )
            stage = "step_2_canonicalization"
            _validate_raw_export(raw_dir)
            raw_hashes.append(_raw_hash(raw_dir))
            dataset = to_canonical(
                raw_dir, source="synthea", source_version=config.synthea_version
            )
            validation = validate(dataset)
            if not validation.passed:
                raise SyntheaGenerationError(
                    f"Canonical validation failed for batch {batch_index}"
                )
            if dataset.output_dir is None:
                raise SyntheaGenerationError(
                    "Canonical converter did not return an output directory"
                )
            shutil.copytree(dataset.output_dir, canonical_dir, dirs_exist_ok=True)
            batch_datasets.append(_read_canonical_dir(canonical_dir))
            batch_manifest_path.write_text(
                json.dumps(
                    {
                        "batch_index": batch_index,
                        "seed": seed,
                        "target_size": config.batch_size,
                        "raw_sha256": raw_hashes[-1],
                        "canonical_sha256": dataset.dataset_sha256,
                        "state": "complete",
                        "java_executable": config.java_executable,
                        "jvm_options": list(config.jvm_options),
                        "exporter_base_directory": raw_dir.resolve().as_posix(),
                        "return_code": 0,
                        "simulation_only": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest.generation_commands.append(command)
            shutil.rmtree(raw_dir)
            progress.update(batch_index + 1)
        except Exception as error:
            cause = error
            if isinstance(error, SyntheaGenerationError):
                error.stage = stage
                error.batch_index = batch_index
            else:
                error = SyntheaGenerationError(
                    _failure_diagnostic(error),
                    stage=stage,
                    batch_index=batch_index,
                )
            _write_failed_batch_manifest(
                batch_manifest_path,
                config,
                batch_index,
                seed,
                raw_dir,
                stage,
                error,
                command,
            )
            logger.exception("Synthea batch %s failed during %s", batch_index, stage)
            progress.close()
            raise error from cause
    progress.close()
    raw_root = run_dir / "raw"
    if raw_root.is_dir() and not any(raw_root.iterdir()):
        raw_root.rmdir()
    merged = _merge_datasets(batch_datasets, final_canonical_root)
    merged.cohort_class = config.cohort_class
    merged.cohort_metadata = {
        "sampling_stratum": config.cohort_class,
        "inverse_probability_weight_policy": (
            "configured_sampling_weight_required_for_enriched_classes"
        ),
        "enabled_cancer_sites": list(config.enabled_cancer_sites),
    }
    (final_canonical_root / "cohort_metadata.json").write_text(
        json.dumps(merged.cohort_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    augmented, augmentation_records = _augment_dataset(
        merged, config.root_seed, augmentation_root
    )
    _write_models(augmented, final_canonical_root)
    final_validation = validate(augmented)
    if not final_validation.passed:
        raise SyntheaGenerationError("Canonical validation failed after augmentation")
    reports_dir = run_dir / "reports"
    capability = build_capability_report(augmented, config.cohort_class)
    capability.write(reports_dir / "cohort_capability_report.json")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "generation_report.json").write_text(
        json.dumps({"state": capability.state, "simulation_only": True}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2(
        Path("artifacts/validation/dataset_validation_report.json"),
        reports_dir / "dataset_validation_report.json",
    )
    shutil.copy2(
        final_canonical_root / "date_normalisation_audit.json",
        reports_dir / "date_normalisation_audit.json",
    )
    shutil.copy2(
        final_canonical_root / "unit_normalisation_audit.json",
        reports_dir / "unit_normalisation_audit.json",
    )
    manifest.generated_patient_count = len(merged.patients)
    manifest.converted_patient_count = len(merged.patients)
    manifest.raw_dataset_sha256 = hashlib.sha256(
        "".join(sorted(raw_hashes)).encode("ascii")
    ).hexdigest()
    manifest.canonical_dataset_sha256 = merged.dataset_sha256
    manifest.date_normalisation_audit_sha256 = merged.date_normalisation_audit_sha256
    manifest.unit_normalisation_audit_sha256 = merged.unit_normalisation_audit_sha256
    manifest.augmentation_modules = augmentation_records
    manifest.event_counts = {
        "patients": len(augmented.patients),
        "events": len(augmented.events),
        "conditions": len(augmented.conditions),
        "outcomes": len(augmented.outcomes),
        "diabetes_type": capability.diabetes_type_counts,
        "cancer_site": capability.incident_cancer_count_by_site,
        "horizon_eligible": capability.horizon_eligible_event_counts,
        "deaths_before_horizon": capability.deaths_before_horizon,
        "capability_state": capability.state,
    }
    manifest.state = "complete"
    manifest.cohort_status = (
        "cohort_insufficient"
        if len(merged.patients) < config.target_patients
        else "complete"
    )
    manifest.write(run_dir / config.manifest_filename)
    return manifest


def generate_synthea_cohort(config: SyntheaGenerationConfig) -> GenerationManifest:
    """Generate, canonicalize, validate, and manifest one simulation cohort."""
    run_id = config.run_id or new_run_id(config.cohort_class, config.root_seed)
    manifest_path = (
        config.output_root / config.cohort_class / run_id / config.manifest_filename
    )
    try:
        return _generate_synthea_cohort(config)
    except Exception as error:
        stage = (
            error.stage
            if isinstance(error, SyntheaGenerationError)
            else "step_3_augmentation"
        )
        diagnostic = _failure_diagnostic(error)
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload.update(
                {"state": "failed", "failure_stage": stage, "diagnostic": diagnostic}
            )
            manifest_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        logger.exception("Synthea generation failed during %s: %s", stage, diagnostic)
        if isinstance(error, SyntheaGenerationError):
            raise
        raise SyntheaGenerationError(
            diagnostic, stage=stage, diagnostic=diagnostic
        ) from error


if __name__ == "__main__":
    from metaboguard.data.__main__ import main

    raise SystemExit(main())
