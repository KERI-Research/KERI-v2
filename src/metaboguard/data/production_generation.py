"""Step 7 production-scale synthetic generation orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from metaboguard.cli.progress import ProgressReporter
from metaboguard.cohort.manifests import construct_endpoint_cohort
from metaboguard.cohort.protocol import (
    EndpointProtocol,
    SplitConfig,
    load_endpoint_registry,
)
from metaboguard.cohort.splits import build_splits, write_split_artifacts
from metaboguard.config import load_config
from metaboguard.data.manifests import CohortClass, file_sha256
from metaboguard.data.production_manifests import (
    ProductionBatchRecord,
    ProductionRunManifest,
    load_production_manifest,
    new_production_manifest,
)
from metaboguard.data.synthea_runner import (
    SyntheaGenerationConfig,
    _read_canonical_dir,
    generate_synthea_cohort,
)
from metaboguard.features.extraction import extract_features
from metaboguard.features.manifests import (
    StreamingFeatureArtifactWriter,
    write_feature_artifacts,
)
from metaboguard.readiness.manifests import build_readiness_bundle
from metaboguard.readiness.production_feasibility import (
    build_production_feasibility,
    write_production_feasibility,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProductionCohortPlan:
    cohort_class: CohortClass
    population_target: int
    root_seed: int
    augmentation_profile: str
    batch_population_size: int | None = None


@dataclass(frozen=True, slots=True)
class ProductionGenerationConfig:
    enabled: bool
    simulation_only: bool
    plans: tuple[ProductionCohortPlan, ...]
    endpoints: tuple[str, ...]
    horizons_years: tuple[int, ...]
    max_attempts: int
    retry_only_failed_batches: bool
    batch_population_size: int
    jvm_options: tuple[str, ...]
    continue_on_independent_batch_failure: bool
    require_canonical_validation_before_cohort_build: bool
    require_complete_feature_build_before_readiness: bool
    write_per_run_reports: bool
    write_cohort_class_feasibility_summary: bool
    write_cross_run_manifest: bool

    @classmethod
    def from_mapping(cls, mapping: dict[str, object]) -> ProductionGenerationConfig:
        """Parse the production section without inventing defaults."""
        class_mappings = mapping["cohort_classes"]
        if not isinstance(class_mappings, dict):
            raise TypeError("cohort_classes must be a mapping")
        plans: list[ProductionCohortPlan] = []
        for cohort_class, raw in class_mappings.items():
            if not isinstance(raw, dict):
                raise TypeError(f"Configuration for {cohort_class} must be a mapping")
            if not bool(raw["enabled"]):
                continue
            populations = raw["population_sizes"]
            seeds = raw["seeds"]
            if not isinstance(populations, list) or not isinstance(seeds, list):
                raise TypeError(
                    f"Population sizes and seeds must be lists for {cohort_class}"
                )
            if len(populations) != len(seeds):
                raise ValueError(
                    f"Population and seed counts differ for {cohort_class}"
                )
            for population, seed in zip(populations, seeds, strict=True):
                plans.append(
                    ProductionCohortPlan(
                        cohort_class=cast(CohortClass, cohort_class),
                        population_target=int(population),
                        root_seed=int(seed),
                        augmentation_profile=str(raw["augmentation_profile"]),
                    )
                )
        retry = mapping["retry"]
        execution = mapping["execution"]
        reporting = mapping["reporting"]
        if not isinstance(retry, dict):
            raise TypeError("retry configuration must be a mapping")
        if not isinstance(execution, dict):
            raise TypeError("execution configuration must be a mapping")
        if not isinstance(reporting, dict):
            raise TypeError("reporting configuration must be a mapping")
        endpoints = mapping["endpoints"]
        horizons = mapping["horizons_years"]
        if not isinstance(endpoints, list) or not isinstance(horizons, list):
            raise TypeError("endpoints and horizons_years must be lists")
        return cls(
            enabled=bool(mapping["enabled"]),
            simulation_only=bool(mapping["simulation_only"]),
            plans=tuple(plans),
            endpoints=tuple(str(endpoint) for endpoint in endpoints),
            horizons_years=tuple(int(horizon) for horizon in horizons),
            max_attempts=int(retry["max_attempts"]),
            retry_only_failed_batches=bool(retry["retry_only_failed_batches"]),
            batch_population_size=int(execution["batch_population_size"]),
            jvm_options=tuple(
                str(option) for option in execution.get("jvm_options", [])
            ),
            continue_on_independent_batch_failure=bool(
                execution["continue_on_independent_batch_failure"]
            ),
            require_canonical_validation_before_cohort_build=bool(
                execution["require_canonical_validation_before_cohort_build"]
            ),
            require_complete_feature_build_before_readiness=bool(
                execution["require_complete_feature_build_before_readiness"]
            ),
            write_per_run_reports=bool(reporting["write_per_run_reports"]),
            write_cohort_class_feasibility_summary=bool(
                reporting["write_cohort_class_feasibility_summary"]
            ),
            write_cross_run_manifest=bool(reporting["write_cross_run_manifest"]),
        )


def load_production_config() -> ProductionGenerationConfig:
    """Load the configured Step 7 production plan."""
    return ProductionGenerationConfig.from_mapping(
        load_config()["production_generation"]
    )


def _generation_config(
    plan: ProductionCohortPlan, output_root: Path, run_id: str
) -> SyntheaGenerationConfig:
    synthea = load_config()["synthea"]
    production = load_config()["production_generation"]["execution"]
    if not isinstance(production, dict):
        raise TypeError("production execution configuration must be a mapping")
    return SyntheaGenerationConfig(
        cohort_class=plan.cohort_class,
        root_seed=plan.root_seed,
        target_patients=plan.population_target,
        batch_size=min(
            plan.batch_population_size or int(production["batch_population_size"]),
            plan.population_target,
        ),
        jar_path=Path(str(synthea["jar_path"])),
        jar_sha256=str(synthea["jar_sha256"]),
        synthea_version=str(synthea["synthea_version"]),
        geography=str(synthea["geography"]),
        min_age=int(synthea["min_age"]),
        max_age=int(synthea["max_age"]),
        output_root=output_root,
        java_executable=str(
            production.get("java_executable", synthea["java_executable"])
        ),
        jvm_options=tuple(str(option) for option in production.get("jvm_options", [])),
        run_id=run_id,
        manifest_filename="generation_manifest.json",
        enabled_cancer_sites=tuple(
            str(site) for site in synthea["enabled_cancer_sites"]
        ),
    )


def _run_frozen_pipeline(
    run_path: Path, endpoints: tuple[str, ...], generation_manifest_sha256: str
) -> None:
    """Run Steps 4-6 for each endpoint independently."""
    logger.info("Steps 4-6 start: run=%s endpoints=%s", run_path.name, endpoints)
    dataset = _read_canonical_dir(run_path / "canonical")
    logger.info(
        "Steps 4-6: canonical dataset loaded for run=%s; patients=%s",
        run_path.name,
        len(dataset.patients),
    )
    dataset.cohort_class = str(
        json.loads((run_path / "generation_manifest.json").read_text(encoding="utf-8"))[
            "cohort_class"
        ]
    )
    registry = load_endpoint_registry()
    use_streaming_features = True
    progress = ProgressReporter(len(endpoints), label="endpoints (steps 4-6)")
    for endpoint_index, endpoint_id in enumerate(endpoints):
        logger.info(
            "Steps 4-6: starting endpoint=%s (%s/%s)",
            endpoint_id,
            endpoint_index + 1,
            len(endpoints),
        )
        try:
            endpoint: EndpointProtocol = registry[endpoint_id]
            cohort_path = run_path / "cohort" / endpoint_id
            logger.info("Steps 4-6: constructing cohort for %s", endpoint_id)
            cohort = construct_endpoint_cohort(
                dataset, endpoint, cohort_path, generation_manifest_sha256
            )
            logger.info(
                "Steps 4-6: cohort built for %s; patient_indexes=%s",
                endpoint_id,
                len(cohort.patient_indexes),
            )
            split = build_splits(cohort, SplitConfig(root_seed=1729))
            logger.info("Steps 4-6: splits built for %s", endpoint_id)
            write_split_artifacts(
                split, cast(list[object], cohort.labels), cohort_path / "splits"
            )
            logger.info("Steps 4-6: split artifacts written for %s", endpoint_id)
            feature_output = run_path / "features" / endpoint_id
            if use_streaming_features:
                logger.info(
                    "Steps 4-6: feature extraction in streaming mode for %s",
                    endpoint_id,
                )
                feature_writer = StreamingFeatureArtifactWriter(
                    feature_output, cohort_path / "cohort_manifest.json"
                )
                feature_dataset = extract_features(
                    dataset,
                    cohort.patient_indexes,
                    split.assignments,
                    source_cohort_manifest_sha256=file_sha256(
                        cohort_path / "cohort_manifest.json"
                    ),
                    source_split_manifest_sha256=file_sha256(
                        cohort_path / "splits" / "split_manifest.json"
                    ),
                    batch_size=25,
                    batch_callback=feature_writer.write_batch,
                )
                logger.info(
                    "Steps 4-6: feature extraction finished for %s; rows=%s",
                    endpoint_id,
                    len(feature_dataset.rows),
                )
                if feature_writer.row_count:
                    feature_writer.close()
                else:
                    write_feature_artifacts(
                        feature_dataset,
                        feature_output,
                        cohort_path / "cohort_manifest.json",
                    )
            else:
                logger.info(
                    "Steps 4-6: feature extraction in memory mode for %s",
                    endpoint_id,
                )
                feature_dataset = extract_features(
                    dataset,
                    cohort.patient_indexes,
                    split.assignments,
                    source_cohort_manifest_sha256=file_sha256(
                        cohort_path / "cohort_manifest.json"
                    ),
                    source_split_manifest_sha256=file_sha256(
                        cohort_path / "splits" / "split_manifest.json"
                    ),
                )
                logger.info(
                    "Steps 4-6: feature matrix assembled for %s; rows=%s",
                    endpoint_id,
                    len(feature_dataset.rows),
                )
                write_feature_artifacts(
                    feature_dataset,
                    feature_output,
                    cohort_path / "cohort_manifest.json",
                )
            logger.info("Steps 4-6: building readiness bundle for %s", endpoint_id)
            build_readiness_bundle(run_path, endpoint_id)
            logger.info("Steps 4-6: readiness bundle built for %s", endpoint_id)
            rows = build_production_feasibility(
                run_path,
                endpoint_id,
                population_target=len(dataset.patients),
                run_id=run_path.name,
            )
            logger.info(
                "Steps 4-6: feasibility rows computed for %s; count=%s",
                endpoint_id,
                len(rows),
            )
            write_production_feasibility(run_path, rows)
            logger.info("Steps 4-6: feasibility report written for %s", endpoint_id)
            progress.update(endpoint_index + 1, suffix=endpoint_id)
        except Exception as error:
            diagnostic = f"{type(error).__name__}: {error}"
            logger.exception(
                "Steps 4-6 failed for endpoint %s: %s", endpoint_id, diagnostic
            )
            progress.close()
            raise RuntimeError(
                f"Steps 4-6 failed for endpoint {endpoint_id}: {diagnostic}"
            ) from error
    progress.close()


def _load_batch_records(
    run_path: Path, started_at: str, default_target_size: int
) -> list[ProductionBatchRecord]:
    records: list[ProductionBatchRecord] = []
    for path in sorted((run_path / "batch_manifests").glob("batch_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            ProductionBatchRecord(
                batch_index=int(payload["batch_index"]),
                seed=int(payload["seed"]),
                target_size=int(payload.get("target_size", 0) or default_target_size),
                source_output_path=str(run_path / "raw" / path.stem),
                status="completed" if payload.get("state") == "complete" else "failed",
                started_at=started_at,
                completed_at=started_at if payload.get("state") == "complete" else None,
                return_code=(
                    int(payload["return_code"])
                    if payload.get("return_code") is not None
                    else None
                ),
                source_sha256=str(payload.get("raw_sha256", "")),
                canonical_sha256=str(payload.get("canonical_sha256", "")),
                failure_stage=(
                    None
                    if payload.get("state") == "complete"
                    else str(payload.get("failure_stage", "generation"))
                ),
                diagnostic=(
                    None
                    if payload.get("state") == "complete"
                    else str(payload.get("diagnostic", "batch incomplete"))
                ),
                java_executable=str(payload.get("java_executable", "")),
                jvm_options=[str(option) for option in payload.get("jvm_options", [])],
                exporter_base_directory=str(payload.get("exporter_base_directory", "")),
            )
        )
    return records


def _existing_files(run_path: Path, relative_paths: list[str]) -> bool:
    return all((run_path / relative_path).is_file() for relative_path in relative_paths)


def _hash_existing_files(run_path: Path, relative_paths: list[str]) -> dict[str, str]:
    return {
        relative_path: hashlib.sha256(
            (run_path / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in relative_paths
        if (run_path / relative_path).is_file()
    }


def reconcile_production_manifest(
    run_path: Path, *, failure_summary: str | None = None
) -> ProductionRunManifest:
    """Rebuild the top-level manifest from durable generation and stage artifacts."""
    logger.info("Loading production manifest state for run=%s", run_path.name)
    manifest_path = run_path / "manifest.json"
    logger.info("Reconcile manifest: reading manifest at %s", manifest_path)
    manifest = load_production_manifest(manifest_path)
    logger.info(
        "Reconcile manifest: loaded manifest status=%s generated_patient_count=%s",
        manifest.status,
        manifest.generated_patient_count,
    )
    generation_path = run_path / "generation_manifest.json"
    generation = (
        json.loads(generation_path.read_text(encoding="utf-8"))
        if generation_path.is_file()
        else {}
    )
    logger.info(
        "Reconcile manifest: generation exists=%s state=%s",
        generation_path.is_file(),
        generation.get("state"),
    )
    batch_records = _load_batch_records(
        run_path, manifest.started_at, manifest.population_target
    )
    logger.info("Reconcile manifest: loaded %s batch records", len(batch_records))
    canonical_files = [
        "canonical/patients.parquet",
        "canonical/events.parquet",
        "canonical/conditions.parquet",
        "canonical/outcomes.parquet",
    ]
    generation_complete = generation.get("state") == "complete"
    canonical_complete = generation_complete and _existing_files(
        run_path, canonical_files
    )
    endpoint_paths = (
        sorted(path for path in (run_path / "cohort").iterdir() if path.is_dir())
        if (run_path / "cohort").is_dir()
        else []
    )
    cohort_complete = bool(endpoint_paths) and all(
        _existing_files(
            run_path,
            [
                f"cohort/{path.name}/cohort_manifest.json",
                f"cohort/{path.name}/endpoint_protocol.json",
                f"cohort/{path.name}/eligible_indexes.parquet",
            ],
        )
        for path in endpoint_paths
    )
    split_complete = cohort_complete and all(
        _existing_files(
            run_path,
            [
                f"cohort/{path.name}/splits/split_manifest.json",
                f"cohort/{path.name}/splits/split_assignments.parquet",
            ],
        )
        for path in endpoint_paths
    )
    feature_complete = split_complete and all(
        _existing_files(
            run_path,
            [
                f"features/{path.name}/feature_manifest.json",
                f"features/{path.name}/feature_matrix.parquet",
                f"features/{path.name}/feature_lineage.parquet",
            ],
        )
        for path in endpoint_paths
    )
    readiness_complete = feature_complete and all(
        (run_path / "readiness" / path.name / "capability_report.json").is_file()
        and (run_path / "readiness" / path.name / "readiness_manifest.json").is_file()
        for path in endpoint_paths
    )
    feasibility_complete = (
        readiness_complete
        and (run_path / "feasibility" / "endpoint_feasibility_report.json").is_file()
    )
    logger.info(
        "Reconcile stages: canonical=%s cohort=%s split=%s feature=%s readiness=%s feasibility=%s",
        canonical_complete,
        cohort_complete,
        split_complete,
        feature_complete,
        readiness_complete,
        feasibility_complete,
    )
    manifest.generated_patient_count = int(
        generation.get(
            "generated_patient_count", generation.get("converted_patient_count", 0)
        )
    )
    if manifest.generated_patient_count == 0 and canonical_complete:
        manifest.generated_patient_count = len(
            _read_canonical_dir(run_path / "canonical").patients
        )
    manifest.batch_records = batch_records
    manifest.canonical_status = "created" if canonical_complete else "not_created"
    manifest.cohort_status = "created" if cohort_complete else "not_created"
    manifest.split_status = "created" if split_complete else "not_created"
    manifest.feature_status = "created" if feature_complete else "not_created"
    manifest.readiness_status = "created" if readiness_complete else "not_created"
    manifest.feasibility_status = "created" if feasibility_complete else "not_created"
    manifest.artifact_sha256 = _hash_existing_files(
        run_path,
        [
            "generation_manifest.json",
            *[
                f"batch_manifests/{record_path.name}"
                for record_path in sorted(
                    (run_path / "batch_manifests").glob("batch_*.json")
                )
            ],
            *canonical_files,
            *[f"cohort/{path.name}/cohort_manifest.json" for path in endpoint_paths],
            *[
                f"cohort/{path.name}/splits/split_manifest.json"
                for path in endpoint_paths
            ],
            *[f"features/{path.name}/feature_manifest.json" for path in endpoint_paths],
            *[
                f"readiness/{path.name}/readiness_manifest.json"
                for path in endpoint_paths
            ],
            "feasibility/endpoint_feasibility_report.json",
        ],
    )
    readiness_decisions = [
        json.loads(
            (run_path / "readiness" / path.name / "capability_report.json").read_text(
                encoding="utf-8"
            )
        ).get("overall_decision")
        for path in endpoint_paths
        if (run_path / "readiness" / path.name / "capability_report.json").is_file()
    ]
    pipeline_requested = bool(endpoint_paths)
    all_required_stages_complete = canonical_complete and (
        not pipeline_requested
        or all(
            (
                cohort_complete,
                split_complete,
                feature_complete,
                readiness_complete,
                feasibility_complete,
            )
        )
    )
    failed_batches = [record for record in batch_records if record.status == "failed"]
    if failure_summary is not None:
        manifest.status = "failed"
        manifest.failure_summary = failure_summary[:1000]
    elif failed_batches:
        failed_batch = failed_batches[0]
        manifest.status = "failed"
        manifest.failure_summary = (
            f"Batch {failed_batch.batch_index} failed during "
            f"{failed_batch.failure_stage or 'generation'}: "
            f"{failed_batch.diagnostic or 'batch failure'}"
        )[:1000]
    elif not generation_complete:
        manifest.status = "partial"
        manifest.failure_summary = "generation artifacts are incomplete"
    elif all_required_stages_complete and (
        manifest.generated_patient_count >= manifest.population_target
    ):
        manifest.status = (
            "completed_not_ready"
            if any(
                decision in {"not_eligible", "blocked"}
                for decision in readiness_decisions
            )
            else (
                "completed_prototype_ready"
                if readiness_decisions
                and all(
                    decision == "prototype_ready" for decision in readiness_decisions
                )
                else "completed"
            )
        )
        manifest.failure_summary = None
    else:
        manifest.status = "partial"
        manifest.failure_summary = "required production artifacts are incomplete"
    manifest.completed_at = manifest.completed_at or datetime.now(UTC).isoformat()
    logger.info(
        "Reconcile manifest: final status=%s for run=%s",
        manifest.status,
        run_path.name,
    )
    manifest.write(manifest_path)
    return manifest


def generate_production_run(
    plan: ProductionCohortPlan,
    output_root: Path,
    *,
    run_pipeline: bool = False,
    generation_runner: Callable[
        [SyntheaGenerationConfig], object
    ] = generate_synthea_cohort,
) -> ProductionRunManifest:
    """Generate one class-specific run and optionally execute frozen Steps 4-6.

    The default behaviour is to stop after Step 1-3 dataset generation so a user
    can generate the requested patient cohort without hanging in the heavier
    feature/cohort pipeline.
    """
    config = {
        "cohort_class": plan.cohort_class,
        "population_target": plan.population_target,
        "root_seed": plan.root_seed,
        "augmentation_profile": plan.augmentation_profile,
        "batch_population_size": plan.batch_population_size,
    }
    manifest = new_production_manifest(
        plan.cohort_class,
        plan.population_target,
        plan.root_seed,
        plan.augmentation_profile,
        str(load_config()["synthea"]["synthea_version"]),
        config,
    )
    run_path = output_root / plan.cohort_class / manifest.run_id
    manifest_path = run_path / "manifest.json"
    # ! REVIEW
    # The initial manifest is persisted as running, but only the normal return
    # path below reconciles it. A hard termination after this write can leave a
    # completed artifact tree with the stale pre-run manifest seen in the 500 run.
    # ----------
    manifest.status = "running"
    manifest.write(manifest_path)
    try:
        generation_config = _generation_config(plan, output_root, manifest.run_id)
        production_config = load_production_config()
        for _attempt in range(
            1, production_config.max_attempts + 1
        ):  # pragma: no branch
            try:
                generation_runner(generation_config)
                break
            except Exception as error:
                logger.exception(
                    "Generation attempt %s/%s failed for run %s: %s",
                    _attempt,
                    production_config.max_attempts,
                    manifest.run_id,
                    error,
                )
                if _attempt == production_config.max_attempts:
                    raise RuntimeError(
                        "Steps 1-3 generation failed after "
                        f"{_attempt} attempt(s): {type(error).__name__}: {error}"
                    ) from error
        manifest.generated_patient_count = len(
            _read_canonical_dir(run_path / "canonical").patients
        )
        manifest.batch_records = _load_batch_records(
            run_path, manifest.started_at, generation_config.batch_size
        )
        manifest.canonical_status = "created"
        if run_pipeline:
            _run_frozen_pipeline(
                run_path,
                load_production_config().endpoints,
                hashlib.sha256(
                    (run_path / "generation_manifest.json").read_bytes()
                ).hexdigest(),
            )
            manifest.cohort_status = "created"
            manifest.split_status = "created"
            manifest.feature_status = "created"
            manifest.readiness_status = "created"
        manifest.status = (
            "completed"
            if manifest.generated_patient_count >= manifest.population_target
            else "partial"
        )
        manifest.completed_at = datetime.now(UTC).isoformat()
    except Exception as error:
        logger.exception("Production run %s failed: %s", manifest.run_id, error)
        reconcile_production_manifest(run_path, failure_summary=str(error))
        raise
    return reconcile_production_manifest(run_path)


def generate_configured_production_runs(
    output_root: Path | None = None, *, run_pipeline: bool = True
) -> list[ProductionRunManifest]:
    """Execute all enabled configured plans independently; never pool classes."""
    config = load_production_config()
    if not config.enabled or not config.simulation_only:
        raise ValueError(
            "Production generation must be enabled in simulation-only mode"
        )
    root = (
        output_root or Path(str(load_config()["synthea"]["output_root"])) / "production"
    )
    return [
        generate_production_run(plan, root, run_pipeline=run_pipeline)
        for plan in config.plans
    ]
