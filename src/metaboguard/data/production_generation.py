"""Step 7 production-scale synthetic generation orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from metaboguard.cohort.manifests import construct_endpoint_cohort
from metaboguard.cohort.protocol import EndpointProtocol, SplitConfig, load_endpoint_registry
from metaboguard.cohort.splits import build_splits, write_split_artifacts
from metaboguard.config import load_config
from metaboguard.data.manifests import CohortClass, file_sha256
from metaboguard.data.production_manifests import (
    ProductionBatchRecord,
    ProductionRunManifest,
    new_production_manifest,
)
from metaboguard.data.synthea_runner import (
    SyntheaGenerationConfig,
    _read_canonical_dir,
    generate_synthea_cohort,
)
from metaboguard.features.extraction import extract_features
from metaboguard.features.manifests import write_feature_artifacts
from metaboguard.readiness.manifests import build_readiness_bundle
from metaboguard.readiness.production_feasibility import (
    build_production_feasibility,
    write_production_feasibility,
)


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
                raise TypeError(f"Population sizes and seeds must be lists for {cohort_class}")
            if len(populations) != len(seeds):
                raise ValueError(f"Population and seed counts differ for {cohort_class}")
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
            jvm_options=tuple(str(option) for option in execution.get("jvm_options", [])),
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
    return ProductionGenerationConfig.from_mapping(load_config()["production_generation"])


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
        java_executable=str(production.get("java_executable", synthea["java_executable"])),
        jvm_options=tuple(str(option) for option in production.get("jvm_options", [])),
        run_id=run_id,
        manifest_filename="generation_manifest.json",
        enabled_cancer_sites=tuple(str(site) for site in synthea["enabled_cancer_sites"]),
    )


def _run_frozen_pipeline(
    run_path: Path, endpoints: tuple[str, ...], generation_manifest_sha256: str
) -> None:
    """Run Steps 4-6 for each endpoint independently."""
    dataset = _read_canonical_dir(run_path / "canonical")
    dataset.cohort_class = str(
        json.loads((run_path / "generation_manifest.json").read_text(encoding="utf-8"))[
            "cohort_class"
        ]
    )
    registry = load_endpoint_registry()
    for endpoint_id in endpoints:
        endpoint: EndpointProtocol = registry[endpoint_id]
        cohort_path = run_path / "cohort" / endpoint_id
        cohort = construct_endpoint_cohort(
            dataset, endpoint, cohort_path, generation_manifest_sha256
        )
        split = build_splits(cohort, SplitConfig(root_seed=1729))
        write_split_artifacts(split, cast(list[object], cohort.labels), cohort_path / "splits")
        feature_dataset = extract_features(
            dataset,
            cohort.patient_indexes,
            split.assignments,
            source_cohort_manifest_sha256=file_sha256(cohort_path / "cohort_manifest.json"),
            source_split_manifest_sha256=file_sha256(
                cohort_path / "splits" / "split_manifest.json"
            ),
        )
        write_feature_artifacts(
            feature_dataset,
            run_path / "features" / endpoint_id,
            cohort_path / "cohort_manifest.json",
        )
        build_readiness_bundle(run_path, endpoint_id)
        rows = build_production_feasibility(
            run_path,
            endpoint_id,
            population_target=len(dataset.patients),
            run_id=run_path.name,
        )
        write_production_feasibility(run_path, rows)


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
                source_sha256=str(payload.get("raw_sha256", "")),
                canonical_sha256=str(payload.get("canonical_sha256", "")),
                failure_stage=None if payload.get("state") == "complete" else "generation",
                diagnostic=None if payload.get("state") == "complete" else "batch incomplete",
                java_executable=str(payload.get("java_executable", "")),
                jvm_options=[str(option) for option in payload.get("jvm_options", [])],
                exporter_base_directory=str(payload.get("exporter_base_directory", "")),
            )
        )
    return records


def generate_production_run(
    plan: ProductionCohortPlan,
    output_root: Path,
    *,
    run_pipeline: bool = True,
    generation_runner: Callable[[SyntheaGenerationConfig], object] = generate_synthea_cohort,
) -> ProductionRunManifest:
    """Generate one class-specific run and optionally execute frozen Steps 4-6."""
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
    manifest.status = "running"
    manifest.write(manifest_path)
    try:
        generation_config = _generation_config(plan, output_root, manifest.run_id)
        production_config = load_production_config()
        for _attempt in range(1, production_config.max_attempts + 1):  # pragma: no branch
            try:
                generation_runner(generation_config)
                break
            except Exception:
                if _attempt == production_config.max_attempts:
                    raise
        manifest.generated_patient_count = len(_read_canonical_dir(run_path / "canonical").patients)
        manifest.batch_records = _load_batch_records(
            run_path, manifest.started_at, generation_config.batch_size
        )
        manifest.canonical_status = "created"
        if run_pipeline:
            _run_frozen_pipeline(
                run_path,
                load_production_config().endpoints,
                hashlib.sha256((run_path / "generation_manifest.json").read_bytes()).hexdigest(),
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
        manifest.status = "failed"
        manifest.failure_summary = str(error)[:1000]
        manifest.completed_at = datetime.now(UTC).isoformat()
        manifest.write(manifest_path)
        raise
    manifest.write(manifest_path)
    return manifest


def generate_configured_production_runs(
    output_root: Path | None = None, *, run_pipeline: bool = True
) -> list[ProductionRunManifest]:
    """Execute all enabled configured plans independently; never pool classes."""
    config = load_production_config()
    if not config.enabled or not config.simulation_only:
        raise ValueError("Production generation must be enabled in simulation-only mode")
    root = output_root or Path(str(load_config()["synthea"]["output_root"])) / "production"
    return [generate_production_run(plan, root, run_pipeline=run_pipeline) for plan in config.plans]
