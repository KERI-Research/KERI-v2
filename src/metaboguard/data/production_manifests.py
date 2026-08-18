"""Strict manifests for Step 7 production-scale synthetic runs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from metaboguard.data.manifests import CohortClass, config_sha256


class ProductionStrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


BatchStatus = Literal["created", "running", "completed", "failed", "partial"]
RunStatus = Literal["created", "running", "completed", "failed", "partial"]
StageStatus = Literal["not_created", "created", "failed"]


class ProductionBatchRecord(ProductionStrictModel):
    batch_index: int = Field(ge=0)
    seed: int
    target_size: int = Field(gt=0)
    source_output_path: str
    status: BatchStatus
    started_at: str
    completed_at: str | None = None
    return_code: int | None = None
    source_sha256: str = ""
    canonical_sha256: str = ""
    failure_stage: str | None = None
    diagnostic: str | None = None
    attempt: int = Field(default=1, ge=1)
    parent_batch_index: int | None = None
    java_executable: str = ""
    jvm_options: list[str] = []
    exporter_base_directory: str = ""


class ProductionRunManifest(ProductionStrictModel):
    manifest_version: Literal["1.0.0"] = "1.0.0"
    run_id: str
    cohort_class: CohortClass
    simulation_only: Literal[True] = True
    pipeline_rehearsal_only: Literal[True] = True
    population_target: int = Field(gt=0)
    root_seed: int
    augmentation_profile: str
    configuration_sha256: str
    synthea_version: str
    started_at: str
    completed_at: str | None = None
    status: RunStatus = "created"
    batch_records: list[ProductionBatchRecord] = []
    canonical_status: StageStatus = "not_created"
    cohort_status: StageStatus = "not_created"
    split_status: StageStatus = "not_created"
    feature_status: StageStatus = "not_created"
    readiness_status: StageStatus = "not_created"
    model_status: Literal["not_created"] = "not_created"
    failure_summary: str | None = None
    generated_patient_count: int = 0

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


def production_run_id(
    cohort_class: CohortClass, population_target: int, seed: int, configuration_sha256: str
) -> str:
    """Create a deterministic, class-specific run identity."""
    digest = hashlib.sha256(configuration_sha256.encode("ascii")).hexdigest()[:12]
    return f"{cohort_class}-{population_target}-{seed}-{digest}"


def new_production_manifest(
    cohort_class: CohortClass,
    population_target: int,
    root_seed: int,
    augmentation_profile: str,
    synthea_version: str,
    configuration: object,
) -> ProductionRunManifest:
    """Create an immutable production-run manifest before generation starts."""
    configuration_hash = config_sha256(configuration)
    return ProductionRunManifest(
        run_id=production_run_id(cohort_class, population_target, root_seed, configuration_hash),
        cohort_class=cohort_class,
        population_target=population_target,
        root_seed=root_seed,
        augmentation_profile=augmentation_profile,
        configuration_sha256=configuration_hash,
        synthea_version=synthea_version,
        started_at=datetime.now(UTC).isoformat(),
    )


def load_production_manifest(path: Path) -> ProductionRunManifest:
    """Load and strictly validate a production manifest."""
    return ProductionRunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def manifest_hash(path: Path) -> str:
    """Hash one persisted production manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_payload(manifest: ProductionRunManifest) -> dict[str, object]:
    """Return a stable JSON-compatible manifest payload for reports."""
    return cast(dict[str, object], json.loads(manifest.model_dump_json()))
