"""Simulation cohort manifests and class-separation guards."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from metaboguard.data.canonical import CanonicalDataset

CohortClass = Literal[
    "ordinary_incidence",
    "enriched_incidence",
    "enriched_pancreatic",
    "enriched_multicancer",
]


class CohortClassMismatchError(ValueError):
    """Raised when simulation cohorts with different sampling classes are combined."""


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config: object) -> str:
    """Hash a JSON-serializable configuration deterministically."""
    payload = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_batch_seed(root_seed: int, batch_index: int) -> int:
    """Derive a stable, non-overlapping integer seed for a batch."""
    payload = f"{root_seed}:{batch_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def assert_same_cohort_class(classes: list[str] | tuple[str, ...]) -> CohortClass:
    """Require all operations to use exactly one cohort sampling class."""
    if not classes:
        raise CohortClassMismatchError("At least one cohort class is required")
    unique = set(classes)
    if len(unique) != 1:
        raise CohortClassMismatchError(
            "Ordinary-incidence and enriched cohorts must not be combined: "
            + ", ".join(sorted(unique))
        )
    cohort_class = next(iter(unique))
    if cohort_class not in {
        "ordinary_incidence",
        "enriched_incidence",
        "enriched_pancreatic",
        "enriched_multicancer",
    }:
        raise CohortClassMismatchError(f"Unknown cohort class: {cohort_class}")
    return cohort_class  # type: ignore[return-value]


def assert_dataset_class(dataset: CanonicalDataset, cohort_class: str) -> None:
    """Require a dataset's attached class metadata to match an operation."""
    dataset_class = getattr(dataset, "cohort_class", None)
    if dataset_class is not None and dataset_class != cohort_class:
        raise CohortClassMismatchError(
            f"Expected {cohort_class}, received {dataset_class}"
        )


@dataclass(slots=True)
class GenerationManifest:
    """Manifest for one reproducible simulation cohort run."""

    cohort_class: CohortClass
    run_id: str
    root_seed: int
    batch_seeds: list[int]
    config_hash: str
    git_sha: str
    synthea_version: str
    jar_sha256: str
    java_version: str
    operating_system: str
    python_version: str
    geography: str
    age_range: tuple[int, int]
    population_settings: dict[str, object]
    sampling_stratum: str
    inverse_probability_weight_policy: str
    generation_commands: list[list[str]]
    generated_patient_count: int = 0
    converted_patient_count: int = 0
    excluded_patient_count: int = 0
    deduplicated_patient_count: int = 0
    event_counts: dict[str, object] = field(default_factory=dict)
    raw_dataset_sha256: str = ""
    canonical_dataset_sha256: str = ""
    date_normalisation_audit_sha256: str = ""
    unit_normalisation_audit_sha256: str = ""
    augmentation_modules: list[dict[str, str]] = field(default_factory=list)
    split_status: str = "not_created"
    simulation_only: bool = True
    state: str = "in_progress"
    cohort_status: str = "in_progress"
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write(self, path: Path) -> None:
        """Write stable JSON for hashing and resumability."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def git_sha() -> str:
    """Return the current Git SHA, or an explicit unavailable marker."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def new_run_id(cohort_class: CohortClass, root_seed: int) -> str:
    """Create a stable-readable run ID from class, seed, and UTC timestamp."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{cohort_class}-{timestamp}-{root_seed}-{uuid.uuid4().hex[:8]}"


def runtime_metadata(java_version: str = "unavailable") -> dict[str, str]:
    """Return runtime metadata recorded in manifests."""
    return {
        "java_version": java_version,
        "operating_system": sys.platform,
        "python_version": sys.version.split()[0],
    }
