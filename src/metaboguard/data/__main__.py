"""Command-line entry point for a small or production Synthea generation run."""

from __future__ import annotations

import argparse
from pathlib import Path

from metaboguard.config import load_config
from metaboguard.data.synthea_runner import (
    SyntheaGenerationConfig,
    generate_synthea_cohort,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a simulation-only Synthea cohort."
    )
    parser.add_argument(
        "--cohort-class",
        required=True,
        choices=(
            "ordinary_incidence",
            "enriched_incidence",
            "enriched_pancreatic",
            "enriched_multicancer",
        ),
    )
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--patients-per-batch", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", choices=("true", "false"), default="true")
    parser.add_argument("--jar-path", type=Path)
    parser.add_argument("--jar-sha256")
    return parser


def main() -> int:
    """Parse CLI arguments and run the configured Synthea cohort."""
    args = _parser().parse_args()
    if args.dry_run == "true":
        print("dry_run=true; no Synthea process started")
        return 0
    if args.batches < 1 or args.patients_per_batch < 1:
        raise SystemExit("--batches and --patients-per-batch must be positive")
    config = load_config()["synthea"]
    jar_path = args.jar_path or Path(config["jar_path"])
    jar_sha256 = args.jar_sha256 or str(config["jar_sha256"])
    generation = SyntheaGenerationConfig(
        cohort_class=args.cohort_class,
        root_seed=args.seed,
        target_patients=args.batches * args.patients_per_batch,
        batch_size=args.patients_per_batch,
        jar_path=jar_path,
        jar_sha256=jar_sha256,
        synthea_version=str(config["synthea_version"]),
        geography=str(config["geography"]),
        min_age=int(config["min_age"]),
        max_age=int(config["max_age"]),
        output_root=args.output_root,
        java_executable=str(config["java_executable"]),
        enabled_cancer_sites=tuple(
            str(site) for site in config["enabled_cancer_sites"]
        ),
    )
    manifest = generate_synthea_cohort(generation)
    print(manifest.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
