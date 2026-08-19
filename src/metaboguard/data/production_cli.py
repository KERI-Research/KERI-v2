"""Command-line boundary for one Step 7 production cohort class."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from metaboguard.data.production_generation import (
    ProductionCohortPlan,
    generate_production_run,
    load_production_config,
    reconcile_production_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one simulation-only production cohort and frozen Steps 4-6 artifacts."
    )
    parser.add_argument(
        "cohort_class",
        choices=("ordinary_incidence", "enriched_incidence"),
        help="Configured cohort class to run; classes are never pooled.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/synthetic_longitudinal/production"),
        help="Production run root.",
    )
    parser.add_argument(
        "--patients",
        choices=("500", "5000", "25000", "all"),
        default="500",
        help="Requested Synthea population. 'all' uses the configured production target.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the configured root seed; changes the deterministic run identity.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the configured batch population size; must be positive.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Start the configured run. Omit to print the selected immutable plan only.",
    )
    parser.add_argument(
        "--reconcile",
        type=Path,
        help="Reconcile an existing run directory and rewrite its top-level manifest.",
    )
    return parser


def _selected_plan(
    configured_plan: ProductionCohortPlan,
    patients: str,
    seed: int | None,
    batch_size: int | None,
) -> ProductionCohortPlan:
    """Apply explicit CLI overrides to one configured cohort class."""
    if batch_size is not None and batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    return replace(
        configured_plan,
        population_target=(
            configured_plan.population_target if patients == "all" else int(patients)
        ),
        root_seed=configured_plan.root_seed if seed is None else seed,
        batch_population_size=batch_size,
    )


def main() -> int:
    """Run exactly one configured production cohort class."""
    args = _parser().parse_args()
    if args.reconcile is not None:
        manifest = reconcile_production_manifest(args.reconcile)
        print(manifest.model_dump_json(indent=2))
        return 0
    config = load_production_config()
    configured_plan = next(
        (item for item in config.plans if item.cohort_class == args.cohort_class), None
    )
    if configured_plan is None:
        raise SystemExit(f"No enabled production plan for {args.cohort_class}")
    plan = _selected_plan(configured_plan, args.patients, args.seed, args.batch_size)
    if not args.execute:
        print(
            json.dumps(
                {
                    "cohort_class": plan.cohort_class,
                    "population_target": plan.population_target,
                    "root_seed": plan.root_seed,
                    "augmentation_profile": plan.augmentation_profile,
                    "batch_population_size": plan.batch_population_size,
                    "simulation_only": True,
                    "pipeline_rehearsal_only": True,
                    "model_status": "not_created",
                    "execute": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = generate_production_run(plan, args.output_root)
    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "cohort_class": manifest.cohort_class,
                "status": manifest.status,
                "population_target": manifest.population_target,
                "generated_patient_count": manifest.generated_patient_count,
                "canonical_status": manifest.canonical_status,
                "cohort_status": manifest.cohort_status,
                "split_status": manifest.split_status,
                "feature_status": manifest.feature_status,
                "readiness_status": manifest.readiness_status,
                "simulation_only": manifest.simulation_only,
                "model_status": manifest.model_status,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
