"""CLI for Step 8 synthetic model feasibility experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metaboguard.models.model_feasibility import (
    DATASET_ALIASES,
    SyntheticFeasibilityAuthorization,
    run_model_feasibility,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", type=Path)
    parser.add_argument("endpoint_id")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument(
        "--synthetic-feasibility",
        action="store_true",
        help="Required authorization flag for simulation-only feasibility experiments.",
    )
    parser.add_argument("--approval-reference", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_ALIASES),
        default=None,
        help=(
            "Explicit cohort class to run: 'ordinary' for ordinary_incidence or "
            "'enriched' for enriched_incidence. Must match the source run's cohort "
            "class; omit to auto-detect from the source run manifest."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/model_feasibility"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write artifacts. Omit to preview immutable experiment plan only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit overwrite when output exists and cannot be resumed by hash match.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    authorization = SyntheticFeasibilityAuthorization(
        synthetic_feasibility=args.synthetic_feasibility,
        approval_reference=args.approval_reference,
    )
    result = run_model_feasibility(
        args.run_path,
        args.endpoint_id,
        args.horizon,
        authorization,
        seed=args.seed,
        artifact_root=args.artifact_root,
        execute=args.execute,
        overwrite=args.overwrite,
        dataset=args.dataset,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
