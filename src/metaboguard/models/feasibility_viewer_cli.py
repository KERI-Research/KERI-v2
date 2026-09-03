"""CLI to view Step 8 synthetic model-feasibility experiment behaviour.

Read-only: loads frozen experiment artifacts already written by
``metaboguard-model-feasibility`` and renders them for inspection. Never fits,
scores, or mutates a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metaboguard.models.feasibility_viewer import (
    discover_experiments,
    load_experiment_summary,
    render_markdown_report,
    render_text_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "A single experiment directory (containing experiment_manifest.json) "
            "or an artifact root to scan for all completed experiments."
        ),
    )
    parser.add_argument(
        "--endpoint",
        help="Only include experiments for this endpoint_id.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        help="Only include experiments for this horizon (years).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write a Markdown viewer report to this path instead of printing text.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (args.path / "experiment_manifest.json").is_file():
        experiment_dirs = [args.path]
    else:
        experiment_dirs = discover_experiments(args.path)
    if not experiment_dirs:
        raise ValueError(f"No completed experiments found under: {args.path}")

    summaries = [load_experiment_summary(directory) for directory in experiment_dirs]
    if args.endpoint is not None:
        summaries = [s for s in summaries if s["endpoint_id"] == args.endpoint]
    if args.horizon is not None:
        summaries = [s for s in summaries if s["horizon_years"] == args.horizon]
    if not summaries:
        raise ValueError("No experiments matched the requested --endpoint/--horizon")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_markdown_report(summaries), encoding="utf-8")
        print(f"Wrote viewer report: {args.output}")
        return 0

    for summary in summaries:
        print(render_text_summary(summary))
        print("-" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
