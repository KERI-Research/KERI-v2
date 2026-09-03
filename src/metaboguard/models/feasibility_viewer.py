"""Read-only viewer for Step 8 synthetic model-feasibility experiment artifacts.

This module never fits, scores, or mutates a model. It only loads and renders
the frozen artifacts already written by ``metaboguard-model-feasibility`` so
that experiment behaviour (metrics, leakage/shortcut checks, label counts) can
be inspected without re-running the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

MANDATORY_VIEWER_RESTRICTION = (
    "Synthetic technical-feasibility artifacts only. Nothing rendered here is a "
    "clinical performance, calibration, or utility claim, and none of it may be "
    "used for diagnosis, screening, patient-level risk, or care decisions."
)

MODEL_DIRECTORIES = ("baseline", "representation_head")
PARTITIONS = ("validation", "test", "temporal_holdout")


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def discover_experiments(artifact_root: Path) -> list[Path]:
    """Return every completed experiment directory under ``artifact_root``.

    A directory is a completed experiment if it contains an
    ``experiment_manifest.json``, sorted for deterministic viewer output.
    """
    if not artifact_root.is_dir():
        return []
    return sorted(
        path.parent
        for path in artifact_root.rglob("experiment_manifest.json")
        if path.is_file()
    )


def load_experiment_summary(experiment_dir: Path) -> dict[str, Any]:
    """Load a single experiment's manifest, model reports, and audit artifacts."""
    manifest_path = experiment_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Not a completed experiment directory: {experiment_dir}")
    manifest = _load_json(manifest_path)

    models: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_DIRECTORIES:
        report_path = experiment_dir / model_id / "evaluation_report.json"
        if report_path.is_file():
            models[model_id] = _load_json(report_path)

    def _optional(name: str) -> dict[str, Any] | None:
        path = experiment_dir / name
        return _load_json(path) if path.is_file() else None

    return {
        "experiment_dir": str(experiment_dir),
        "experiment_id": manifest.get("experiment_id"),
        "source_run_id": manifest.get("source_run_id"),
        "cohort_class": manifest.get("cohort_class"),
        "endpoint_id": manifest.get("endpoint_id"),
        "horizon_years": manifest.get("horizon_years"),
        "final_status": manifest.get("final_status"),
        "source_provenance_discrepancy": manifest.get("source_provenance_discrepancy"),
        "partition_label_counts": manifest.get("partition_label_counts", {}),
        "models": models,
        "leakage_audit": _optional("leakage_audit.json"),
        "shortcut_checks": _optional("shortcut_checks.json"),
        "split_audit": _optional("split_audit.json"),
    }


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_text_summary(summary: dict[str, Any]) -> str:
    """Render one experiment as a concise, human-readable text report."""
    lines: list[str] = [
        f"Experiment: {summary['experiment_id']}",
        f"Source run: {summary['source_run_id']} ({summary['cohort_class']})",
        f"Endpoint: {summary['endpoint_id']}  Horizon: {summary['horizon_years']}y",
        f"Status: {summary['final_status']}  "
        f"Provenance discrepancy: {summary['source_provenance_discrepancy']}",
        "",
    ]
    for model_id in MODEL_DIRECTORIES:
        report = summary["models"].get(model_id)
        if report is None:
            continue
        lines.append(f"Model: {model_id}")
        for partition_report in report.get("partition_reports", []):
            partition = partition_report.get("partition")
            status = partition_report.get("status")
            if status != "evaluable":
                lines.append(f"  {partition}: not_evaluable ({status})")
                continue
            metrics = partition_report.get("metrics", {})
            lines.append(
                f"  {partition}: rows={partition_report.get('row_count')} "
                f"auroc={_format_metric(metrics.get('auroc_synthetic_technical_feasibility_only'))} "
                f"auprc={_format_metric(metrics.get('average_precision_synthetic_technical_feasibility_only'))} "
                f"brier={_format_metric(metrics.get('brier_score_synthetic_technical_feasibility_only'))}"
            )
        lines.append("")

    leakage = summary.get("leakage_audit")
    if leakage is not None:
        lines.append(f"Leakage audit passed: {leakage.get('passed')}")
    split_audit = summary.get("split_audit")
    if split_audit is not None:
        lines.append(f"Patient isolation passed: {split_audit.get('passed')}")
    shortcuts = summary.get("shortcut_checks")
    if shortcuts is not None:
        suspicious = shortcuts.get("label_permutation_sanity", {}).get(
            "suspicious_similarity_flag"
        )
        lines.append(f"Label-permutation suspicious similarity flag: {suspicious}")
    lines.append("")
    lines.append(f"Restriction: {MANDATORY_VIEWER_RESTRICTION}")
    return "\n".join(lines)


def render_markdown_report(summaries: list[dict[str, Any]]) -> str:
    """Render a multi-experiment Markdown viewer report."""
    lines = [
        "# Step 8 Synthetic Feasibility Viewer Report",
        "",
        f"> {MANDATORY_VIEWER_RESTRICTION}",
        "",
        f"Experiments rendered: {len(summaries)}",
        "",
    ]
    for summary in summaries:
        lines.append(
            f"## {summary['endpoint_id']} / {summary['horizon_years']}y "
            f"/ {summary['cohort_class']}"
        )
        lines.append("")
        lines.append(f"- Experiment ID: `{summary['experiment_id']}`")
        lines.append(f"- Source run: `{summary['source_run_id']}`")
        lines.append(f"- Status: {summary['final_status']}")
        lines.append(
            f"- Provenance discrepancy: {summary['source_provenance_discrepancy']}"
        )
        lines.append("")
        lines.append("| Model | Partition | Rows | AUROC | AUPRC | Brier |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for model_id in MODEL_DIRECTORIES:
            report = summary["models"].get(model_id)
            if report is None:
                continue
            for partition_report in report.get("partition_reports", []):
                if partition_report.get("status") != "evaluable":
                    lines.append(
                        f"| {model_id} | {partition_report.get('partition')} "
                        f"| {partition_report.get('row_count')} | not_evaluable "
                        "| not_evaluable | not_evaluable |"
                    )
                    continue
                metrics = partition_report.get("metrics", {})
                lines.append(
                    f"| {model_id} | {partition_report.get('partition')} "
                    f"| {partition_report.get('row_count')} "
                    f"| {_format_metric(metrics.get('auroc_synthetic_technical_feasibility_only'))} "
                    f"| {_format_metric(metrics.get('average_precision_synthetic_technical_feasibility_only'))} "
                    f"| {_format_metric(metrics.get('brier_score_synthetic_technical_feasibility_only'))} |"
                )
        lines.append("")
    return "\n".join(lines)
