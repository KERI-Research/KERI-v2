from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from metaboguard.models.feasibility_viewer import (
    discover_experiments,
    load_experiment_summary,
    render_markdown_report,
    render_text_summary,
)
from metaboguard.models.feasibility_viewer_cli import main as viewer_cli_main


def _build_experiment(root: Path, *, evaluable: bool = True) -> Path:
    experiment_dir = root / "run-001" / "pancreatic_cancer" / "3y" / "exp-1"
    baseline_dir = experiment_dir / "baseline"
    representation_dir = experiment_dir / "representation_head"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    representation_dir.mkdir(parents=True, exist_ok=True)

    status = "evaluable" if evaluable else "not_evaluable"
    partition_report = {
        "partition": "validation",
        "status": status,
        "row_count": 100,
        "metrics": {
            "auroc_synthetic_technical_feasibility_only": 0.6,
            "average_precision_synthetic_technical_feasibility_only": 0.3,
            "brier_score_synthetic_technical_feasibility_only": 0.2,
        },
    }
    for model_id, model_dir in (
        ("baseline", baseline_dir),
        ("representation_head", representation_dir),
    ):
        (model_dir / "evaluation_report.json").write_text(
            json.dumps(
                {
                    "model_identifier": model_id,
                    "partition_reports": [partition_report],
                }
            ),
            encoding="utf-8",
        )

    (experiment_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-1",
                "source_run_id": "run-001",
                "cohort_class": "ordinary_incidence",
                "endpoint_id": "pancreatic_cancer",
                "horizon_years": 3,
                "final_status": "completed",
                "source_provenance_discrepancy": False,
                "partition_label_counts": {},
            }
        ),
        encoding="utf-8",
    )
    (experiment_dir / "leakage_audit.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    (experiment_dir / "split_audit.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    (experiment_dir / "shortcut_checks.json").write_text(
        json.dumps({"label_permutation_sanity": {"suspicious_similarity_flag": False}}),
        encoding="utf-8",
    )
    return experiment_dir


def test_discover_experiments_finds_completed_dirs(tmp_path: Path) -> None:
    _build_experiment(tmp_path)
    found = discover_experiments(tmp_path)
    assert len(found) == 1
    assert (found[0] / "experiment_manifest.json").is_file()


def test_discover_experiments_empty_root_returns_empty(tmp_path: Path) -> None:
    assert discover_experiments(tmp_path / "missing") == []


def test_load_experiment_summary_reads_all_artifacts(tmp_path: Path) -> None:
    experiment_dir = _build_experiment(tmp_path)
    summary = load_experiment_summary(experiment_dir)
    assert summary["experiment_id"] == "exp-1"
    assert summary["endpoint_id"] == "pancreatic_cancer"
    assert "baseline" in summary["models"]
    assert "representation_head" in summary["models"]
    assert summary["leakage_audit"]["passed"] is True
    assert summary["split_audit"]["passed"] is True


def test_load_experiment_summary_rejects_non_experiment_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Not a completed experiment directory"):
        load_experiment_summary(tmp_path)


def test_render_text_summary_includes_metrics_and_restriction(tmp_path: Path) -> None:
    summary = load_experiment_summary(_build_experiment(tmp_path))
    text = render_text_summary(summary)
    assert "auroc=0.600" in text
    assert "Synthetic technical-feasibility artifacts only" in text
    assert "Leakage audit passed: True" in text


def test_render_text_summary_marks_not_evaluable_partitions(tmp_path: Path) -> None:
    summary = load_experiment_summary(_build_experiment(tmp_path, evaluable=False))
    text = render_text_summary(summary)
    assert "not_evaluable" in text


def test_render_markdown_report_produces_table(tmp_path: Path) -> None:
    summary = load_experiment_summary(_build_experiment(tmp_path))
    markdown = render_markdown_report([summary])
    assert "| Model | Partition | Rows | AUROC | AUPRC | Brier |" in markdown
    assert "pancreatic_cancer" in markdown


def test_cli_prints_text_summary_for_single_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    experiment_dir = _build_experiment(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["metaboguard-feasibility-report", str(experiment_dir)]
    )
    assert viewer_cli_main() == 0
    captured = capsys.readouterr()
    assert "Experiment: exp-1" in captured.out


def test_cli_writes_markdown_report_when_output_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_experiment(tmp_path)
    output_path = tmp_path / "report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "metaboguard-feasibility-report",
            str(tmp_path),
            "--output",
            str(output_path),
        ],
    )
    assert viewer_cli_main() == 0
    assert output_path.is_file()
    assert "Step 8 Synthetic Feasibility Viewer Report" in output_path.read_text(
        encoding="utf-8"
    )


def test_cli_filters_by_endpoint_and_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_experiment(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "metaboguard-feasibility-report",
            str(tmp_path),
            "--endpoint",
            "type2_diabetes",
        ],
    )
    with pytest.raises(ValueError, match="No experiments matched"):
        viewer_cli_main()


def test_cli_raises_on_empty_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["metaboguard-feasibility-report", str(tmp_path)])
    with pytest.raises(ValueError, match="No completed experiments found"):
        viewer_cli_main()
