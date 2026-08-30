"""Synthetic-only Step 8 model feasibility pipeline.

This module is a bounded research workflow for synthetic technical-feasibility
experiments. It must not be used for clinical inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, RobustScaler  # type: ignore[import-untyped]

from metaboguard.cohort.protocol import load_endpoint_registry
from metaboguard.config import load_config
from metaboguard.data.manifests import assert_same_cohort_class, config_sha256
from metaboguard.features.extraction import IDENTIFIER_COLUMNS
from metaboguard.models.ssl_encoder import (
    RobustPCAEncoder,
    RobustPCAEncoderConfig,
    SyntheticPrototypeAuthorization,
)

MANDATORY_MODEL_CARD_RESTRICTION = (
    "Synthetic technical-feasibility prototype only. This artifact was trained and "
    "evaluated exclusively on Synthea-derived synthetic data. It is not a diagnostic "
    "device and must not be used for clinical decision-making, patient screening, "
    "risk assessment, medical advice, treatment decisions, triage, or patient care. "
    "Reported results demonstrate software and research-workflow feasibility only; "
    "they do not establish predictive performance, calibration, clinical validity, "
    "clinical utility, prevalence, or generalisability. External validation on an "
    "approved real longitudinal cohort is required before any clinical interpretation."
)

SYNTHETIC_FLAGS = {
    "simulation_only": True,
    "pipeline_rehearsal_only": True,
    "clinical_use_prohibited": True,
    "research_feasibility_only": True,
}

SUPPORTED_HORIZONS = {1, 3, 5}
SUPPORTED_PRIMARY_COHORT = "ordinary_incidence"
EVAL_PARTITIONS = ("validation", "test", "temporal_holdout")
FIT_LABEL_STATES = {"positive", "eligible_negative"}
DENYLIST_TOKENS = (
    "label",
    "outcome",
    "censor",
    "death",
    "diagnosis",
    "treatment",
    "prognosis",
    "survival",
    "tumour",
    "histology",
)


@dataclass(frozen=True, slots=True)
class SyntheticFeasibilityAuthorization:
    """Explicit professor-approved authorization for synthetic feasibility only."""

    synthetic_feasibility: bool
    approval_reference: str

    def __post_init__(self) -> None:
        if not self.synthetic_feasibility:
            raise ValueError("--synthetic-feasibility is required")
        if not self.approval_reference.strip():
            raise ValueError("--approval-reference must be non-empty")


@dataclass(frozen=True, slots=True)
class FeasibilityPaths:
    run_manifest: Path
    generation_manifest: Path
    cohort_manifest: Path
    split_manifest: Path
    split_assignments: Path
    endpoint_protocol: Path
    feature_manifest: Path
    feature_matrix: Path
    feature_lineage: Path
    feature_registry: Path
    labels: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as tmp:
        pickle.dump(payload, tmp)
        temporary_path = Path(tmp.name)
    os.replace(temporary_path, path)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".parquet", dir=path.parent, delete=False
    ) as tmp:
        temp_path = Path(tmp.name)
    frame.to_parquet(temp_path, index=False, engine="pyarrow", compression="zstd")
    os.replace(temp_path, path)


def _path_for(run_path: Path, endpoint_id: str, horizon: int) -> FeasibilityPaths:
    endpoint_dir = run_path / "cohort" / endpoint_id
    feature_dir = run_path / "features" / endpoint_id
    return FeasibilityPaths(
        run_manifest=run_path / "manifest.json",
        generation_manifest=run_path / "generation_manifest.json",
        cohort_manifest=endpoint_dir / "cohort_manifest.json",
        split_manifest=endpoint_dir / "splits" / "split_manifest.json",
        split_assignments=endpoint_dir / "splits" / "split_assignments.parquet",
        endpoint_protocol=endpoint_dir / "endpoint_protocol.json",
        feature_manifest=feature_dir / "feature_manifest.json",
        feature_matrix=feature_dir / "feature_matrix.parquet",
        feature_lineage=feature_dir / "feature_lineage.parquet",
        feature_registry=feature_dir / "feature_definition_registry.json",
        labels=endpoint_dir / f"horizon_labels_{horizon}y.parquet",
    )


def _assert_required_files(paths: FeasibilityPaths) -> None:
    for path in (
        paths.run_manifest,
        paths.generation_manifest,
        paths.cohort_manifest,
        paths.split_manifest,
        paths.split_assignments,
        paths.endpoint_protocol,
        paths.feature_manifest,
        paths.feature_matrix,
        paths.feature_lineage,
        paths.feature_registry,
        paths.labels,
    ):
        if not path.is_file():
            raise ValueError(f"Missing required feasibility input artifact: {path}")


def _validate_endpoint_horizon(endpoint_id: str, horizon: int) -> None:
    registry = load_endpoint_registry()
    if endpoint_id not in registry:
        raise ValueError(f"Unsupported endpoint: {endpoint_id}")
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    if horizon not in set(registry[endpoint_id].horizon_years):
        raise ValueError(
            f"Horizon {horizon} is not configured for endpoint {endpoint_id}"
        )


def _durable_synthea(generation_manifest: dict[str, Any]) -> bool:
    return (
        generation_manifest.get("state") == "complete"
        and bool(generation_manifest.get("synthea_version"))
        and bool(generation_manifest.get("jar_sha256"))
        and bool(generation_manifest.get("canonical_dataset_sha256"))
    )


def _provenance_discrepancies(
    artifacts: dict[str, dict[str, Any]],
) -> list[dict[str, object]]:
    discrepancies: list[dict[str, object]] = []
    for name, payload in artifacts.items():
        if payload.get("simulation_only") is not True:
            discrepancies.append(
                {
                    "artifact": name,
                    "observed_field": "simulation_only",
                    "observed_value": payload.get("simulation_only"),
                }
            )
    return discrepancies


def _feature_policy_audit(
    matrix: pd.DataFrame,
    registry_rows: list[dict[str, object]],
    lineage: pd.DataFrame,
) -> dict[str, Any]:
    registry_ids = {str(item["feature_id"]) for item in registry_rows}
    identifiers = set(IDENTIFIER_COLUMNS)
    predictor_columns = [
        str(column) for column in matrix.columns if str(column) not in identifiers
    ]
    unknown = [column for column in predictor_columns if column not in registry_ids]
    denied = [
        column
        for column in predictor_columns
        if any(token in column.lower() for token in DENYLIST_TOKENS)
    ]
    post_index_flags = (
        int(lineage["contains_post_index_record"].fillna(False).astype(bool).sum())
        if "contains_post_index_record" in lineage
        else 0
    )
    future_dates = 0
    if "latest_source_date" in lineage and "index_date" in lineage:
        latest = pd.to_datetime(lineage["latest_source_date"], errors="coerce")
        index = pd.to_datetime(lineage["index_date"], errors="coerce")
        future_dates = int((latest > index).sum())
    checks = {
        "registry_columns_only": len(unknown) == 0,
        "denylist_clean": len(denied) == 0,
        "no_post_index_flagged": post_index_flags == 0,
        "no_future_dates": future_dates == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "unknown_columns": unknown,
        "denylisted_columns": denied,
        "post_index_flag_count": post_index_flags,
        "future_date_count": future_dates,
    }


def _row_key(
    patient_id: str, index_date: object, endpoint_id: str, horizon: int
) -> str:
    return _digest(f"{patient_id}|{index_date}|{endpoint_id}|{horizon}")


def _split_map(split_manifest: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    assignments = {
        str(patient_id): str(split)
        for patient_id, split in cast(
            dict[str, str], split_manifest["assignments"]
        ).items()
    }
    fingerprints = {
        str(patient_id): str(fingerprint)
        for patient_id, fingerprint in cast(
            dict[str, str], split_manifest["patient_fingerprints"]
        ).items()
    }
    return assignments, fingerprints


def _assert_patient_isolation(matrix: pd.DataFrame) -> dict[str, Any]:
    split_counts = matrix.groupby("patient_id")["split"].nunique(dropna=True)
    violating = split_counts[split_counts > 1]
    return {
        "passed": len(violating) == 0,
        "violating_patient_count": len(violating),
    }


def _assert_identity(
    matrix: pd.DataFrame,
    labels: pd.DataFrame,
    run_manifest: dict[str, Any],
    endpoint_id: str,
    horizon: int,
) -> None:
    cohort_values = {str(value) for value in matrix["cohort_class"].dropna().unique()}
    endpoint_values = {str(value) for value in matrix["endpoint_id"].dropna().unique()}
    if len(cohort_values) != 1:
        raise ValueError("Feature matrix must contain exactly one cohort class")
    cohort_class = assert_same_cohort_class(list(cohort_values))
    if cohort_class != str(run_manifest.get("cohort_class", "")):
        raise ValueError("Feature matrix cohort class does not match source run")
    if cohort_class != SUPPORTED_PRIMARY_COHORT:
        raise ValueError("Step 8 primary feasibility supports ordinary_incidence only")
    if endpoint_values != {endpoint_id}:
        raise ValueError("Feature matrix endpoint mismatch")
    label_endpoints = {str(value) for value in labels["endpoint_id"].dropna().unique()}
    if label_endpoints != {endpoint_id}:
        raise ValueError("Label artifact endpoint mismatch")
    label_horizons = {int(value) for value in labels["horizon_years"].dropna().unique()}
    if label_horizons != {horizon}:
        raise ValueError("Label artifact horizon mismatch")


def _merge_features_labels(
    matrix: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ["patient_id", "index_date", "endpoint_id", "cohort_class"]
    left = matrix.copy()
    right = labels.copy()
    left["index_date"] = pd.to_datetime(left["index_date"], errors="coerce")
    right["index_date"] = pd.to_datetime(right["index_date"], errors="coerce")
    merged = left.merge(
        right[[*key_columns, "horizon_years", "label_state"]],
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    left_keys = set(
        tuple(row) for row in left[key_columns].itertuples(index=False, name=None)
    )
    right_keys = set(
        tuple(row) for row in right[key_columns].itertuples(index=False, name=None)
    )
    if left_keys != right_keys:
        raise ValueError(
            "Feature matrix and horizon labels do not share identical keys"
        )
    if len(merged) != len(left):
        raise ValueError("Merged feature-label rows are incomplete")
    return merged


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_brier(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(y_true) == 0:
        return None
    return float(brier_score_loss(y_true, y_score))


def _predictor_columns(
    frame: pd.DataFrame, registry: list[dict[str, object]]
) -> list[str]:
    allowed = {str(item["feature_id"]) for item in registry}
    return [
        column
        for column in frame.columns
        if column in allowed
        and all(token not in column.lower() for token in DENYLIST_TOKENS)
    ]


def _prepare_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    prepared = frame[columns].copy()
    categorical_columns = [
        str(column)
        for column in prepared.columns
        if pd.api.types.is_string_dtype(prepared[column])
        or pd.api.types.is_object_dtype(prepared[column])
        or isinstance(prepared[column].dtype, pd.CategoricalDtype)
    ]
    for column in categorical_columns:
        prepared[column] = (
            prepared[column].astype("string").replace({"<NA>": pd.NA}).astype("object")
        )
    return prepared


def _fit_preprocessor(
    train_frame: pd.DataFrame,
    predictor_columns: list[str],
) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    prepared = _prepare_frame(train_frame, predictor_columns)
    all_null = [column for column in predictor_columns if prepared[column].isna().all()]
    non_null_columns = [
        column for column in predictor_columns if column not in all_null
    ]
    zero_variance: list[str] = []
    for column in non_null_columns:
        if prepared[column].dropna().nunique() <= 1:
            zero_variance.append(column)
    retained = [
        column for column in non_null_columns if column not in set(zero_variance)
    ]
    if not retained:
        raise ValueError("No retained predictor columns after train-only filtering")
    numeric = [
        column for column in retained if pd.api.types.is_numeric_dtype(prepared[column])
    ]
    categorical = [column for column in retained if column not in numeric]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", RobustScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers, sparse_threshold=0.0)
    preprocessor.fit(_prepare_frame(train_frame, retained))
    return preprocessor, retained, all_null, zero_variance


def _encode_labels(states: pd.Series) -> np.ndarray:
    return np.asarray((states == "positive").astype(int), dtype=int)


def _select_logistic_hyperparameter(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
) -> tuple[LogisticRegression, dict[str, object]]:
    candidates = [0.1, 1.0, 10.0]
    best: LogisticRegression | None = None
    best_score = -np.inf
    details: list[dict[str, object]] = []
    for c_value in candidates:
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
            max_iter=1000,
        )
        model.fit(train_x, train_y)
        probs = model.predict_proba(validation_x)[:, 1]
        ap = _safe_ap(validation_y, probs)
        auc = _safe_auc(validation_y, probs)
        primary = ap if ap is not None else (auc if auc is not None else -np.inf)
        details.append({"C": c_value, "ap": ap, "auroc": auc, "score": primary})
        if primary > best_score:
            best_score = primary
            best = model
    if best is None:
        raise ValueError("Unable to select a logistic-regression hyperparameter")
    selected_c = float(best.get_params()["C"])
    return best, {
        "grid": [{"C": value} for value in candidates],
        "selection_metric": "validation_ap_then_auroc_synthetic_technical_feasibility_only",
        "selected": {"C": selected_c},
        "candidates": details,
    }


def _partition_report(
    partition: str,
    rows: pd.DataFrame,
    min_events: int,
) -> dict[str, Any]:
    states = rows["label_state"].astype("string")
    counts = {
        "positive": int((states == "positive").sum()),
        "eligible_negative": int((states == "eligible_negative").sum()),
        "censored": int((states == "censored").sum()),
        "competing_death": int((states == "competing_death").sum()),
        "excluded": int((states == "excluded").sum()),
    }
    fit_rows = rows[states.isin(FIT_LABEL_STATES)].copy()
    missing_scores = (
        int(fit_rows["prediction"].isna().sum()) if "prediction" in fit_rows else 0
    )
    finite_scores = (
        int(np.isfinite(fit_rows["prediction"].astype(float)).sum())
        if "prediction" in fit_rows and len(fit_rows)
        else 0
    )
    report: dict[str, Any] = {
        "partition": partition,
        "status": "evaluable",
        "row_count": len(rows),
        "label_state_counts": counts,
        "missing_score_count": missing_scores,
        "finite_score_count": finite_scores,
        "metrics": {},
        "metric_label": "synthetic technical feasibility only",
    }
    if len(fit_rows) == 0:
        report["status"] = "not_evaluable"
        report["reason"] = "no_supervised_rows_after_label_state_filter"
        return report
    y_true = _encode_labels(fit_rows["label_state"])
    y_score = np.asarray(fit_rows["prediction"].astype(float), dtype=float)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        report["status"] = "not_evaluable"
        report["reason"] = "partition_lacks_both_supervised_classes"
        return report
    if positives < min_events:
        report["status"] = "not_evaluable"
        report["reason"] = "partition_below_minimum_event_count"
        return report
    report["metrics"] = {
        "auroc_synthetic_technical_feasibility_only": _safe_auc(y_true, y_score),
        "average_precision_synthetic_technical_feasibility_only": _safe_ap(
            y_true, y_score
        ),
        "brier_score_synthetic_technical_feasibility_only": _safe_brier(
            y_true, y_score
        ),
    }
    demo_threshold = 0.20
    confusion = confusion_matrix(y_true, (y_score >= demo_threshold).astype(int))
    report["confusion_matrix_synthetic_technical_feasibility_only"] = {
        "threshold": demo_threshold,
        "tn": int(confusion[0, 0]),
        "fp": int(confusion[0, 1]),
        "fn": int(confusion[1, 0]),
        "tp": int(confusion[1, 1]),
    }
    report["prediction_distribution"] = {
        "min": float(np.min(y_score)),
        "max": float(np.max(y_score)),
        "mean": float(np.mean(y_score)),
        "std": float(np.std(y_score)),
        "p10": float(np.quantile(y_score, 0.10)),
        "p50": float(np.quantile(y_score, 0.50)),
        "p90": float(np.quantile(y_score, 0.90)),
        "hash": _digest("|".join(f"{value:.10f}" for value in sorted(y_score))),
    }
    return report


def _write_predictions(
    destination: Path,
    rows: pd.DataFrame,
    endpoint_id: str,
    horizon: int,
    model_id: str,
    experiment_id: str,
) -> None:
    states = rows["label_state"].astype("string")
    filtered = rows[states.isin(FIT_LABEL_STATES)].copy()
    prediction_frame = pd.DataFrame(
        {
            "row_key": [
                _row_key(str(row.patient_id), row.index_date, endpoint_id, horizon)
                for row in filtered.itertuples(index=False)
            ],
            "partition": filtered["split"].astype("string"),
            "endpoint": endpoint_id,
            "horizon_years": horizon,
            "model_identifier": model_id,
            "prediction": filtered["prediction"].astype(float),
            "experiment_id": experiment_id,
            **SYNTHETIC_FLAGS,
        }
    )
    _atomic_write_parquet(destination, prediction_frame)


def _drift_report(
    train_rows: pd.DataFrame,
    holdout_rows: pd.DataFrame,
) -> dict[str, Any]:
    train_scores = train_rows["prediction"].dropna().astype(float)
    holdout_scores = holdout_rows["prediction"].dropna().astype(float)
    score_drift = None
    if len(train_scores) and len(holdout_scores):
        score_drift = {
            "mean_delta": float(holdout_scores.mean() - train_scores.mean()),
            "std_delta": float(holdout_scores.std() - train_scores.std()),
        }
    return {
        "warning_only": True,
        "train_row_count": len(train_rows),
        "temporal_holdout_row_count": len(holdout_rows),
        "score_distribution_drift": score_drift,
    }


def _model_card(summary: dict[str, Any]) -> str:
    return (
        "# Step 8 Synthetic Model Feasibility Card\n\n"
        "## Mandatory Restriction\n\n"
        f"> {MANDATORY_MODEL_CARD_RESTRICTION}\n\n"
        "## Purpose and Intended Use\n\n"
        "This experiment demonstrates software workflow feasibility for train-only "
        "preprocessing, synthetic supervised baselines, and a representation-head "
        "prototype under immutable synthetic artifacts.\n\n"
        "## Prohibited Uses\n\n"
        "Clinical use is prohibited. No diagnosis, screening, patient-level risk, "
        "or care decision usage is permitted.\n\n"
        "## Data Provenance\n\n"
        f"- Source run ID: {summary['source_run_id']}\n"
        f"- Cohort class: {summary['cohort_class']}\n"
        f"- Endpoint: {summary['endpoint_id']}\n"
        f"- Horizon years: {summary['horizon_years']}\n"
        f"- Source provenance discrepancy: {summary['source_provenance_discrepancy']}\n\n"
        "## Partition and Fitting Rules\n\n"
        "- Train-only preprocessing and fitting.\n"
        "- Validation-only hyperparameter selection.\n"
        "- Test and temporal holdout are evaluation-only.\n\n"
        "## Evaluation Limits\n\n"
        "All metrics are synthetic technical feasibility only and not clinical "
        "performance claims.\n\n"
        "## External Validation Requirement\n\n"
        "Approved real longitudinal external validation is required before any "
        "clinical interpretation.\n"
    )


def _evaluate_model(
    merged: pd.DataFrame,
    predictor_columns: list[str],
    seed: int,
    model_id: str,
) -> tuple[
    dict[str, Any],
    ColumnTransformer,
    LogisticRegression,
    pd.DataFrame,
    list[str],
    list[str],
]:
    train_rows = merged[merged["split"] == "train"].copy()
    validation_rows = merged[merged["split"] == "validation"].copy()
    if train_rows.empty or validation_rows.empty:
        raise ValueError("Train and validation partitions are required")
    train_fit = train_rows[train_rows["label_state"].isin(FIT_LABEL_STATES)].copy()
    validation_fit = validation_rows[
        validation_rows["label_state"].isin(FIT_LABEL_STATES)
    ].copy()
    if train_fit.empty or validation_fit.empty:
        raise ValueError("Train and validation need positive/eligible_negative rows")
    preprocessor, retained, all_null, zero_variance = _fit_preprocessor(
        train_fit, predictor_columns
    )
    train_x = np.asarray(
        preprocessor.transform(_prepare_frame(train_fit, retained)), dtype=float
    )
    validation_x = np.asarray(
        preprocessor.transform(_prepare_frame(validation_fit, retained)), dtype=float
    )
    train_y = _encode_labels(train_fit["label_state"])
    validation_y = _encode_labels(validation_fit["label_state"])
    model, selection = _select_logistic_hyperparameter(
        train_x, train_y, validation_x, validation_y, seed
    )
    scored = merged.copy()
    scored["prediction"] = np.nan
    for partition in ("train", *EVAL_PARTITIONS):
        part = scored[scored["split"] == partition]
        if part.empty:
            continue
        transformed = np.asarray(
            preprocessor.transform(_prepare_frame(part, retained)), dtype=float
        )
        probs = model.predict_proba(transformed)[:, 1]
        scored.loc[part.index, "prediction"] = probs
    minimum_partition_events = int(
        load_config()["readiness"]["minimum_events_per_evaluation_partition"]
    )
    report = {
        "model_identifier": model_id,
        "class_weight_strategy": "balanced_from_training_labels_only",
        "preprocessing": {
            "retained_feature_order": retained,
            "all_null_training_columns": all_null,
            "zero_variance_training_columns": zero_variance,
            "train_only_fit": True,
        },
        "hyperparameter_selection": selection,
        "partition_reports": [
            _partition_report(
                partition,
                scored[scored["split"] == partition],
                minimum_partition_events,
            )
            for partition in EVAL_PARTITIONS
        ],
        "train_to_holdout_drift_warning": _drift_report(
            scored[scored["split"] == "train"],
            scored[scored["split"] == "temporal_holdout"],
        ),
    }
    return report, preprocessor, model, scored, all_null, zero_variance


def _representation_head(
    merged: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, Any], RobustPCAEncoder, LogisticRegression, pd.DataFrame]:
    train_rows = merged[merged["split"] == "train"].copy()
    encoder_train_rows = train_rows.drop(
        columns=["label_state", "horizon_years"], errors="ignore"
    )
    train_fit = train_rows[train_rows["label_state"].isin(FIT_LABEL_STATES)].copy()
    validation_fit = merged[
        (merged["split"] == "validation") & merged["label_state"].isin(FIT_LABEL_STATES)
    ].copy()
    latent = max(1, min(8, len(train_fit) - 1 if len(train_fit) > 1 else 1))
    encoder = RobustPCAEncoder(RobustPCAEncoderConfig(latent_dimensions=latent))
    encoder.fit_synthetic_prototype(
        encoder_train_rows,
        SyntheticPrototypeAuthorization("feasibility-train-only-encoder"),
    )
    train_embedding = encoder.score(
        train_fit.drop(columns=["label_state", "horizon_years"], errors="ignore")
    ).embedding
    validation_embedding = encoder.score(
        validation_fit.drop(columns=["label_state", "horizon_years"], errors="ignore")
    ).embedding
    train_y = _encode_labels(train_fit["label_state"])
    validation_y = _encode_labels(validation_fit["label_state"])
    head, selection = _select_logistic_hyperparameter(
        train_embedding,
        train_y,
        validation_embedding,
        validation_y,
        seed,
    )
    scored = merged.copy()
    scored["prediction"] = np.nan
    for partition in ("train", *EVAL_PARTITIONS):
        part = scored[scored["split"] == partition]
        if part.empty:
            continue
        emb = encoder.score(
            part.drop(columns=["label_state", "horizon_years"], errors="ignore")
        ).embedding
        scored.loc[part.index, "prediction"] = head.predict_proba(emb)[:, 1]
    minimum_partition_events = int(
        load_config()["readiness"]["minimum_events_per_evaluation_partition"]
    )
    report = {
        "model_identifier": "representation_head",
        "encoder": {
            "latent_dimensions": encoder.config.latent_dimensions,
            "retained_source_features": list(encoder._feature_columns),
            "train_only_fit": True,
        },
        "head_hyperparameter_selection": selection,
        "partition_reports": [
            _partition_report(
                partition,
                scored[scored["split"] == partition],
                minimum_partition_events,
            )
            for partition in EVAL_PARTITIONS
        ],
        "train_to_holdout_drift_warning": _drift_report(
            scored[scored["split"] == "train"],
            scored[scored["split"] == "temporal_holdout"],
        ),
    }
    return report, encoder, head, scored


def _shortcut_checks(
    merged: pd.DataFrame,
    predictor_columns: list[str],
    seed: int,
    baseline_scored: pd.DataFrame,
) -> dict[str, Any]:
    train = merged[
        (merged["split"] == "train") & merged["label_state"].isin(FIT_LABEL_STATES)
    ].copy()
    test = merged[
        (merged["split"] == "test") & merged["label_state"].isin(FIT_LABEL_STATES)
    ].copy()
    permuted = train.copy()
    permutation = np.random.default_rng(seed).permutation(len(permuted))
    permuted["label_state"] = permuted["label_state"].iloc[permutation].to_numpy()
    preprocessor, retained, _, _ = _fit_preprocessor(permuted, predictor_columns)
    train_x = np.asarray(
        preprocessor.transform(_prepare_frame(permuted, retained)), dtype=float
    )
    test_x = np.asarray(
        preprocessor.transform(_prepare_frame(test, retained)), dtype=float
    )
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
        max_iter=1000,
    )
    model.fit(train_x, _encode_labels(permuted["label_state"]))
    test_pred = model.predict_proba(test_x)[:, 1]
    y_test = _encode_labels(test["label_state"])
    perm_auc = _safe_auc(y_test, test_pred)
    base_test = baseline_scored[
        (baseline_scored["split"] == "test")
        & baseline_scored["label_state"].isin(FIT_LABEL_STATES)
    ]
    base_auc = _safe_auc(
        _encode_labels(base_test["label_state"]),
        np.asarray(base_test["prediction"].astype(float), dtype=float),
    )
    suspicious = (
        perm_auc is not None
        and base_auc is not None
        and abs(base_auc - perm_auc) < 0.02
    )

    laboratory_roots = {
        "hba1c",
        "glucose",
        "insulin",
        "c_peptide",
        "ca_19_9",
        "total_cholesterol",
        "hdl",
        "ldl",
        "triglycerides",
        "haemoglobin",
        "platelets",
        "alt",
        "creatinine",
        "alkaline_phosphatase",
    }
    lab_subset = [
        column
        for column in predictor_columns
        if column.split("__", 1)[0] in laboratory_roots
    ]
    if not lab_subset:
        lab_subset = predictor_columns[: min(20, len(predictor_columns))]
    ablation_rows: list[dict[str, object]] = []
    for name, subset in (
        ("all_allowed_retained_features", predictor_columns),
        ("compact_metabolic_laboratory_subset", lab_subset),
    ):
        p, retained_cols, _, _ = _fit_preprocessor(train, subset)
        train_x2 = np.asarray(
            p.transform(_prepare_frame(train, retained_cols)), dtype=float
        )
        test_x2 = np.asarray(
            p.transform(_prepare_frame(test, retained_cols)), dtype=float
        )
        m = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
            max_iter=1000,
        )
        m.fit(train_x2, _encode_labels(train["label_state"]))
        pred = m.predict_proba(test_x2)[:, 1]
        ablation_rows.append(
            {
                "ablation": name,
                "retained_feature_count": len(retained_cols),
                "auroc_synthetic_technical_feasibility_only": _safe_auc(
                    _encode_labels(test["label_state"]), pred
                ),
                "average_precision_synthetic_technical_feasibility_only": _safe_ap(
                    _encode_labels(test["label_state"]), pred
                ),
            }
        )
    return {
        "label_permutation_sanity": {
            "test_auroc_synthetic_technical_feasibility_only": perm_auc,
            "baseline_test_auroc_synthetic_technical_feasibility_only": base_auc,
            "suspicious_similarity_flag": suspicious,
        },
        "feature_ablation": ablation_rows,
    }


def _build_experiment_id(
    run_manifest: dict[str, Any],
    endpoint_id: str,
    horizon: int,
    seed: int,
    approval_reference: str,
    feature_manifest: dict[str, Any],
    split_manifest_sha: str,
) -> str:
    payload = {
        "run_id": run_manifest["run_id"],
        "cohort_class": run_manifest["cohort_class"],
        "endpoint_id": endpoint_id,
        "horizon_years": horizon,
        "seed": seed,
        "approval_digest": _digest(approval_reference),
        "feature_manifest_sha": feature_manifest.get("feature_matrix_sha256", ""),
        "split_manifest_sha": split_manifest_sha,
        "model_config": {
            "baseline_grid": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
            "representation": {"latent_dimensions": "dynamic_train_bound"},
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return (
        f"{run_manifest['run_id']}-{run_manifest['cohort_class']}-"
        f"{endpoint_id}-{horizon}y-{seed}-{digest}"
    )


def _assert_partition_values(matrix: pd.DataFrame) -> None:
    expected = {"train", "validation", "test", "temporal_holdout"}
    observed = {str(value) for value in matrix["split"].dropna().unique()}
    if not observed.issubset(expected):
        raise ValueError("Feature matrix contains unsupported split assignments")
    for required in expected:
        if required not in observed:
            raise ValueError(f"Missing required split partition: {required}")


def _validate_split_linkage(
    matrix: pd.DataFrame,
    split_manifest: dict[str, Any],
) -> None:
    assignments, fingerprints = _split_map(split_manifest)
    for row in matrix.itertuples(index=False):
        patient_id = str(row.patient_id)
        split = str(row.split)
        if assignments.get(patient_id) != split:
            raise ValueError(
                "Feature matrix split assignment mismatch against split manifest"
            )
        expected_fingerprint = _digest(patient_id)
        if fingerprints.get(patient_id) != expected_fingerprint:
            raise ValueError("Split fingerprint mismatch")


def _assert_hash_chain(
    matrix: pd.DataFrame,
    cohort_manifest_sha: str,
    split_manifest_sha: str,
) -> None:
    cohort_values = {
        str(value)
        for value in matrix["source_cohort_manifest_sha256"].dropna().unique()
    }
    split_values = {
        str(value) for value in matrix["source_split_manifest_sha256"].dropna().unique()
    }
    if cohort_values != {cohort_manifest_sha}:
        raise ValueError(
            "Feature matrix source cohort hash does not match cohort manifest"
        )
    if split_values != {split_manifest_sha}:
        raise ValueError(
            "Feature matrix source split hash does not match split manifest"
        )


def _existing_artifact_hashes(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(output_dir)).replace("\\", "/")
            if relative == "experiment_manifest.json":
                continue
            hashes[relative] = _sha256(path)
    return hashes


def _check_resume(output_dir: Path) -> dict[str, Any] | None:
    manifest_path = output_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _load_json(manifest_path)
    recorded = cast(dict[str, str], manifest.get("artifact_hashes", {}))
    current = _existing_artifact_hashes(output_dir)
    if recorded != current:
        raise ValueError(
            "Existing experiment artifacts do not match recorded hash manifest"
        )
    return manifest


def build_feasibility_plan(
    run_path: Path,
    endpoint_id: str,
    horizon: int,
    authorization: SyntheticFeasibilityAuthorization,
    seed: int,
    artifact_root: Path,
) -> dict[str, Any]:
    _ = authorization
    _validate_endpoint_horizon(endpoint_id, horizon)
    paths = _path_for(run_path, endpoint_id, horizon)
    _assert_required_files(paths)
    run_manifest = _load_json(paths.run_manifest)
    generation_manifest = _load_json(paths.generation_manifest)
    cohort_manifest = _load_json(paths.cohort_manifest)
    split_manifest = _load_json(paths.split_manifest)
    feature_manifest = _load_json(paths.feature_manifest)
    endpoint_protocol = _load_json(paths.endpoint_protocol)
    if run_manifest.get("status") not in {
        "completed",
        "completed_not_ready",
        "completed_prototype_ready",
    }:
        raise ValueError("Feasibility requires a completed immutable source run")
    if not _durable_synthea(generation_manifest):
        raise ValueError("Durable Synthea provenance evidence is required")
    if not bool(feature_manifest.get("validation_passed", False)):
        raise ValueError("Feature manifest validation must have passed")
    if endpoint_protocol.get("endpoint_id") != endpoint_id:
        raise ValueError("Endpoint protocol does not match requested endpoint")
    artifacts = {
        str(paths.run_manifest): run_manifest,
        str(paths.generation_manifest): generation_manifest,
        str(paths.cohort_manifest): cohort_manifest,
        str(paths.split_manifest): split_manifest,
        str(paths.feature_manifest): feature_manifest,
    }
    discrepancies = _provenance_discrepancies(artifacts)
    experiment_id = _build_experiment_id(
        run_manifest,
        endpoint_id,
        horizon,
        seed,
        authorization.approval_reference,
        feature_manifest,
        _sha256(paths.split_manifest),
    )
    output_dir = (
        artifact_root
        / str(run_manifest["run_id"])
        / endpoint_id
        / f"{horizon}y"
        / experiment_id
    )
    return {
        "experiment_id": experiment_id,
        "source_run_id": str(run_manifest["run_id"]),
        "cohort_class": str(run_manifest["cohort_class"]),
        "endpoint_id": endpoint_id,
        "horizon_years": horizon,
        "seed": seed,
        "output_dir": str(output_dir),
        "source_provenance_discrepancy": bool(discrepancies),
        "source_provenance_discrepancy_details": discrepancies,
        "approval_reference_digest": _digest(authorization.approval_reference),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in {
                "run_manifest": paths.run_manifest,
                "generation_manifest": paths.generation_manifest,
                "cohort_manifest": paths.cohort_manifest,
                "split_manifest": paths.split_manifest,
                "split_assignments": paths.split_assignments,
                "endpoint_protocol": paths.endpoint_protocol,
                "feature_manifest": paths.feature_manifest,
                "feature_matrix": paths.feature_matrix,
                "feature_lineage": paths.feature_lineage,
                "feature_registry": paths.feature_registry,
                "horizon_labels": paths.labels,
            }.items()
        },
        **SYNTHETIC_FLAGS,
    }


def run_model_feasibility(
    run_path: Path,
    endpoint_id: str,
    horizon: int,
    authorization: SyntheticFeasibilityAuthorization,
    *,
    seed: int = 1729,
    artifact_root: Path = Path("artifacts/model_feasibility"),
    execute: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    plan = build_feasibility_plan(
        run_path,
        endpoint_id,
        horizon,
        authorization,
        seed,
        artifact_root,
    )
    if not execute:
        return {"execute": False, "plan": plan}
    output_dir = Path(str(plan["output_dir"]))
    if output_dir.exists():
        resumed = _check_resume(output_dir)
        if resumed is not None:
            return {
                "execute": True,
                "resumed": True,
                "experiment_id": resumed.get("experiment_id"),
                "output_dir": str(output_dir),
            }
        if not overwrite:
            raise ValueError(
                "Experiment directory exists; overwrite is disabled and resume hash check failed"
            )
    paths = _path_for(run_path, endpoint_id, horizon)
    matrix = pd.read_parquet(paths.feature_matrix)
    labels = pd.read_parquet(paths.labels)
    lineage = pd.read_parquet(paths.feature_lineage)
    split_manifest = _load_json(paths.split_manifest)
    run_manifest = _load_json(paths.run_manifest)
    registry = cast(
        list[dict[str, object]],
        json.loads(paths.feature_registry.read_text(encoding="utf-8")),
    )
    _assert_identity(matrix, labels, run_manifest, endpoint_id, horizon)
    _assert_partition_values(matrix)
    _validate_split_linkage(matrix, split_manifest)
    _assert_hash_chain(
        matrix, _sha256(paths.cohort_manifest), _sha256(paths.split_manifest)
    )
    patient_isolation = _assert_patient_isolation(matrix)
    if not patient_isolation["passed"]:
        raise ValueError("Patient isolation check failed")
    leakage = _feature_policy_audit(matrix, registry, lineage)
    if not leakage["passed"]:
        raise ValueError("Feature leakage audit failed")

    merged = _merge_features_labels(matrix, labels)

    predictor_columns = _predictor_columns(merged, registry)
    (
        baseline_report,
        baseline_preprocessor,
        baseline_model,
        baseline_scored,
        all_null,
        zero_var,
    ) = _evaluate_model(merged, predictor_columns, seed, "baseline")
    representation_report, encoder, head, representation_scored = _representation_head(
        merged, seed
    )
    shortcuts = _shortcut_checks(merged, predictor_columns, seed, baseline_scored)

    # Build manifests before writing so we can atomically fail closed if needed.
    feature_selection_manifest = {
        "retained_predictor_columns": predictor_columns,
        "all_null_training_columns": all_null,
        "zero_variance_training_columns": zero_var,
        "ordered_retained_feature_registry": predictor_columns,
        "simulation_only": True,
    }
    preprocessing_manifest = {
        "train_only_fit": True,
        "baseline_preprocessor": "median_impute_plus_robust_scale_and_ohe",
        "representation_encoder": {
            "config": {
                "latent_dimensions": encoder.config.latent_dimensions,
                "reconstruction_weight": encoder.config.reconstruction_weight,
                "latent_distance_weight": encoder.config.latent_distance_weight,
            },
            "retained_source_features": list(encoder._feature_columns),
            "train_only_fit": True,
        },
        "simulation_only": True,
    }
    authorization_payload = {
        "authorization_type": "SyntheticFeasibilityAuthorization",
        "synthetic_feasibility": authorization.synthetic_feasibility,
        "approval_reference_digest": _digest(authorization.approval_reference),
        **SYNTHETIC_FLAGS,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = output_dir / "baseline"
    representation_dir = output_dir / "representation_head"
    _atomic_write_pickle(
        baseline_dir / "model.pkl", (baseline_preprocessor, baseline_model)
    )
    _atomic_write_json(
        baseline_dir / "evaluation_report.json",
        {
            **baseline_report,
            "source_provenance_discrepancy": plan["source_provenance_discrepancy"],
            "source_provenance_discrepancy_details": plan[
                "source_provenance_discrepancy_details"
            ],
            **SYNTHETIC_FLAGS,
        },
    )
    for partition in EVAL_PARTITIONS:
        _write_predictions(
            baseline_dir / f"predictions_{partition}.parquet",
            baseline_scored[baseline_scored["split"] == partition],
            endpoint_id,
            horizon,
            "baseline",
            str(plan["experiment_id"]),
        )

    _atomic_write_pickle(representation_dir / "encoder.pkl", encoder)
    _atomic_write_pickle(representation_dir / "head.pkl", head)
    _atomic_write_json(
        representation_dir / "evaluation_report.json",
        {
            **representation_report,
            "source_provenance_discrepancy": plan["source_provenance_discrepancy"],
            "source_provenance_discrepancy_details": plan[
                "source_provenance_discrepancy_details"
            ],
            **SYNTHETIC_FLAGS,
        },
    )
    for partition in EVAL_PARTITIONS:
        _write_predictions(
            representation_dir / f"predictions_{partition}.parquet",
            representation_scored[representation_scored["split"] == partition],
            endpoint_id,
            horizon,
            "representation_head",
            str(plan["experiment_id"]),
        )

    shortcut_payload = {
        **shortcuts,
        "source_provenance_discrepancy": plan["source_provenance_discrepancy"],
        "source_provenance_discrepancy_details": plan[
            "source_provenance_discrepancy_details"
        ],
        **SYNTHETIC_FLAGS,
    }
    _atomic_write_json(output_dir / "shortcut_checks.json", shortcut_payload)
    _atomic_write_json(
        output_dir / "leakage_audit.json", {**leakage, **SYNTHETIC_FLAGS}
    )
    _atomic_write_json(
        output_dir / "split_audit.json", {**patient_isolation, **SYNTHETIC_FLAGS}
    )
    _atomic_write_json(output_dir / "authorization.json", authorization_payload)
    _atomic_write_json(
        output_dir / "preprocessing_manifest.json", preprocessing_manifest
    )
    _atomic_write_json(
        output_dir / "feature_selection_manifest.json", feature_selection_manifest
    )
    _atomic_write_json(
        output_dir / "input_artifact_hashes.json",
        cast(dict[str, Any], plan["inputs"]),
    )

    report_payload = {
        "experiment_id": plan["experiment_id"],
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": "synthetic technical feasibility only",
        "endpoint_id": endpoint_id,
        "horizon_years": horizon,
        "cohort_class": run_manifest["cohort_class"],
        "source_provenance_discrepancy": plan["source_provenance_discrepancy"],
        "source_provenance_discrepancy_details": plan[
            "source_provenance_discrepancy_details"
        ],
        "interpretation": (
            "This report is synthetic technical feasibility only and does not "
            "establish clinical performance or utility."
        ),
        **SYNTHETIC_FLAGS,
    }
    _atomic_write_text(
        output_dir / "SYNTHETIC_FEASIBILITY_REPORT.md",
        "# Synthetic Feasibility Report\n\n"
        "All reported metrics are synthetic technical feasibility only.\n"
        "No diagnostic, screening, or clinical risk claims are allowed.\n",
    )
    _atomic_write_text(
        output_dir / "MODEL_CARD.md",
        _model_card(
            {
                "source_run_id": run_manifest["run_id"],
                "cohort_class": run_manifest["cohort_class"],
                "endpoint_id": endpoint_id,
                "horizon_years": horizon,
                "source_provenance_discrepancy": plan["source_provenance_discrepancy"],
            }
        ),
    )

    manifest_payload = {
        "experiment_id": plan["experiment_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "source_run_id": run_manifest["run_id"],
        "cohort_class": run_manifest["cohort_class"],
        "endpoint_id": endpoint_id,
        "horizon_years": horizon,
        "seed": seed,
        "approval_reference_digest": _digest(authorization.approval_reference),
        "source_provenance_discrepancy": plan["source_provenance_discrepancy"],
        "source_provenance_discrepancy_details": plan[
            "source_provenance_discrepancy_details"
        ],
        "input_artifacts": plan["inputs"],
        "feature_definition_registry_sha256": _sha256(paths.feature_registry),
        "endpoint_protocol_sha256": _sha256(paths.endpoint_protocol),
        "split_manifest_sha256": _sha256(paths.split_manifest),
        "source_run_configuration_sha256": run_manifest.get("configuration_sha256", ""),
        "package_metadata": {
            "config_hash": config_sha256(load_config()),
            "git_sha": run_manifest.get("artifact_sha256", {}).get(
                "git_sha", "unavailable"
            ),
        },
        "partition_label_counts": {
            partition: {
                state: int(
                    (
                        (merged["split"] == partition)
                        & (merged["label_state"].astype("string") == state)
                    ).sum()
                )
                for state in (
                    "positive",
                    "eligible_negative",
                    "censored",
                    "competing_death",
                    "excluded",
                )
            }
            for partition in ("train", "validation", "test", "temporal_holdout")
        },
        "model_status": "created",
        "clinical_use_restrictions": {
            **SYNTHETIC_FLAGS,
            "not_for_clinical_use": MANDATORY_MODEL_CARD_RESTRICTION,
        },
        "final_status": "completed",
    }
    _atomic_write_json(output_dir / "synthetic_feasibility_report.json", report_payload)
    _atomic_write_json(output_dir / "experiment_manifest.json", manifest_payload)
    manifest_payload["artifact_hashes"] = _existing_artifact_hashes(output_dir)
    _atomic_write_json(output_dir / "experiment_manifest.json", manifest_payload)
    return {
        "execute": True,
        "resumed": False,
        "experiment_id": plan["experiment_id"],
        "output_dir": str(output_dir),
        **SYNTHETIC_FLAGS,
    }
