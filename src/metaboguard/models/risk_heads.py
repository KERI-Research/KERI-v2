"""Risk-head model boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, RobustScaler  # type: ignore[import-untyped]

from metaboguard.features.extraction import IDENTIFIER_COLUMNS
from metaboguard.models.ssl_encoder import SyntheticPrototypeAuthorization

MANDATORY_MODEL_CARD_RESTRICTION = (
    "This risk-head artifact was trained and evaluated exclusively on Synthea-derived "
    "synthetic data. It is not a diagnostic device and must not be used for clinical "
    "decision-making, patient screening, risk assessment, medical advice, treatment "
    "decisions, triage, or patient care. Reported probabilities demonstrate software "
    "and research-workflow feasibility only; they do not establish predictive "
    "performance, calibration, clinical validity, clinical utility, prevalence, or "
    "generalisability. External validation on an approved real longitudinal cohort "
    "is required before any clinical interpretation."
)

FIT_LABEL_STATES = frozenset({"positive", "eligible_negative"})

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

NON_PREDICTOR_COLUMNS = frozenset({"label_state", *IDENTIFIER_COLUMNS})


class RiskHeadAuthorizationError(ValueError):
    """Raised when immutable readiness evidence does not authorize fitting."""


@dataclass(frozen=True, slots=True)
class LogisticRiskHeadConfig:
    """Deterministic configuration for the bounded logistic risk head."""

    l2_penalty: float = 1.0
    max_iterations: int = 1000
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.l2_penalty <= 0:
            raise ValueError("l2_penalty must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True, slots=True)
class RiskPrediction:
    """Raw, uncalibrated risk-head outputs for research review."""

    probability: np.ndarray


def _authorize_research_fit(capability_report: dict[str, Any]) -> None:
    if bool(capability_report.get("simulation_only", True)):
        raise RiskHeadAuthorizationError(
            "Risk-head fitting is forbidden for simulation-only artifacts"
        )
    if (
        capability_report.get("overall_decision")
        != "eligible_for_future_model_research"
    ):
        raise RiskHeadAuthorizationError(
            "Risk-head fitting requires eligible_for_future_model_research"
        )
    if not bool(capability_report.get("clinical_model_research_authorized", False)):
        raise RiskHeadAuthorizationError(
            "Risk-head fitting requires explicit clinical research authorization"
        )


class LogisticRiskHead:
    """Fit a regularized logistic risk head only on authorized training rows."""

    def __init__(self, config: LogisticRiskHeadConfig | None = None) -> None:
        self.config = config or LogisticRiskHeadConfig()
        self._preprocessor: ColumnTransformer | None = None
        self._model: LogisticRegression | None = None
        self._feature_columns: list[str] = []
        self._categorical_columns: list[str] = []

    def fit(
        self, feature_matrix: pd.DataFrame, capability_report: dict[str, Any]
    ) -> LogisticRiskHead:
        """Fit from authorized, pre-index training rows with frozen label states."""
        _authorize_research_fit(capability_report)
        return self._fit_training_rows(feature_matrix)

    def fit_synthetic_prototype(
        self,
        feature_matrix: pd.DataFrame,
        authorization: SyntheticPrototypeAuthorization,
    ) -> LogisticRiskHead:
        """Fit an explicitly approved, simulation-only prototype rehearsal."""
        if "simulation_only" not in feature_matrix:
            raise ValueError("synthetic prototype matrix must declare simulation_only")
        if not feature_matrix["simulation_only"].eq(True).all():
            raise ValueError(
                "synthetic prototype matrix must contain only synthetic rows"
            )
        _ = authorization
        return self._fit_training_rows(feature_matrix)

    def _fit_training_rows(self, feature_matrix: pd.DataFrame) -> LogisticRiskHead:
        if "split" not in feature_matrix:
            raise ValueError("feature matrix must include a split column")
        if not feature_matrix["split"].eq("train").all():
            raise ValueError("risk-head fitting accepts train split rows only")
        if "label_state" not in feature_matrix:
            raise ValueError("feature matrix must include a label_state column")
        fittable = feature_matrix[
            feature_matrix["label_state"].isin(FIT_LABEL_STATES)
        ].copy()
        if fittable.empty:
            raise ValueError(
                "feature matrix has no positive/eligible_negative rows to fit"
            )
        target = (fittable["label_state"] == "positive").astype(int).to_numpy()
        if len(np.unique(target)) < 2:
            raise ValueError("fittable rows must include both label classes")
        self._feature_columns = [
            column
            for column in feature_matrix.columns
            if column not in NON_PREDICTOR_COLUMNS
        ]
        forbidden = [
            column
            for column in self._feature_columns
            if any(token in column.lower() for token in DENYLIST_TOKENS)
        ]
        if forbidden:
            raise ValueError(
                f"feature matrix includes prohibited outcome columns: {forbidden}"
            )
        if not self._feature_columns:
            raise ValueError("feature matrix has no trainable feature columns")
        features = fittable[self._feature_columns]
        numeric_columns = list(features.select_dtypes(include=[np.number]).columns)
        numeric_columns = [
            column for column in numeric_columns if features[column].notna().any()
        ]
        categorical_columns = [
            column
            for column in self._feature_columns
            if column not in numeric_columns and features[column].notna().any()
        ]
        self._feature_columns = numeric_columns + categorical_columns
        if not self._feature_columns:
            raise ValueError("feature matrix has no usable feature columns")
        prepared = self._prepare_features(fittable, categorical_columns)
        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if numeric_columns:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(strategy="median", add_indicator=True),
                            ),
                            ("scale", RobustScaler()),
                        ]
                    ),
                    numeric_columns,
                )
            )
        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_columns,
                )
            )
        self._preprocessor = ColumnTransformer(transformers, sparse_threshold=0.0)
        transformed = np.asarray(
            self._preprocessor.fit_transform(prepared), dtype=float
        )
        self._model = LogisticRegression(
            C=1.0 / self.config.l2_penalty,
            max_iter=self.config.max_iterations,
            class_weight="balanced",
            random_state=self.config.random_state,
        )
        self._model.fit(transformed, target)
        self._categorical_columns = categorical_columns
        return self

    def predict_proba(self, feature_matrix: pd.DataFrame) -> np.ndarray:
        """Return raw, uncalibrated positive-class probabilities."""
        if self._preprocessor is None or self._model is None:
            raise RuntimeError("risk head must be fitted before scoring")
        missing = set(self._feature_columns).difference(feature_matrix.columns)
        if missing:
            raise ValueError(
                f"feature matrix is missing fitted columns: {sorted(missing)}"
            )
        prepared = self._prepare_features(feature_matrix, self._categorical_columns)
        transformed = np.asarray(self._preprocessor.transform(prepared), dtype=float)
        return np.asarray(self._model.predict_proba(transformed)[:, 1], dtype=float)

    def score(self, feature_matrix: pd.DataFrame) -> RiskPrediction:
        """Return raw, uncalibrated risk-head outputs for research review."""
        return RiskPrediction(probability=self.predict_proba(feature_matrix))

    def _prepare_features(
        self, feature_matrix: pd.DataFrame, categorical_columns: list[str]
    ) -> pd.DataFrame:
        features = feature_matrix[self._feature_columns].copy()
        for column in categorical_columns:
            features[column] = (
                features[column].astype(object).where(features[column].notna(), np.nan)
            )
        return features
