"""Research-only unsupervised metabolic representation learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, RobustScaler  # type: ignore[import-untyped]

from metaboguard.features.extraction import IDENTIFIER_COLUMNS


class ResearchModelAuthorizationError(ValueError):
    """Raised when immutable readiness evidence does not authorize fitting."""


@dataclass(frozen=True, slots=True)
class SyntheticPrototypeAuthorization:
    """Explicit authorization for a bounded synthetic prototype rehearsal."""

    approval_reference: str

    def __post_init__(self) -> None:
        if not self.approval_reference.strip():
            raise ValueError("synthetic prototype approval_reference must be non-empty")


@dataclass(frozen=True, slots=True)
class RobustPCAEncoderConfig:
    """Deterministic configuration for the research-only representation encoder."""

    latent_dimensions: int = 16
    reconstruction_weight: float = 0.7
    latent_distance_weight: float = 0.3

    def __post_init__(self) -> None:
        if self.latent_dimensions < 1:
            raise ValueError("latent_dimensions must be positive")
        if (
            self.reconstruction_weight < 0
            or self.latent_distance_weight < 0
            or self.reconstruction_weight + self.latent_distance_weight != 1
        ):
            raise ValueError("deviation score weights must be non-negative and sum to 1")


@dataclass(frozen=True, slots=True)
class MetabolicDeviation:
    """Label-free representation and deviation outputs for research review."""

    embedding: np.ndarray
    reconstruction_error: np.ndarray
    latent_distance: np.ndarray
    metabolic_deviation_score: np.ndarray
    reference_percentile: np.ndarray


def _authorize_research_fit(capability_report: dict[str, Any]) -> None:
    if bool(capability_report.get("simulation_only", True)):
        raise ResearchModelAuthorizationError(
            "Research model fitting is forbidden for simulation-only artifacts"
        )
    if capability_report.get("overall_decision") != "eligible_for_future_model_research":
        raise ResearchModelAuthorizationError(
            "Research model fitting requires eligible_for_future_model_research"
        )
    if not bool(capability_report.get("clinical_model_research_authorized", False)):
        raise ResearchModelAuthorizationError(
            "Research model fitting requires explicit clinical research authorization"
        )


class RobustPCAEncoder:
    """Fit a label-free robust-PCA encoder only on authorized training features."""

    def __init__(self, config: RobustPCAEncoderConfig | None = None) -> None:
        self.config = config or RobustPCAEncoderConfig()
        self._preprocessor: ColumnTransformer | None = None
        self._pca: PCA | None = None
        self._feature_columns: list[str] = []
        self._categorical_columns: list[str] = []
        self._latent_center: np.ndarray | None = None
        self._reference_scores: np.ndarray | None = None
        self._reconstruction_center = 0.0
        self._reconstruction_scale = 1.0
        self._distance_center = 0.0
        self._distance_scale = 1.0

    def fit(
        self, feature_matrix: pd.DataFrame, capability_report: dict[str, Any]
    ) -> RobustPCAEncoder:
        """Fit from authorized, pre-index training rows without outcome labels."""
        _authorize_research_fit(capability_report)
        return self._fit_training_features(feature_matrix)

    def fit_synthetic_prototype(
        self,
        feature_matrix: pd.DataFrame,
        authorization: SyntheticPrototypeAuthorization,
    ) -> RobustPCAEncoder:
        """Fit an explicitly approved, simulation-only prototype rehearsal."""
        if "simulation_only" not in feature_matrix:
            raise ValueError("synthetic prototype matrix must declare simulation_only")
        if not feature_matrix["simulation_only"].eq(True).all():
            raise ValueError("synthetic prototype matrix must contain only synthetic rows")
        _ = authorization
        return self._fit_training_features(feature_matrix)

    def _fit_training_features(self, feature_matrix: pd.DataFrame) -> RobustPCAEncoder:
        if "split" not in feature_matrix:
            raise ValueError("feature matrix must include a split column")
        if not feature_matrix["split"].eq("train").all():
            raise ValueError("unsupervised fitting accepts train split rows only")
        self._feature_columns = [
            column for column in feature_matrix.columns if column not in IDENTIFIER_COLUMNS
        ]
        if not self._feature_columns:
            raise ValueError("feature matrix has no trainable feature columns")
        forbidden = [
            column
            for column in self._feature_columns
            if any(token in column.lower() for token in ("label", "outcome", "diagnosis"))
        ]
        if forbidden:
            raise ValueError(f"feature matrix includes prohibited outcome columns: {forbidden}")
        features = feature_matrix[self._feature_columns]
        numeric_columns = list(features.select_dtypes(include=[np.number]).columns)
        categorical_columns = [
            column for column in self._feature_columns if column not in numeric_columns
        ]
        numeric_columns = [column for column in numeric_columns if features[column].notna().any()]
        categorical_columns = [
            column for column in categorical_columns if features[column].notna().any()
        ]
        self._feature_columns = numeric_columns + categorical_columns
        self._categorical_columns = categorical_columns
        features = self._prepare_features(feature_matrix)
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
        if not transformers:
            raise ValueError("feature matrix has no usable feature columns")
        self._preprocessor = ColumnTransformer(transformers, sparse_threshold=0.0)
        transformed = np.asarray(self._preprocessor.fit_transform(features), dtype=float)
        max_dimensions = min(transformed.shape)
        if self.config.latent_dimensions > max_dimensions:
            raise ValueError(
                "latent_dimensions must not exceed the number of training rows "
                "or transformed features"
            )
        self._pca = PCA(n_components=self.config.latent_dimensions, svd_solver="full")
        embedding = self._pca.fit_transform(transformed)
        self._latent_center = np.median(embedding, axis=0)
        reconstruction, distance = self._raw_scores(transformed, embedding)
        self._reconstruction_center, self._reconstruction_scale = _robust_location_scale(
            reconstruction
        )
        self._distance_center, self._distance_scale = _robust_location_scale(distance)
        self._reference_scores = self._combine_scores(reconstruction, distance)
        return self

    def score(self, feature_matrix: pd.DataFrame) -> MetabolicDeviation:
        """Return label-free embeddings and deviation signals for research review."""
        if self._preprocessor is None or self._pca is None:
            raise RuntimeError("encoder must be fitted before scoring")
        if self._reference_scores is None:
            raise RuntimeError("encoder reference distribution is unavailable")
        missing = set(self._feature_columns).difference(feature_matrix.columns)
        if missing:
            raise ValueError(f"feature matrix is missing fitted columns: {sorted(missing)}")
        transformed = np.asarray(
            self._preprocessor.transform(self._prepare_features(feature_matrix)),
            dtype=float,
        )
        embedding = self._pca.transform(transformed)
        reconstruction, distance = self._raw_scores(transformed, embedding)
        scores = self._combine_scores(reconstruction, distance)
        percentile = (
            np.searchsorted(np.sort(self._reference_scores), scores, side="right")
            / len(self._reference_scores)
            * 100
        )
        return MetabolicDeviation(
            embedding=embedding,
            reconstruction_error=reconstruction,
            latent_distance=distance,
            metabolic_deviation_score=scores,
            reference_percentile=percentile,
        )

    def _prepare_features(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        features = feature_matrix[self._feature_columns].copy()
        for column in self._categorical_columns:
            features[column] = (
                features[column].astype(object).where(features[column].notna(), np.nan)
            )
        return features

    def _raw_scores(
        self, transformed: np.ndarray, embedding: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._pca is None or self._latent_center is None:
            raise RuntimeError("encoder must be fitted before scoring")
        reconstruction = self._pca.inverse_transform(embedding)
        reconstruction_error = np.mean((transformed - reconstruction) ** 2, axis=1)
        latent_distance = np.linalg.norm(embedding - self._latent_center, axis=1)
        return reconstruction_error, latent_distance

    def _combine_scores(self, reconstruction: np.ndarray, distance: np.ndarray) -> np.ndarray:
        return self.config.reconstruction_weight * (
            (reconstruction - self._reconstruction_center) / self._reconstruction_scale
        ) + self.config.latent_distance_weight * (
            (distance - self._distance_center) / self._distance_scale
        )


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    scale = float(np.median(np.abs(values - center)))
    return center, scale if scale > 0 else 1.0
