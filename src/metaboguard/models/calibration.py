"""Train-only calibration boundary.

Calibration is fitted exclusively on validation-split rows so that no test or
temporal-holdout row ever influences the calibration mapping, matching the
train/validation/test/temporal-holdout split contract used elsewhere in this
project (see docs/v1-docs/FUTURE_RISK_EVALUATION.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import brier_score_loss  # type: ignore[import-untyped]

from metaboguard.models.ssl_encoder import SyntheticPrototypeAuthorization

MANDATORY_MODEL_CARD_RESTRICTION = (
    "This calibration artifact was fitted exclusively on Synthea-derived synthetic "
    "data. It is not a diagnostic device and must not be used for clinical "
    "decision-making, patient screening, risk assessment, medical advice, treatment "
    "decisions, triage, or patient care. Calibrated probabilities demonstrate "
    "software and research-workflow feasibility only; they establish no clinical "
    "calibration, validity, utility, prevalence, or generalisability. External "
    "validation on an approved real longitudinal cohort is required before any "
    "clinical interpretation."
)

FIT_LABEL_STATES = frozenset({"positive", "eligible_negative"})

ISOTONIC_MIN_ROWS = 50
ISOTONIC_MIN_EVENTS = 10


class CalibrationAuthorizationError(ValueError):
    """Raised when calibration is fitted outside the authorized validation split."""


@dataclass(frozen=True, slots=True)
class CalibrationCurveBin:
    """One decile bin of a calibration curve."""

    row_count: int
    mean_predicted: float
    observed_rate: float


@dataclass(frozen=True, slots=True)
class CalibrationEvaluation:
    """Calibration diagnostics for one evaluated partition."""

    method: str
    brier_score: float
    calibration_intercept: float
    calibration_slope: float
    calibration_curve: list[CalibrationCurveBin] = field(default_factory=list)


def _label_target(label_state: pd.Series) -> np.ndarray:
    return (label_state == "positive").astype(int).to_numpy()


def _decile_curve(y_true: np.ndarray, y_pred: np.ndarray) -> list[CalibrationCurveBin]:
    if len(y_true) == 0:
        return []
    bin_count = min(10, len(np.unique(y_pred)))
    if bin_count < 2:
        return [
            CalibrationCurveBin(
                row_count=len(y_true),
                mean_predicted=float(np.mean(y_pred)),
                observed_rate=float(np.mean(y_true)),
            )
        ]
    edges = np.quantile(y_pred, np.linspace(0, 1, bin_count + 1))
    edges[0] -= 1e-9
    bin_indices = np.clip(
        np.digitize(y_pred, edges[1:-1], right=True), 0, bin_count - 1
    )
    bins: list[CalibrationCurveBin] = []
    for bin_index in range(bin_count):
        mask = bin_indices == bin_index
        if not np.any(mask):
            continue
        bins.append(
            CalibrationCurveBin(
                row_count=int(np.sum(mask)),
                mean_predicted=float(np.mean(y_pred[mask])),
                observed_rate=float(np.mean(y_true[mask])),
            )
        )
    return bins


def _calibration_intercept_slope(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    clipped = np.clip(y_pred, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true) - np.mean(y_pred)), float("nan")
    fit = LogisticRegression().fit(logit.reshape(-1, 1), y_true)
    return float(fit.intercept_[0]), float(fit.coef_[0][0])


class RiskCalibrator:
    """Fit a validation-only calibration mapping over raw risk-head probabilities."""

    def __init__(self) -> None:
        self._method: str | None = None
        self._isotonic: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None

    def fit(
        self, validation_frame: pd.DataFrame, probability_column: str
    ) -> RiskCalibrator:
        """Fit strictly on validation-split, frozen-label rows."""
        if "split" not in validation_frame:
            raise ValueError("calibration frame must include a split column")
        if not validation_frame["split"].eq("validation").all():
            raise CalibrationAuthorizationError(
                "calibration must be fitted on validation split rows only"
            )
        if "label_state" not in validation_frame:
            raise ValueError("calibration frame must include a label_state column")
        fittable = validation_frame[
            validation_frame["label_state"].isin(FIT_LABEL_STATES)
        ]
        if fittable.empty:
            raise ValueError(
                "calibration frame has no positive/eligible_negative rows to fit"
            )
        probability = fittable[probability_column].to_numpy(dtype=float)
        target = _label_target(fittable["label_state"])
        if len(np.unique(target)) < 2:
            raise ValueError("calibration frame must include both label classes")
        event_count = int(np.sum(target))
        if len(fittable) >= ISOTONIC_MIN_ROWS and event_count >= ISOTONIC_MIN_EVENTS:
            self._method = "isotonic"
            self._isotonic = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            ).fit(probability, target)
        else:
            self._method = "platt"
            self._platt = LogisticRegression().fit(probability.reshape(-1, 1), target)
        return self

    def calibrate(self, probability: np.ndarray) -> np.ndarray:
        """Map raw probabilities through the fitted validation-only calibrator."""
        if self._method is None:
            raise RuntimeError("calibrator must be fitted before use")
        if self._method == "isotonic":
            if self._isotonic is None:
                raise RuntimeError("calibrator must be fitted before use")
            return np.asarray(self._isotonic.predict(probability), dtype=float)
        if self._platt is None:
            raise RuntimeError("calibrator must be fitted before use")
        return np.asarray(
            self._platt.predict_proba(probability.reshape(-1, 1))[:, 1], dtype=float
        )

    def fit_synthetic_prototype(
        self,
        validation_frame: pd.DataFrame,
        probability_column: str,
        authorization: SyntheticPrototypeAuthorization,
    ) -> RiskCalibrator:
        """Fit an explicitly approved, simulation-only prototype rehearsal."""
        if "simulation_only" not in validation_frame:
            raise ValueError("synthetic prototype frame must declare simulation_only")
        if not validation_frame["simulation_only"].eq(True).all():
            raise ValueError(
                "synthetic prototype frame must contain only synthetic rows"
            )
        _ = authorization
        return self.fit(validation_frame, probability_column)

    def evaluate(
        self, frame: pd.DataFrame, probability_column: str
    ) -> CalibrationEvaluation:
        """Score raw probabilities against observed outcomes for one partition."""
        if self._method is None:
            raise RuntimeError("calibrator must be fitted before evaluation")
        fittable = frame[frame["label_state"].isin(FIT_LABEL_STATES)]
        if fittable.empty:
            raise ValueError(
                "evaluation frame has no positive/eligible_negative rows to score"
            )
        raw_probability = fittable[probability_column].to_numpy(dtype=float)
        target = _label_target(fittable["label_state"])
        calibrated = self.calibrate(raw_probability)
        intercept, slope = _calibration_intercept_slope(target, calibrated)
        return CalibrationEvaluation(
            method=self._method,
            brier_score=float(brier_score_loss(target, calibrated)),
            calibration_intercept=intercept,
            calibration_slope=slope,
            calibration_curve=_decile_curve(target, calibrated),
        )
