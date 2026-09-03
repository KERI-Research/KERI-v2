import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from metaboguard.models.calibration import (
    CalibrationAuthorizationError,
    RiskCalibrator,
)
from metaboguard.models.ssl_encoder import SyntheticPrototypeAuthorization


def _validation_frame(rows: int, event_rate: float) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    label_state = rng.choice(
        ["positive", "eligible_negative"], size=rows, p=[event_rate, 1 - event_rate]
    )
    probability = rng.uniform(0, 1, rows)
    return pd.DataFrame(
        {
            "split": ["validation"] * rows,
            "label_state": label_state,
            "raw_probability": probability,
        }
    )


def test_calibrator_rejects_non_validation_rows() -> None:
    frame = _validation_frame(60, 0.3).assign(split="train")
    with pytest.raises(CalibrationAuthorizationError, match="validation split"):
        RiskCalibrator().fit(frame, "raw_probability")


def test_calibrator_selects_isotonic_with_enough_rows_and_events() -> None:
    frame = _validation_frame(60, 0.3)
    calibrator = RiskCalibrator().fit(frame, "raw_probability")
    assert calibrator._method == "isotonic"
    calibrated = calibrator.calibrate(frame["raw_probability"].to_numpy())
    assert calibrated.shape == (60,)
    assert (calibrated >= 0).all() and (calibrated <= 1).all()


def test_calibrator_falls_back_to_platt_with_few_rows() -> None:
    frame = _validation_frame(12, 0.5)
    calibrator = RiskCalibrator().fit(frame, "raw_probability")
    assert calibrator._method == "platt"


def test_calibrator_evaluate_reports_brier_and_curve() -> None:
    frame = _validation_frame(60, 0.3)
    calibrator = RiskCalibrator().fit(frame, "raw_probability")
    evaluation = calibrator.evaluate(frame, "raw_probability")
    assert evaluation.method == "isotonic"
    assert 0 <= evaluation.brier_score <= 1
    assert len(evaluation.calibration_curve) > 0


def test_calibrator_allows_explicitly_approved_synthetic_prototype() -> None:
    frame = _validation_frame(60, 0.3).assign(simulation_only=True)
    calibrator = RiskCalibrator().fit_synthetic_prototype(
        frame, "raw_probability", SyntheticPrototypeAuthorization("approval-2026")
    )
    assert calibrator._method == "isotonic"
    with pytest.raises(ValueError, match="only synthetic rows"):
        RiskCalibrator().fit_synthetic_prototype(
            frame.assign(simulation_only=False),
            "raw_probability",
            SyntheticPrototypeAuthorization("approval-2026"),
        )


def test_calibrator_requires_fit_before_use() -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        RiskCalibrator().calibrate(np.array([0.1, 0.2]))
