import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from metaboguard.models.risk_heads import (
    LogisticRiskHead,
    LogisticRiskHeadConfig,
    RiskHeadAuthorizationError,
)
from metaboguard.models.ssl_encoder import SyntheticPrototypeAuthorization


def _authorized_report() -> dict[str, object]:
    return {
        "simulation_only": False,
        "overall_decision": "eligible_for_future_model_research",
        "clinical_model_research_authorized": True,
    }


def _training_features() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = 20
    return pd.DataFrame(
        {
            "patient_id": [f"patient-{index}" for index in range(rows)],
            "split": ["train"] * rows,
            "label_state": ["positive", "eligible_negative"] * (rows // 2),
            "hba1c_recent_mean": rng.normal(5.5, 0.5, rows),
            "smoking_status": (["never", "former"] * (rows // 2)),
        }
    )


def test_risk_head_rejects_simulation_only_artifacts() -> None:
    with pytest.raises(RiskHeadAuthorizationError, match="simulation-only"):
        LogisticRiskHead(LogisticRiskHeadConfig()).fit(
            _training_features(),
            {
                "simulation_only": True,
                "overall_decision": "prototype_ready",
                "clinical_model_research_authorized": False,
            },
        )


def test_risk_head_allows_explicitly_approved_synthetic_prototype() -> None:
    features = _training_features().assign(simulation_only=True)
    head = LogisticRiskHead(LogisticRiskHeadConfig())
    fitted = head.fit_synthetic_prototype(
        features, SyntheticPrototypeAuthorization("professor-approval-2026-08-30")
    )
    prediction = fitted.score(features)
    assert prediction.probability.shape == (20,)
    assert (prediction.probability >= 0).all()
    assert (prediction.probability <= 1).all()
    with pytest.raises(ValueError, match="only synthetic rows"):
        head.fit_synthetic_prototype(
            features.assign(simulation_only=False),
            SyntheticPrototypeAuthorization("professor-approval-2026-08-30"),
        )


def test_risk_head_fits_train_rows_and_predicts_probabilities() -> None:
    features = _training_features()
    head = LogisticRiskHead(LogisticRiskHeadConfig()).fit(
        features, _authorized_report()
    )
    probability = head.predict_proba(features)
    assert probability.shape == (20,)
    assert (probability >= 0).all() and (probability <= 1).all()


def test_risk_head_rejects_outcome_columns_and_non_train_rows() -> None:
    with pytest.raises(ValueError, match="prohibited outcome"):
        LogisticRiskHead(LogisticRiskHeadConfig()).fit(
            _training_features().assign(outcome_flag=0), _authorized_report()
        )
    with pytest.raises(ValueError, match="train split"):
        LogisticRiskHead(LogisticRiskHeadConfig()).fit(
            _training_features().assign(split="test"), _authorized_report()
        )


def test_risk_head_rejects_missing_label_state_or_single_class() -> None:
    with pytest.raises(ValueError, match="label_state"):
        LogisticRiskHead(LogisticRiskHeadConfig()).fit(
            _training_features().drop(columns=["label_state"]), _authorized_report()
        )
    with pytest.raises(ValueError, match="both label classes"):
        LogisticRiskHead(LogisticRiskHeadConfig()).fit(
            _training_features().assign(label_state="positive"), _authorized_report()
        )
