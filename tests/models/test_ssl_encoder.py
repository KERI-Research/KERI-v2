from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from metaboguard.models.prototype_training import train_synthetic_prototype
from metaboguard.models.ssl_encoder import (
    ResearchModelAuthorizationError,
    RobustPCAEncoder,
    RobustPCAEncoderConfig,
    SyntheticPrototypeAuthorization,
)


def _authorized_report() -> dict[str, object]:
    return {
        "simulation_only": False,
        "overall_decision": "eligible_for_future_model_research",
        "clinical_model_research_authorized": True,
    }


def _training_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"patient-{index}" for index in range(6)],
            "split": ["train"] * 6,
            "hba1c_recent_mean": [5.1, 5.4, 5.5, 5.7, 6.0, 6.4],
            "weight_change_annual": [-1.0, 0.0, 1.2, -0.5, 2.0, 1.5],
            "smoking_status": [
                "never",
                "former",
                "never",
                pd.NA,
                "former",
                "never",
            ],
        }
    )


def test_encoder_rejects_simulation_only_artifacts() -> None:
    with pytest.raises(ResearchModelAuthorizationError, match="simulation-only"):
        RobustPCAEncoder(RobustPCAEncoderConfig(latent_dimensions=2)).fit(
            _training_features(),
            {
                "simulation_only": True,
                "overall_decision": "prototype_ready",
                "clinical_model_research_authorized": False,
            },
        )


def test_encoder_allows_explicitly_approved_synthetic_prototype() -> None:
    features = _training_features().assign(simulation_only=True)
    encoder = RobustPCAEncoder(RobustPCAEncoderConfig(latent_dimensions=2))
    fitted = encoder.fit_synthetic_prototype(
        features, SyntheticPrototypeAuthorization("professor-approval-2026-08-30")
    )
    assert fitted.score(features).embedding.shape == (6, 2)
    with pytest.raises(ValueError, match="only synthetic rows"):
        encoder.fit_synthetic_prototype(
            features.assign(simulation_only=False),
            SyntheticPrototypeAuthorization("professor-approval-2026-08-30"),
        )


def test_encoder_fits_train_rows_and_returns_label_free_deviation() -> None:
    features = _training_features().assign(unavailable_measurement=None)
    encoder = RobustPCAEncoder(RobustPCAEncoderConfig(latent_dimensions=2)).fit(
        features, _authorized_report()
    )
    scored = encoder.score(features)
    assert scored.embedding.shape == (6, 2)
    assert scored.reconstruction_error.shape == (6,)
    assert scored.latent_distance.shape == (6,)
    assert scored.metabolic_deviation_score.shape == (6,)
    assert scored.reference_percentile.min() >= 0
    assert scored.reference_percentile.max() <= 100


def test_encoder_rejects_outcome_columns_and_non_train_rows() -> None:
    with pytest.raises(ValueError, match="prohibited outcome"):
        RobustPCAEncoder(RobustPCAEncoderConfig(latent_dimensions=2)).fit(
            _training_features().assign(outcome_label=0), _authorized_report()
        )
    with pytest.raises(ValueError, match="train split"):
        RobustPCAEncoder(RobustPCAEncoderConfig(latent_dimensions=2)).fit(
            _training_features().assign(split="test"), _authorized_report()
        )


def test_synthetic_prototype_writes_non_diagnostic_model_card(tmp_path: Path) -> None:
    run = tmp_path / "run"
    features = run / "features" / "pancreatic_cancer"
    features.mkdir(parents=True)
    (run / "manifest.json").write_text(
        '{"run_id":"synthetic-run","simulation_only":true,"status":"completed"}',
        encoding="utf-8",
    )
    (run / "generation_manifest.json").write_text(
        '{"state":"complete","synthea_version":"4.0.0","jar_sha256":"x",'
        '"canonical_dataset_sha256":"y","simulation_only":true}',
        encoding="utf-8",
    )
    _training_features().assign(simulation_only=True).to_parquet(
        features / "feature_matrix.parquet", index=False
    )
    (features / "feature_manifest.json").write_text(
        '{"simulation_only":true,"validation_passed":true}', encoding="utf-8"
    )
    summary = train_synthetic_prototype(
        run,
        "pancreatic_cancer",
        "professor-approval-2026-08-30",
        tmp_path / "artifacts",
        RobustPCAEncoderConfig(latent_dimensions=2),
    )
    card = (
        tmp_path / "artifacts" / "synthetic-run" / "pancreatic_cancer" / "MODEL_CARD.md"
    ).read_text()
    assert summary["clinical_use_prohibited"] is True
    assert summary["source_provenance_discrepancy"] is False
    assert "must not be used for diagnosis" in card
    assert "synthetic Synthea-derived dataset" in card
