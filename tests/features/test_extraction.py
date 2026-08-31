from dataclasses import replace

import pytest

from metaboguard.features.extraction import extract_features


def test_extraction_is_future_blind(feature_dataset, feature_index) -> None:
    result = extract_features(feature_dataset, [feature_index], {"p1": "train"})
    row = result.rows[0]
    assert row["split"] == "train"
    assert row["glucose__latest_lifetime"] == 130.0
    assert row["glucose__latest_date_offset_days"] is not None
    assert all(item["contains_post_index_record"] is False for item in result.lineage)
    assert all("outcome" not in key.lower() for key in row)


def test_feature_extraction_flushes_bounded_batches(
    feature_dataset, feature_index
) -> None:
    batches = []
    result = extract_features(
        feature_dataset,
        [feature_index],
        {"p1": "train"},
        batch_size=1,
        batch_callback=batches.append,
    )
    assert result.rows == []
    assert len(batches) == 1
    assert batches[0].rows[0]["patient_id"] == "p1"


def test_feature_extraction_flushes_final_partial_batch(
    feature_dataset, feature_index
) -> None:
    batches = []
    result = extract_features(
        feature_dataset,
        [feature_index],
        {"p1": "train"},
        batch_size=2,
        batch_callback=batches.append,
    )
    assert result.rows == []
    assert len(batches) == 1
    assert batches[0].rows[0]["patient_id"] == "p1"


def test_feature_extraction_rejects_nonpositive_batch_size(
    feature_dataset, feature_index
) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        extract_features(
            feature_dataset, [feature_index], {"p1": "train"}, batch_size=0
        )


def test_unknown_registry_family_fails_to_null_without_future_data(
    feature_dataset, feature_index
) -> None:
    result = extract_features(feature_dataset, [feature_index], {"p1": "train"})
    definition = next(iter(result.registry.values()))
    custom = {
        "custom": replace(definition, feature_id="custom", feature_family="unknown")
    }
    result = extract_features(feature_dataset, [feature_index], {"p1": "train"})
    result.registry.update(custom)
    assert result.registry["custom"].feature_family == "unknown"
