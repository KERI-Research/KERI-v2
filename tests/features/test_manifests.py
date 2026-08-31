import json
from pathlib import Path

import pytest

from metaboguard.features.definitions import build_feature_registry
from metaboguard.features.extraction import FeatureDataset, extract_features
from metaboguard.features.manifests import (
    StreamingFeatureArtifactWriter,
    write_feature_artifacts,
)


def test_feature_artifacts_update_parent_manifest(
    feature_dataset, feature_index, tmp_path
) -> None:
    result = extract_features(feature_dataset, [feature_index], {"p1": "train"})
    parent = tmp_path / "cohort"
    parent.mkdir()
    (parent / "cohort_manifest.json").write_text(
        json.dumps(
            {
                "split_status": "created",
                "feature_status": "not_created",
                "model_status": "not_created",
            }
        )
    )
    manifest = write_feature_artifacts(
        result, parent / "features", parent / "cohort_manifest.json"
    )
    assert manifest["feature_status"] == "created"
    assert (
        json.loads((parent / "cohort_manifest.json").read_text())["model_status"]
        == "not_created"
    )


def test_empty_feature_artifacts_and_unvalidated_parent(tmp_path: Path) -> None:
    empty = FeatureDataset([], [], build_feature_registry(), "1.0.0")
    manifest = write_feature_artifacts(empty, tmp_path / "features")
    assert manifest["row_count"] == 0


def test_failed_validation_does_not_update_parent(
    feature_dataset, feature_index, tmp_path: Path
) -> None:
    result = extract_features(feature_dataset, [feature_index], {"p1": "train"})
    failed = FeatureDataset(
        result.rows,
        [{**item, "contains_post_index_record": True} for item in result.lineage],
        result.registry,
        result.feature_definition_version,
    )
    parent = tmp_path / "cohort"
    parent.mkdir()
    path = parent / "cohort_manifest.json"
    path.write_text(json.dumps({"feature_status": "not_created"}))
    write_feature_artifacts(failed, parent / "features", path)
    assert json.loads(path.read_text())["feature_status"] == "not_created"


def test_streaming_feature_artifacts_update_parent_manifest(
    feature_dataset, feature_index, tmp_path: Path
) -> None:
    result = extract_features(feature_dataset, [feature_index], {"p1": "train"})
    parent = tmp_path / "cohort"
    parent.mkdir()
    path = parent / "cohort_manifest.json"
    path.write_text(
        json.dumps({"split_status": "created", "feature_status": "not_created"})
    )
    writer = StreamingFeatureArtifactWriter(parent / "features", path)
    writer.write_batch(result)
    manifest = writer.close()
    assert manifest["row_count"] == 1
    assert (parent / "features" / "feature_lineage.parquet").exists()
    assert json.loads(path.read_text())["feature_status"] == "created"


def test_streaming_feature_artifacts_cover_empty_existing_and_failed_parent(
    tmp_path: Path,
) -> None:
    registry = build_feature_registry()
    integer_feature_id = next(
        feature_id
        for feature_id, definition in registry.items()
        if definition.value_type == "integer"
    )
    registry = {integer_feature_id: registry[integer_feature_id]}
    row = {
        "patient_id": "p1",
        "cohort_class": "ordinary_incidence",
        "endpoint_id": "type2_diabetes",
        "index_date": "2020-01-01",
        "index_sequence_number": 1,
        "feature_definition_version": "1.0.0",
        "feature_window_config_sha256": "windows",
        "source_canonical_manifest_sha256": "canonical",
        "source_cohort_manifest_sha256": "cohort",
        "source_split_manifest_sha256": "split",
        "simulation_only": True,
        integer_feature_id: "1",
    }
    lineage = [
        {
            "patient_id": "p1",
            "endpoint_id": "type2_diabetes",
            "index_date": "2020-01-01",
            "feature_id": integer_feature_id,
            "source_feature_id": registry[integer_feature_id].source_feature_id,
            "window_name": "recent",
            "window_start_date": "2019-01-01",
            "window_end_date": "2020-01-01",
            "source_record_count": "1",
            "earliest_source_date": "2019-01-01",
            "latest_source_date": "2019-01-01",
            "contains_post_index_record": True,
            "source_value_summary_sha256": "summary",
        }
    ]
    empty = FeatureDataset([], [], registry, "1.0.0")
    failed = FeatureDataset([row], lineage, registry, "1.0.0")
    parent = tmp_path / "cohort"
    parent.mkdir()
    path = parent / "cohort_manifest.json"
    path.write_text(json.dumps({"feature_status": "not_created"}), encoding="utf-8")

    writer = StreamingFeatureArtifactWriter(parent / "features", path)
    writer.write_batch(empty)
    writer.write_batch(failed)
    writer.write_batch(failed)
    manifest = writer.close()

    assert manifest["row_count"] == 2
    assert manifest["validation_passed"] is False
    assert (
        json.loads(path.read_text(encoding="utf-8"))["feature_status"] == "not_created"
    )


def test_empty_streaming_writer_close_has_no_parquet_hashes(tmp_path: Path) -> None:
    writer = StreamingFeatureArtifactWriter(tmp_path / "features")

    with pytest.raises(FileNotFoundError):
        writer.close()
