import json
from pathlib import Path

from metaboguard.features.definitions import build_feature_registry
from metaboguard.features.extraction import FeatureDataset, extract_features
from metaboguard.features.manifests import StreamingFeatureArtifactWriter, write_feature_artifacts


def test_feature_artifacts_update_parent_manifest(feature_dataset, feature_index, tmp_path) -> None:
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
    manifest = write_feature_artifacts(result, parent / "features", parent / "cohort_manifest.json")
    assert manifest["feature_status"] == "created"
    assert (
        json.loads((parent / "cohort_manifest.json").read_text())["model_status"] == "not_created"
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
    path.write_text(json.dumps({"split_status": "created", "feature_status": "not_created"}))
    writer = StreamingFeatureArtifactWriter(parent / "features", path)
    writer.write_batch(result)
    manifest = writer.close()
    assert manifest["row_count"] == 1
    assert (parent / "features" / "feature_lineage.parquet").exists()
    assert json.loads(path.read_text())["feature_status"] == "created"
