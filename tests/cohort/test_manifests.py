import json

from metaboguard.cohort.manifests import construct_endpoint_cohort


def test_constructed_cohort_writes_all_artifacts(
    cohort_dataset, diabetes_endpoint, tmp_path
) -> None:
    cohort = construct_endpoint_cohort(cohort_dataset, diabetes_endpoint, tmp_path / "cohort")
    assert cohort.labels
    for name in (
        "endpoint_protocol.json",
        "eligible_indexes.parquet",
        "excluded_indexes.parquet",
        "horizon_labels_1y.parquet",
        "horizon_labels_3y.parquet",
        "horizon_labels_5y.parquet",
        "cohort_summary.json",
        "washout_impact.json",
        "cohort_validation_report.json",
        "cohort_manifest.json",
    ):
        assert (tmp_path / "cohort" / name).exists()
    manifest = json.loads((tmp_path / "cohort" / "cohort_manifest.json").read_text())
    assert manifest["simulation_only"] is True
    assert manifest["feature_status"] == "not_created"
