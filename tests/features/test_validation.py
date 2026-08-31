from metaboguard.features.extraction import extract_features
from metaboguard.features.validation import validate_feature_dataset


def test_feature_validation_warnings_are_nonblocking(
    feature_dataset, feature_index
) -> None:
    result = extract_features(feature_dataset, [feature_index], {"p1": "train"})
    report = validate_feature_dataset(result, result.lineage, result.registry, {})
    assert report.passed
    warning = next(check for check in report.checks if check.level == "warning")
    assert warning.status == "warning"
    assert warning.passed is True


def test_future_lineage_fails(feature_dataset, feature_index) -> None:
    result = extract_features(feature_dataset, [feature_index], {"p1": "train"})
    lineage = [{**item, "contains_post_index_record": True} for item in result.lineage]
    report = validate_feature_dataset(result, lineage, result.registry, {})
    assert not report.passed
