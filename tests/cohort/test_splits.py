import json
from datetime import date

from metaboguard.cohort.endpoints import assign_outcomes
from metaboguard.cohort.index_dates import generate_rolling_index_dates
from metaboguard.cohort.protocol import ConstructedCohort, SplitConfig
from metaboguard.cohort.splits import build_splits, write_split_artifacts


def test_patient_level_deterministic_splits(cohort_dataset, diabetes_endpoint, tmp_path) -> None:
    indexes = generate_rolling_index_dates(cohort_dataset, diabetes_endpoint)
    labels = assign_outcomes(indexes, cohort_dataset, diabetes_endpoint)
    cohort = ConstructedCohort("ordinary_incidence", diabetes_endpoint, indexes, [], labels, "hash")
    first = build_splits(cohort, SplitConfig(root_seed=9))
    second = build_splits(cohort, SplitConfig(root_seed=9))
    assert first.assignments == second.assignments
    assert len(first.patient_fingerprints) == len(first.assignments)
    assert all(len(value) == 64 for value in first.patient_fingerprints.values())
    write_split_artifacts(first, labels, tmp_path / "splits")
    assert (tmp_path / "splits" / "split_assignments.parquet").exists()


def test_split_artifacts_update_parent_manifest(
    cohort_dataset, diabetes_endpoint, tmp_path
) -> None:
    indexes = generate_rolling_index_dates(cohort_dataset, diabetes_endpoint)
    labels = assign_outcomes(indexes, cohort_dataset, diabetes_endpoint)
    cohort = ConstructedCohort("ordinary_incidence", diabetes_endpoint, indexes, [], labels, "hash")
    parent = tmp_path / "cohort"
    parent.mkdir()
    (parent / "cohort_manifest.json").write_text(
        json.dumps(
            {
                "split_status": "not_created",
                "feature_status": "not_created",
                "model_status": "not_created",
            }
        )
    )
    write_split_artifacts(build_splits(cohort, SplitConfig(root_seed=9)), labels, parent / "splits")
    manifest = json.loads((parent / "cohort_manifest.json").read_text())
    assert manifest["split_status"] == "created"
    assert manifest["feature_status"] == "not_created"
    assert manifest["model_status"] == "not_created"


def test_fixed_date_over_cap_uses_quantile_fallback(cohort_dataset, diabetes_endpoint) -> None:
    indexes = generate_rolling_index_dates(cohort_dataset, diabetes_endpoint)
    labels = assign_outcomes(indexes, cohort_dataset, diabetes_endpoint)
    cohort = ConstructedCohort("ordinary_incidence", diabetes_endpoint, indexes, [], labels, "hash")
    split = build_splits(cohort, SplitConfig(root_seed=1, fixed_date=date(2015, 1, 1)))
    assert split.quantile_fallback_used
    assert split.temporal_cutoff is not None


def test_empty_cohort_split_is_deterministic(diabetes_endpoint) -> None:
    empty = ConstructedCohort("ordinary_incidence", diabetes_endpoint, [], [], [], "hash")
    split = build_splits(empty, SplitConfig(root_seed=1))
    assert split.assignments == {}
    assert split.temporal_cutoff is None
    fixed_empty = build_splits(empty, SplitConfig(root_seed=1, fixed_date=date(2020, 1, 1)))
    assert fixed_empty.temporal_cutoff == date(2020, 1, 1)


def test_fixed_date_without_fallback_keeps_development_assignments(
    cohort_dataset, diabetes_endpoint
) -> None:
    indexes = generate_rolling_index_dates(cohort_dataset, diabetes_endpoint)
    labels = assign_outcomes(indexes, cohort_dataset, diabetes_endpoint)
    cohort = ConstructedCohort("ordinary_incidence", diabetes_endpoint, indexes, [], labels, "hash")
    split = build_splits(cohort, SplitConfig(root_seed=2, fixed_date=date(2099, 1, 1)))
    assert not split.quantile_fallback_used
    assert "temporal_holdout" not in split.assignments.values()
