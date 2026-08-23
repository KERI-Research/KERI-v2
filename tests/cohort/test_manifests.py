import json
from dataclasses import dataclass
from datetime import date

import pandas as pd

from metaboguard.cohort.manifests import _write_rows, construct_endpoint_cohort
from metaboguard.cohort.protocol import PatientIndex


@dataclass(frozen=True)
class ListOnlyRow:
    values: list[str]


def test_constructed_cohort_writes_all_artifacts(
    cohort_dataset, diabetes_endpoint, tmp_path
) -> None:
    cohort = construct_endpoint_cohort(
        cohort_dataset, diabetes_endpoint, tmp_path / "cohort"
    )
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


def test_write_rows_sorts_patient_indexes_with_list_columns(tmp_path) -> None:
    path = tmp_path / "indexes.parquet"
    rows = [
        PatientIndex(
            "p2",
            "ordinary_incidence",
            "type2_diabetes",
            date(2020, 1, 1),
            "rolling",
            1,
            (date(2019, 1, 1), date(2019, 6, 1)),
            (date(2019, 1, 1),),
            2,
        ),
        PatientIndex(
            "p1",
            "ordinary_incidence",
            "type2_diabetes",
            date(2020, 1, 1),
            "rolling",
            1,
            (date(2018, 1, 1), date(2019, 1, 1)),
            (date(2018, 1, 1),),
            2,
        ),
    ]

    _write_rows(rows, path)

    frame = pd.read_parquet(path)
    assert frame["patient_id"].tolist() == ["p1", "p2"]
    assert [list(values) for values in frame["preindex_event_dates"]] == [
        ["2018-01-01", "2019-01-01"],
        ["2019-01-01", "2019-06-01"],
    ]


def test_write_rows_allows_dataclass_rows_without_sortable_columns(tmp_path) -> None:
    path = tmp_path / "list_only.parquet"

    _write_rows([ListOnlyRow(["b"]), ListOnlyRow(["a"])], path)

    frame = pd.read_parquet(path)
    assert [list(values) for values in frame["values"]] == [["b"], ["a"]]
