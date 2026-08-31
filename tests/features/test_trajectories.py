from datetime import date

from metaboguard.features.trajectories import change, summaries, trajectory


def test_summary_change_and_trajectory() -> None:
    values = [
        (date(2020, 1, 1), 1.0),
        (date(2020, 2, 1), 2.0),
        (date(2020, 3, 15), 4.0),
    ]
    summary = summaries(values)
    assert summary["mean"] == 7 / 3
    assert summary["std"] is not None
    assert change(values)["absolute_change"] == 3.0
    assert (
        change([(date(2020, 1, 1), 0.0), (date(2020, 2, 1), 1.0)])["relative_change"]
        is None
    )
    result = trajectory(values)
    assert result["trend_direction"] == "increasing"
    assert trajectory(values[:2])["trend_direction"] == "unknown"
