from datetime import date

from metaboguard.features.windows import select_preindex_events


def test_window_boundaries(feature_dataset) -> None:
    selected = select_preindex_events(
        feature_dataset.events, date(2020, 1, 1), 365, date(1980, 1, 1)
    )
    assert [event.event_date for event in selected] == [date(2019, 6, 1), date(2020, 1, 1)]
    lifetime = select_preindex_events(
        feature_dataset.events, date(2020, 1, 1), None, date(1980, 1, 1)
    )
    assert len(lifetime) == 3
    assert all(event.event_date <= date(2020, 1, 1) for event in lifetime)
