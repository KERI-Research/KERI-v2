"""Missingness and observation-density calculations without imputation."""

from __future__ import annotations

from datetime import date

from metaboguard.data.schema import ClinicalEvent
from metaboguard.features.windows import select_preindex_events


def missingness_flags(
    events: list[ClinicalEvent],
    source_feature: str,
    index_date: date,
    birth_date: date,
    recent_days: int = 30,
) -> dict[str, int]:
    lifetime = [
        event
        for event in select_preindex_events(events, index_date, None, birth_date)
        if event.feature_name == source_feature
    ]
    recent = [
        event
        for event in select_preindex_events(events, index_date, recent_days, birth_date)
        if event.feature_name == source_feature
    ]
    return {
        "observed_lifetime": int(any(event.value is not None for event in lifetime)),
        "missing_lifetime": int(not any(event.value is not None for event in lifetime)),
        "observed_recent": int(any(event.value is not None for event in recent)),
        "missing_recent": int(not any(event.value is not None for event in recent)),
    }


def density_features(
    events: list[ClinicalEvent], index_date: date, birth_date: date, window_days: int | None
) -> dict[str, float | int]:
    selected = select_preindex_events(events, index_date, window_days, birth_date)
    dates = {event.event_date for event in selected}
    span_years = (window_days or 365) / 365.25
    return {
        "measurement_days_count": len(dates),
        "measurement_events_count": len(selected),
        "distinct_features_measured_count": len({event.feature_name for event in selected}),
        "measurement_density_per_year": len(dates) / span_years if span_years else 0.0,
    }
