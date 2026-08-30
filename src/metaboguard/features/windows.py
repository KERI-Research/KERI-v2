"""Pre-index window selection with explicit date-boundary semantics."""

from __future__ import annotations

from datetime import date, timedelta

from metaboguard.data.schema import ClinicalEvent


def window_bounds(index_date: date, window_days: int | None, birth_date: date) -> tuple[date, date]:
    """Return `(start, end]` bounds, or lifetime `[birth, index]` bounds."""
    return (
        (birth_date, index_date)
        if window_days is None
        else (index_date - timedelta(days=window_days), index_date)
    )


def select_preindex_events(
    events: list[ClinicalEvent],
    index_date: date,
    window_days: int | None,
    birth_date: date,
) -> list[ClinicalEvent]:
    """Select only events satisfying the documented `(start, index]` rule."""
    start, end = window_bounds(index_date, window_days, birth_date)
    selected = [
        event
        for event in events
        if (start < event.event_date if window_days is not None else start <= event.event_date)
        and event.event_date <= end
    ]
    return sorted(
        selected,
        key=lambda event: (event.event_date, event.feature_name, event.patient_id),
    )
