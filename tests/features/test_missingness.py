from datetime import date

from metaboguard.features.missingness import density_features, missingness_flags


def test_missingness_and_density(feature_dataset) -> None:
    flags = missingness_flags(
        feature_dataset.events, "glucose", date(2020, 1, 1), date(1980, 1, 1)
    )
    assert flags["observed_lifetime"] == 1
    assert flags["missing_lifetime"] == 0
    density = density_features(
        feature_dataset.events, date(2020, 1, 1), date(1980, 1, 1), 365
    )
    assert density["measurement_events_count"] == 2
