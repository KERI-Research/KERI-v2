from datetime import date

from metaboguard.cohort.censoring import death_before, follow_up_end_date


def test_follow_up_is_bounded_and_death_rule(cohort_dataset) -> None:
    assert follow_up_end_date(cohort_dataset, "p-positive") == date(2020, 1, 1)
    assert follow_up_end_date(cohort_dataset, "p-competing") == date(2018, 6, 1)
    assert death_before(date(2018, 1, 1), date(2019, 1, 1), True)
    assert not death_before(None, date(2019, 1, 1), True)
    assert not death_before(date(2018, 1, 1), date(2019, 1, 1), False)
