import pandas as pd

from tempmodel.model.calendar import Clock, is_29_february

DATES = pd.DatetimeIndex(
    [
        "2023-01-01",
        "2023-12-31",
        "2024-02-28",
        "2024-02-29",
        "2024-03-01",
        "2024-12-31",
        "2025-01-01",
    ]
)


def test_29_february_shares_the_time_of_28_february() -> None:
    expected = [0, 364, 423, 423, 424, 729, 730]
    assert Clock(first_year=2023).time(DATES).tolist() == expected


def test_finds_29_february() -> None:
    assert is_29_february(DATES).tolist() == [False] * 3 + [True] + [False] * 3
