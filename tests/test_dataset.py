import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tempmodel.config import load_settings
from tempmodel.data import dataset
from tempmodel.data.dataset import (
    MAX,
    MEAN,
    MIN,
    DataQualityError,
    check_quality,
    get_daily_temperature,
    load_csv,
    period_ending_yesterday,
    save_csv,
)

START, END = date(2026, 1, 1), date(2026, 1, 3)


def make_data() -> pd.DataFrame:
    return pd.DataFrame(
        {MEAN: [5.0, 6.0, 7.0], MAX: [8.0, 9.0, 9.0], MIN: [1.0, 2.0, 3.0]},
        index=pd.date_range(START, END, name="date"),
    )


@pytest.mark.parametrize(
    ("years", "today", "expected"),
    [
        (10, date(2026, 11, 25), (date(2016, 11, 25), date(2026, 11, 24))),
        (1, date(2026, 1, 1), (date(2025, 1, 1), date(2025, 12, 31))),
        (1, date(2024, 3, 1), (date(2023, 3, 1), date(2024, 2, 29))),
    ],
)
def test_period(years: int, today: date, expected: tuple[date, date]) -> None:
    assert period_ending_yesterday(years, today) == expected


def test_good_data_passes_without_warning(caplog: pytest.LogCaptureFixture) -> None:
    check_quality(make_data(), START, END)
    assert not caplog.records


def test_missing_day_is_rejected() -> None:
    with pytest.raises(DataQualityError, match="every day"):
        check_quality(make_data().drop(pd.Timestamp("2026-01-02")), START, END)


def test_repeated_day_is_rejected() -> None:
    data = make_data()
    with pytest.raises(DataQualityError, match="every day"):
        check_quality(pd.concat([data, data.tail(1)]), START, END)


def test_inconsistent_values_are_rejected() -> None:
    data = make_data()
    data[MIN] = [1.0, 99.0, 3.0]
    with pytest.raises(DataQualityError, match="2026-01-02"):
        check_quality(data, START, END)


def test_missing_values_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    data = make_data()
    data[MAX] = [8.0, 9.0, None]
    with caplog.at_level(logging.WARNING):
        check_quality(data, START, END)
    assert "1 days have missing values" in caplog.text


def test_get_daily_temperature_requests_the_right_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[tuple[date, date]] = []

    def fake_fetch(api: object, location: object, start: date, end: date) -> object:
        requested.append((start, end))
        index = pd.date_range(start, end, name="date")
        return pd.DataFrame({MEAN: 1.0, MAX: 2.0, MIN: 0.0}, index=index)

    monkeypatch.setattr(dataset, "local_today", lambda timezone: date(2026, 11, 25))
    monkeypatch.setattr(dataset, "fetch_daily", fake_fetch)
    data = get_daily_temperature(load_settings())
    assert requested == [(date(2016, 11, 25), date(2026, 11, 24))]
    assert len(data) == 3652


def test_saved_csv_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "new" / "data.csv"
    save_csv(make_data(), path)
    loaded = load_csv(path)
    pd.testing.assert_frame_equal(
        loaded, make_data(), check_freq=False, check_index_type=False
    )
    assert list(path.parent.iterdir()) == [path]


def test_failed_save_keeps_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "data.csv"
    path.write_text("old")

    def fail(self: pd.DataFrame, target: Path) -> None:
        target.write_text("half")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError, match="disk full"):
        save_csv(make_data(), path)
    assert path.read_text() == "old"
    assert list(tmp_path.iterdir()) == [path]
