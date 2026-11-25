"""Build, check and save the daily temperature dataset."""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from tempmodel.config import Settings
from tempmodel.data.openmeteo import fetch_daily

logger = logging.getLogger(__name__)

MEAN, MAX, MIN = "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min"


class DataQualityError(ValueError):
    """The data is not what was asked for."""


def local_today(timezone: str) -> date:
    """Return today's date in the given timezone."""
    return datetime.now(ZoneInfo(timezone)).date()


def period_ending_yesterday(years: int, today: date) -> tuple[date, date]:
    """Return the first and last day of a period of whole years up to yesterday."""
    end = today - timedelta(days=1)
    start = pd.Timestamp(end) - pd.DateOffset(years=years) + pd.Timedelta(days=1)
    return start.date(), end


def check_quality(data: pd.DataFrame, start: date, end: date) -> None:
    """Check that every day is present and the values are consistent.

    Missing values are logged rather than rejected, since some models lag behind.

    Raises:
        DataQualityError: If days are missing, repeated or out of order, or if a
            day's minimum, mean and maximum are not in increasing order.
    """
    if not data.index.equals(pd.date_range(start, end)):
        raise DataQualityError(
            f"expected every day from {start} to {end} exactly once, "
            f"got {len(data)} rows"
        )

    incomplete = data.index[data.isna().any(axis=1)]
    if len(incomplete):
        logger.warning(
            "%d days have missing values (first %s, last %s)",
            len(incomplete),
            incomplete[0].date(),
            incomplete[-1].date(),
        )

    for low, high in ((MIN, MEAN), (MEAN, MAX), (MIN, MAX)):
        if low in data and high in data:
            wrong = data.index[data[low] > data[high]]
            if len(wrong):
                raise DataQualityError(
                    f"{low} exceeds {high} on {len(wrong)} days, "
                    f"first {wrong[0].date()}"
                )


def get_daily_temperature(settings: Settings) -> pd.DataFrame:
    """Fetch and check daily temperature (deg C), indexed by local date."""
    today = local_today(settings.location.timezone)
    start, end = period_ending_yesterday(settings.period.years, today)
    data = fetch_daily(settings.api, settings.location, start, end)
    check_quality(data, start, end)
    return data


def load_csv(path: Path) -> pd.DataFrame:
    """Load data saved by ``save_csv``."""
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def save_csv(data: pd.DataFrame, path: Path) -> None:
    """Write the data, replacing any existing file only once it is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        data.to_csv(partial)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
