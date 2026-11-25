"""Checks on what the user passes in."""

import numpy as np
import pandas as pd

from tempmodel.model.calendar import DAYS_PER_YEAR

MIN_YEARS = 3


def check_count(name: str, value: int) -> None:
    """Raise ValueError unless ``value`` is a positive whole number."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive whole number, got {value!r}")


def check_level(level: float) -> None:
    """Raise ValueError unless ``level`` is a probability between 0 and 1."""
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level!r}")


def check_temperature(temperature: pd.Series) -> pd.DatetimeIndex:
    """Return the dates, or raise ValueError if the data cannot be fitted."""
    dates = temperature.index
    if not isinstance(dates, pd.DatetimeIndex):
        raise ValueError("temperature must have a DatetimeIndex")
    if len(dates) < MIN_YEARS * DAYS_PER_YEAR:
        raise ValueError(f"at least {MIN_YEARS} years of data are needed")
    if not dates.equals(pd.date_range(dates[0], periods=len(dates))):
        raise ValueError("temperature must have one value for every day, in order")
    if not np.isfinite(temperature.to_numpy(dtype=float)).all():
        raise ValueError("temperature has missing or infinite values")
    return dates
