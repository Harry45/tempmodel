"""The paper's clock: days counted in years of exactly 365 days."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

DAYS_PER_YEAR = 365
DAYS_BEFORE_29_FEBRUARY = 59

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def is_29_february(dates: pd.DatetimeIndex) -> NDArray[np.bool_]:
    """Return which dates are 29 February."""
    return np.asarray((dates.month == 2) & (dates.day == 29))


@dataclass(frozen=True)
class Clock:
    """Converts dates to the paper's time t.

    t counts days from 1 January of ``first_year``, so ``t % 365`` is the day of
    the year, starting from 0. 29 February does not exist on this clock and
    shares the time of 28 February.
    """

    first_year: int

    def time(self, dates: pd.DatetimeIndex) -> IntArray:
        """Return t for each date."""
        day = dates.day_of_year.to_numpy() - 1
        skipped = dates.is_leap_year & (day >= DAYS_BEFORE_29_FEBRUARY)
        years = dates.year.to_numpy() - self.first_year
        return np.asarray(years * DAYS_PER_YEAR + day - skipped, dtype=np.int64)
