"""The days a contract is measured over."""

from dataclasses import dataclass
from datetime import date
from typing import Self

import pandas as pd

from tempmodel.model.validation import check_count


@dataclass(frozen=True)
class DeliveryPeriod:
    """Days from ``start`` to ``end``, both included."""

    start: date
    end: date

    def __post_init__(self) -> None:
        """Check that the period is not empty."""
        if self.start > self.end:
            raise ValueError(f"period ends ({self.end}) before it starts")

    @classmethod
    def month(cls, year: int, month: int) -> Self:
        """Return a calendar month."""
        first = pd.Period(year=year, month=month, freq="M")
        return cls(first.start_time.date(), first.end_time.date())

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Every day of the period."""
        return pd.date_range(self.start, self.end, name="date")

    @property
    def n_days(self) -> int:
        """Number of days."""
        return len(self.dates)


def next_months(after: date, n_months: int = 12) -> list[DeliveryPeriod]:
    """Return the next calendar months that start after a date."""
    check_count("n_months", n_months)
    first = pd.Period(after, freq="M") + 1
    months = (first + i for i in range(n_months))
    return [DeliveryPeriod.month(month.year, month.month) for month in months]
