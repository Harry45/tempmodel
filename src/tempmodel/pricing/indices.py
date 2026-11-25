"""The temperature indices that contracts settle on."""

from typing import Literal

import pandas as pd

Index = Literal["HDD", "CDD", "CAT", "PRIM"]
INDICES: tuple[Index, ...] = ("HDD", "CDD", "CAT", "PRIM")


def realised_indices(
    temperature: pd.Series, *, base_temperature: float = 18.0
) -> pd.DataFrame:
    """Return the indices for each complete calendar month of data.

    HDD and CDD add up the degrees below and above the base temperature, CAT adds up
    the temperature, and PRIM is its average.
    """
    daily = pd.DataFrame(
        {
            "HDD": (base_temperature - temperature).clip(lower=0),
            "CDD": (temperature - base_temperature).clip(lower=0),
            "CAT": temperature,
        }
    )
    month = pd.DatetimeIndex(daily.index).to_period("M").rename("month")
    monthly = daily.groupby(month).sum()
    n_days = daily["CAT"].groupby(month).count()
    monthly["PRIM"] = monthly["CAT"] / n_days
    days_in_month = pd.PeriodIndex(monthly.index).days_in_month
    complete: pd.DataFrame = monthly.loc[n_days.to_numpy() == days_in_month]
    return complete
