"""The paper's Stockholm model, written independently of the package."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Section 3 and Table 1.
A, B, B1, G1, ALPHA = 6.38, 0.0001, 10.44, -154.76, 0.823
TABLE_1 = [6.12, 1.04, -0.21, 0.66, -0.02, 2.00, 1.19, 0.27, -0.08]


def daily_variance(t: NDArray[np.int64]) -> NDArray[np.float64]:
    angle = 2 * np.pi * np.outer(t, np.arange(1, 5)) / 365
    sigma2 = TABLE_1[0] + np.sin(angle) @ TABLE_1[1:5] + np.cos(angle) @ TABLE_1[5:]
    return np.asarray(ALPHA**2 * sigma2)


def simulate(start: str = "1961-01-01", end: str = "2006-05-25") -> pd.Series:
    """Simulate temperature; ``start`` must be 1 January for t to begin at 0."""
    dates = pd.date_range(start, end, name="date")
    assert dates[0].day_of_year == 1
    without_29_february = dates[~((dates.month == 2) & (dates.day == 29))]
    t = np.arange(len(without_29_february))
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(len(t)) * np.sqrt(daily_variance(t))
    deviation = np.zeros(len(t))
    for i in range(1, len(t)):
        deviation[i] = ALPHA * deviation[i - 1] + noise[i - 1]
    mean = A + B * t + B1 * np.cos(2 * np.pi * (t - G1) / 365)
    series = pd.Series(mean + deviation, index=without_29_february)
    return series.reindex(dates).ffill()
