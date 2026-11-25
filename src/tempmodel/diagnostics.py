"""Statistics used to check the fitted model."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats


def autocorrelation(values: ArrayLike, max_lag: int) -> NDArray[np.float64]:
    """Return the sample autocorrelation at lags 1 to ``max_lag``."""
    x = np.asarray(values, dtype=float)
    if not 1 <= max_lag < len(x):
        raise ValueError(f"max_lag must be between 1 and {len(x) - 1}, got {max_lag}")
    x = x - x.mean()
    size = 2 ** int(np.ceil(np.log2(2 * len(x))))  # zero-padded, so not circular
    power = np.abs(np.fft.rfft(x, size)) ** 2
    covariance = np.fft.irfft(power, size)[: max_lag + 1]
    return np.asarray(covariance[1:] / covariance[0])


def normality_summary(values: ArrayLike) -> pd.Series:
    """Return moments and the D'Agostino-Pearson normality test.

    Kurtosis is in excess of the normal distribution's.
    """
    x = np.asarray(values, dtype=float)
    return pd.Series(
        {
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)),
            "skewness": float(stats.skew(x)),
            "kurtosis": float(stats.kurtosis(x)),
            "normality_p_value": float(stats.normaltest(x).pvalue),
        }
    )
