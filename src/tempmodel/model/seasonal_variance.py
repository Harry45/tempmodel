"""The seasonal variance of the residuals, equation (3)."""

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd

from tempmodel.model.calendar import DAYS_PER_YEAR, FloatArray, IntArray
from tempmodel.model.fourier import least_squares, sine_and_cosine


@dataclass(frozen=True, eq=False)
class SeasonalVariance:
    """The daily variance, a sum of yearly waves.

    daily_variance(t) = constant + sum_i sine_i sin(2 pi i t / 365)
                                 + sum_j cosine_j cos(2 pi j t / 365)

    This is the paper's sigma-tilde squared. Table 1 reports it divided by alpha^2.

    Attributes:
        constant: Average over the year.
        sine: Weight on the sine of each harmonic.
        cosine: Weight on the cosine of each harmonic.
        empirical: Variance of the residuals for each day of the year, which the
            waves are fitted to.
    """

    constant: float
    sine: FloatArray
    cosine: FloatArray
    empirical: pd.Series

    @classmethod
    def fit(cls, t: IntArray, residual: FloatArray, *, harmonics: int) -> Self:
        """Fit by least squares to the variance on each day of the year.

        Args:
            t: Time of the volatility behind each residual. By equation (5), that
                is the day before the residual is seen.
            residual: Residuals; missing values are ignored.
            harmonics: Number of waves.

        Raises:
            ValueError: If the fitted variance is not positive all year.
        """
        day_of_year = pd.Index(t % DAYS_PER_YEAR, name="day_of_year")
        empirical = pd.Series(residual, name="variance").groupby(day_of_year).var()

        days = empirical.index.to_numpy(dtype=np.int64)
        sine, cosine = sine_and_cosine(days, harmonics)
        weights = least_squares(
            [np.ones(len(days)), sine, cosine], np.asarray(empirical, dtype=np.float64)
        )
        fitted = cls(
            constant=float(weights[0]),
            sine=weights[1 : 1 + harmonics],
            cosine=weights[1 + harmonics :],
            empirical=empirical,
        )
        if fitted(np.arange(DAYS_PER_YEAR)).min() <= 0:
            raise ValueError(
                "the fitted variance is not positive all year; use fewer harmonics"
            )
        return fitted

    def __call__(self, t: IntArray) -> FloatArray:
        """Return the daily variance at time t."""
        sine, cosine = sine_and_cosine(t, len(self.sine))
        return np.asarray(self.constant + sine @ self.sine + cosine @ self.cosine)
