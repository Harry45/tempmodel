"""Mean reversion of the deviation X(t) = T(t) - s(t), equation (5)."""

from dataclasses import dataclass
from typing import Self

import numpy as np

from tempmodel.model.calendar import FloatArray


@dataclass(frozen=True)
class MeanReversion:
    """Each day keeps a fraction alpha of the previous day's deviation.

    X(t+1) = alpha X(t) + residual

    Attributes:
        alpha: Fraction of the deviation that survives a day.
    """

    alpha: float

    @classmethod
    def fit(cls, deviation: FloatArray) -> Self:
        """Regress each deviation on the previous day's, without a constant.

        Raises:
            ValueError: If the deviation does not revert to zero.
        """
        yesterday, today = deviation[:-1], deviation[1:]
        alpha = float(yesterday @ today / (yesterday @ yesterday))
        if not 0 < alpha < 1:
            raise ValueError(f"temperature is not mean-reverting: alpha = {alpha:.3f}")
        return cls(alpha)

    @property
    def kappa(self) -> float:
        """Speed of mean reversion in the continuous-time equation (1)."""
        return float(-np.log(self.alpha))

    def residuals(self, deviation: FloatArray) -> FloatArray:
        """Return what yesterday does not explain; unknown (NaN) on the first day."""
        residual = np.full_like(deviation, np.nan)
        residual[1:] = deviation[1:] - self.alpha * deviation[:-1]
        return residual

    def expected_path(self, past: FloatArray, n_days: int) -> FloatArray:
        """Return the expected deviation on each of the next ``n_days`` days.

        Args:
            past: Deviations up to today.
            n_days: Number of days ahead.
        """
        return np.asarray(past[-1] * self.alpha ** np.arange(1, n_days + 1))

    def path_variance(self, daily_variance: FloatArray) -> FloatArray:
        """Return the variance of the deviation on each day ahead.

        Args:
            daily_variance: Variance of the residual on each day ahead.
        """
        total = 0.0
        path = np.empty_like(daily_variance)
        for day, added in enumerate(daily_variance):
            total = self.alpha**2 * total + added
            path[day] = total
        return path

    def simulate(self, past: FloatArray, residual: FloatArray) -> FloatArray:
        """Return the deviations that result from the given residuals.

        Args:
            past: Deviations up to today.
            residual: Residuals ahead, with a row per day and a column per path.
        """
        paths = np.empty_like(residual)
        previous = np.full(residual.shape[1], past[-1])
        for day, shock in enumerate(residual):
            previous = self.alpha * previous + shock
            paths[day] = previous
        return paths
