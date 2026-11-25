"""Everything learned from fitting the model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tempmodel.model.calendar import Clock, FloatArray
from tempmodel.model.mean_reversion import MeanReversion
from tempmodel.model.seasonal_mean import SeasonalMean
from tempmodel.model.seasonal_variance import SeasonalVariance


@dataclass(frozen=True, eq=False)
class FitResult:
    """The fitted parts of the model and the data they were fitted to.

    Attributes:
        temperature: The data, including any 29 February.
        clock: Converts dates to the model's time.
        mean: The seasonal mean.
        reversion: The mean reversion.
        variance: The seasonal variance.
        history: For each day used in fitting (without 29 February):
            temperature, seasonal_mean, deviation, residual, volatility and
            standardised_residual.
    """

    temperature: pd.Series
    clock: Clock
    mean: SeasonalMean
    reversion: MeanReversion
    variance: SeasonalVariance
    history: pd.DataFrame

    @property
    def residual_std(self) -> float:
        """Standard deviation of the residuals over the whole year."""
        return float(self.history["residual"].std())

    @property
    def last_date(self) -> pd.Timestamp:
        """The last day of data."""
        return pd.Timestamp(self.temperature.index[-1])

    def seasonal_mean(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Return the seasonal mean on the given dates."""
        mean = self.mean(self.clock.time(dates))
        return pd.Series(mean, index=dates, name="seasonal_mean")

    def past_deviation(self) -> FloatArray:
        """Return the deviation on every day of data, including 29 February."""
        dates = pd.DatetimeIndex(self.temperature.index)
        mean = self.mean(self.clock.time(dates))
        return np.asarray(self.temperature.to_numpy(dtype=float) - mean)

    def parameters(self) -> pd.Series:
        """Return the parameters under the paper's names.

        As in Table 1, c, c_i and d_j describe the daily variance / alpha^2.
        """
        alpha = self.reversion.alpha
        waves = enumerate(
            zip(self.mean.amplitudes, self.mean.phases, strict=True), start=1
        )
        values = {"a": self.mean.level, "b": self.mean.trend}
        for j, (amplitude, phase) in waves:
            values |= {f"b{j}": float(amplitude), f"g{j}": float(phase)}
        values |= {
            "alpha": alpha,
            "kappa": self.reversion.kappa,
            "residual_std": self.residual_std,
            "c": self.variance.constant / alpha**2,
        }
        for letter, weights in (("c", self.variance.sine), ("d", self.variance.cosine)):
            for i, weight in enumerate(weights, start=1):
                values[f"{letter}{i}"] = float(weight) / alpha**2
        return pd.Series(values, name="value")
