"""The complete model: fit it, then predict or sample."""

import logging
from typing import Self

import numpy as np
import pandas as pd
from scipy import stats

from tempmodel.model.calendar import Clock, FloatArray, is_29_february
from tempmodel.model.fit_result import FitResult
from tempmodel.model.mean_reversion import MeanReversion
from tempmodel.model.seasonal_mean import SeasonalMean
from tempmodel.model.seasonal_variance import SeasonalVariance
from tempmodel.model.validation import check_count, check_level, check_temperature

logger = logging.getLogger(__name__)

ONE_DAY = pd.Timedelta(days=1)


class NotFittedError(RuntimeError):
    """The model was used before calling ``fit``."""


class TemperatureModel:
    """Seasonal, mean-reverting temperature with seasonal variance.

    Args:
        seasonal_harmonics: Number of waves in the seasonal mean (the paper's J1).
        variance_harmonics: Number of waves in the variance (I2 = J2).
        trend: Whether the mean includes a linear trend.
    """

    def __init__(
        self,
        *,
        seasonal_harmonics: int = 1,
        variance_harmonics: int = 4,
        trend: bool = True,
    ) -> None:
        self.seasonal_harmonics = seasonal_harmonics
        self.variance_harmonics = variance_harmonics
        self.trend = trend
        self._result: FitResult | None = None

    def fit(self, temperature: pd.Series) -> Self:
        """Fit the model to daily average temperature, as in section 3.

        Args:
            temperature: Consecutive daily values with a DatetimeIndex, at least
                three years long.

        Raises:
            ValueError: If the data or the resulting fit is not usable.
        """
        check_count("seasonal_harmonics", self.seasonal_harmonics)
        check_count("variance_harmonics", self.variance_harmonics)
        dates = check_temperature(temperature)

        # The paper removes 29 February, so that all years have the same length.
        clock = Clock(first_year=int(dates[0].year))
        series = temperature[~is_29_february(dates)]
        t = clock.time(pd.DatetimeIndex(series.index))
        values = np.asarray(series, dtype=np.float64)

        # Step 1, equation (2): the trend and seasonal mean.
        mean = SeasonalMean.fit(
            t, values, harmonics=self.seasonal_harmonics, trend=self.trend
        )
        deviation = values - mean(t)

        # Step 2, equation (5): how fast deviations from the mean die out.
        reversion = MeanReversion.fit(deviation)
        residual = reversion.residuals(deviation)

        # Step 3, equation (3): how large the residuals are over the year. The
        # residual seen on a day has the volatility of the day before.
        t_before = t - 1
        variance = SeasonalVariance.fit(
            t_before, residual, harmonics=self.variance_harmonics
        )
        volatility = np.sqrt(variance(t_before))

        history = pd.DataFrame(
            {
                "temperature": values,
                "seasonal_mean": mean(t),
                "deviation": deviation,
                "residual": residual,
                "volatility": volatility,
                "standardised_residual": residual / volatility,
            },
            index=series.index.rename("date"),
        )
        self._result = FitResult(
            temperature.copy(), clock, mean, reversion, variance, history
        )
        logger.info(
            "Fitted %d days: alpha %.3f, kappa %.3f, residual std %.2f",
            len(history),
            reversion.alpha,
            reversion.kappa,
            self._result.residual_std,
        )
        return self

    @property
    def result_(self) -> FitResult:
        """The result of the last successful fit."""
        if self._result is None:
            raise NotFittedError("call fit before using the model")
        return self._result

    def predict(self, n_days: int, *, level: float = 0.95) -> pd.DataFrame:
        """Forecast the distribution of temperature for the next ``n_days`` days.

        The interval is where each day's temperature should fall with probability
        ``level``. It does not include uncertainty in the fitted parameters.

        Args:
            n_days: Number of calendar days after the last day of data.
            level: Probability covered by the interval.

        Returns:
            Columns mean, std, lower and upper, indexed by date.
        """
        result = self.result_
        check_level(level)
        dates, daily_variance = self._days_ahead(n_days)

        expected = result.reversion.expected_path(result.past_deviation(), n_days)
        mean = result.seasonal_mean(dates).to_numpy() + expected
        std = np.sqrt(result.reversion.path_variance(daily_variance))
        half_width = float(stats.norm.ppf((1 + level) / 2)) * std
        return pd.DataFrame(
            {
                "mean": mean,
                "std": std,
                "lower": mean - half_width,
                "upper": mean + half_width,
            },
            index=dates,
        )

    def sample(
        self,
        n_days: int,
        n_samples: int = 1000,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Simulate temperature paths for the next ``n_days`` days.

        Args:
            n_days: Number of calendar days after the last day of data.
            n_samples: Number of paths.
            random_state: Seed or generator, for reproducible paths.

        Returns:
            One column per path, indexed by date.
        """
        result = self.result_
        check_count("n_samples", n_samples)
        dates, daily_variance = self._days_ahead(n_days)

        rng = np.random.default_rng(random_state)
        noise = rng.standard_normal((n_days, n_samples))
        residual = noise * np.sqrt(daily_variance)[:, np.newaxis]
        deviation = result.reversion.simulate(result.past_deviation(), residual)

        mean = result.seasonal_mean(dates).to_numpy()[:, np.newaxis]
        columns = pd.RangeIndex(n_samples, name="sample")
        return pd.DataFrame(mean + deviation, index=dates, columns=columns)

    def _days_ahead(self, n_days: int) -> tuple[pd.DatetimeIndex, FloatArray]:
        """Return the next dates and the variance of the residual on each."""
        check_count("n_days", n_days)
        result = self.result_
        dates = pd.date_range(result.last_date + ONE_DAY, periods=n_days, name="date")
        return dates, result.variance(result.clock.time(dates - ONE_DAY))
