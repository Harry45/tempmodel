"""Prices of futures and options on temperature."""

from collections.abc import Sequence
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from tempmodel.model import FitResult
from tempmodel.model.calendar import FloatArray
from tempmodel.pricing.indices import INDICES, Index
from tempmodel.pricing.period import DeliveryPeriod

ONE_DAY = pd.Timedelta(days=1)
DAYS_IN_RATE_YEAR = 365

# Grid over the state of the weather at exercise, for options without a formula.
GRID = np.linspace(-8, 8, 2001)
GRID_WEIGHTS = stats.norm.pdf(GRID) / stats.norm.pdf(GRID).sum()


class Pricer:
    """Prices contracts as of the last day of data.

    Under the pricing probability, each day's deviation from the seasonal mean is
    pushed by the market price of risk, theta, as in equation (14). Prices are then
    expected values, in index points: multiply by the contract's value per point.

    Args:
        result: The fitted model.
        base_temperature: Temperature that HDD and CDD are measured from, in °C.
        market_price_of_risk: Theta, in °C per day. Either a number, or a Series with
            a value for every day from the last day of data onwards.
        interest_rate: Continuously compounded rate per year, for options.
        constant_volatility: Whether to ignore the seasons in the volatility, and use
            the standard deviation of the residuals all year.
    """

    def __init__(
        self,
        result: FitResult,
        *,
        base_temperature: float = 18.0,
        market_price_of_risk: float | pd.Series = 0.0,
        interest_rate: float = 0.0,
        constant_volatility: bool = False,
    ) -> None:
        self.result = result
        self.base_temperature = base_temperature
        self.market_price_of_risk = market_price_of_risk
        self.interest_rate = interest_rate
        self.constant_volatility = constant_volatility

    @property
    def valuation_date(self) -> pd.Timestamp:
        """The date of the prices: the last day of data."""
        return self.result.last_date

    def futures_price(self, index: Index, period: DeliveryPeriod) -> float:
        """Return the futures price: Propositions 4.1 and 4.4, and equation (10)."""
        self._check_period(period)
        today = np.array([self.result.past_deviation()[-1]])
        return float(self._price(index, period, self.valuation_date, today)[0])

    def futures_curve(self, periods: Sequence[DeliveryPeriod]) -> pd.DataFrame:
        """Return the futures price of every index for each period."""
        return pd.DataFrame(
            {
                index: [self.futures_price(index, period) for period in periods]
                for index in INDICES
            },
            index=pd.DatetimeIndex([p.start for p in periods], name="start"),
        )

    def cat_volatility(self, period: DeliveryPeriod) -> pd.Series:
        """Return the daily volatility of the CAT futures price, equation (25).

        The value on a date is the standard deviation of the price change over the
        following day. Dates run from the valuation date to the day before delivery.
        """
        self._check_period(period)
        dates = pd.date_range(
            self.valuation_date, pd.Timestamp(period.start) - ONE_DAY, name="date"
        )
        alpha = self.result.reversion.alpha
        # A shock on the next day is still felt, shrunk by alpha each day, throughout
        # the delivery period.
        days_to_start = (pd.Timestamp(period.start) - dates).days.to_numpy()
        felt = alpha ** (days_to_start - 1) * (1 - alpha**period.n_days) / (1 - alpha)
        volatility = np.sqrt(self._residual_variance(dates)) * felt
        return pd.Series(volatility, index=dates, name="volatility")

    def option_price(
        self,
        index: Index,
        kind: Literal["call", "put"],
        period: DeliveryPeriod,
        *,
        strike: float,
        exercise: date,
    ) -> float:
        """Return the price of a European option on a futures contract.

        CAT and PRIM use the formula in Proposition 4.5. HDD and CDD have none, so
        their payoff is averaged over the state of the weather at exercise. (The
        paper suggests simulating it.)

        Args:
            index: Index of the futures contract.
            kind: "call" or "put".
            period: Delivery period of the futures contract.
            strike: Exercise price.
            exercise: Exercise date, after the valuation date and before delivery.
        """
        if kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
        futures_price = self.futures_price(index, period)
        expiry = pd.Timestamp(exercise)
        if not self.valuation_date < expiry < pd.Timestamp(period.start):
            raise ValueError(
                "exercise must be after the valuation date, before delivery"
            )

        if index in ("CAT", "PRIM"):
            days = 1 if index == "CAT" else period.n_days
            variance = self.cat_volatility(period).loc[: expiry - ONE_DAY] ** 2
            std = float(np.sqrt(variance.sum())) / days
            call = _normal_call(futures_price, strike, std)
        else:
            payoff = self._price_at(index, period, expiry) - strike
            call = float(GRID_WEIGHTS @ np.maximum(payoff, 0))

        # Put-call parity, before discounting.
        value = call if kind == "call" else call - (futures_price - strike)
        years = (expiry - self.valuation_date).days / DAYS_IN_RATE_YEAR
        return float(np.exp(-self.interest_rate * years) * value)

    def _price_at(
        self, index: Index, period: DeliveryPeriod, when: pd.Timestamp
    ) -> FloatArray:
        """Return the futures price at a later date, for each weather in GRID."""
        dates = pd.date_range(self.valuation_date + ONE_DAY, when)
        today = np.array([self.result.past_deviation()[-1]])
        mean, std = self._deviation_ahead(dates, today)
        deviation = mean[0, -1] + std[-1] * GRID
        return self._price(index, period, when, deviation)

    def _price(
        self,
        index: Index,
        period: DeliveryPeriod,
        when: pd.Timestamp,
        deviation: FloatArray,
    ) -> FloatArray:
        """Return the futures price at ``when``, for each deviation on that date."""
        dates = pd.date_range(when + ONE_DAY, period.end)
        mean, std = self._deviation_ahead(dates, deviation)
        mean = mean + self.result.seasonal_mean(dates).to_numpy()
        # Only the days of the delivery period count.
        mean, std = mean[:, -period.n_days :], std[-period.n_days :]

        if index == "CAT":
            daily = mean
        elif index == "PRIM":
            daily = mean / period.n_days
        elif index == "HDD":
            daily = _expected_excess(self.base_temperature - mean, std)
        elif index == "CDD":
            daily = _expected_excess(mean - self.base_temperature, std)
        else:
            raise ValueError(f"index must be one of {INDICES}, got {index!r}")
        return np.asarray(daily.sum(axis=1))

    def _deviation_ahead(
        self, dates: pd.DatetimeIndex, deviation: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Return the mean and std of the deviation, equations (16) and (17).

        Args:
            dates: Consecutive dates, starting the day after the deviation.
            deviation: Possible deviations, one per row of the mean.
        """
        reversion = self.result.reversion
        before = dates - ONE_DAY
        decay = reversion.alpha ** np.arange(1, len(dates) + 1)

        # Theta acts through each day while decaying, which adds this each day.
        push = self._theta(before) * (1 - reversion.alpha) / reversion.kappa
        pushed = reversion.simulate(np.zeros(1), push[:, np.newaxis])[:, 0]

        mean = np.outer(deviation, decay) + pushed
        std = np.sqrt(reversion.path_variance(self._residual_variance(before)))
        return mean, std

    def _residual_variance(self, dates: pd.DatetimeIndex) -> FloatArray:
        """Return the variance of the residual seen the day after each date."""
        if self.constant_volatility:
            return np.full(len(dates), self.result.residual_std**2)
        return self.result.variance(self.result.clock.time(dates))

    def _theta(self, dates: pd.DatetimeIndex) -> FloatArray:
        if not isinstance(self.market_price_of_risk, pd.Series):
            return np.full(len(dates), float(self.market_price_of_risk))
        theta = self.market_price_of_risk.reindex(dates)
        if theta.isna().any():
            raise ValueError(
                f"market_price_of_risk must cover every day from {dates[0].date()} "
                f"to {dates[-1].date()}"
            )
        return theta.to_numpy(dtype=np.float64)

    def _check_period(self, period: DeliveryPeriod) -> None:
        if pd.Timestamp(period.start) <= self.valuation_date:
            raise ValueError(
                f"delivery must start after the valuation date, "
                f"{self.valuation_date.date()}"
            )


def _expected_excess(mean: FloatArray, std: FloatArray) -> FloatArray:
    """Return the expected value of max(x, 0) for normal x."""
    d = mean / std
    return np.asarray(std * (d * stats.norm.cdf(d) + stats.norm.pdf(d)))


def _normal_call(futures_price: float, strike: float, std: float) -> float:
    """Return the undiscounted call price in Proposition 4.5."""
    mean = np.array([futures_price - strike])
    return float(_expected_excess(mean, np.array([std]))[0])
