"""Check the forecasts against what happened."""

import copy

import pandas as pd

from tempmodel.model import TemperatureModel
from tempmodel.model.validation import (
    MIN_YEARS,
    check_count,
    check_level,
    check_temperature,
)


def backtest(
    temperature: pd.Series,
    model: TemperatureModel,
    *,
    n_days: int = 365,
    level: float = 0.95,
    min_years: int = 5,
) -> pd.DataFrame:
    """Forecast from the first day of each month, using only earlier data.

    Only months with ``n_days`` of data after them are used. Forecasts from nearby
    months overlap, so they are not independent.

    Args:
        temperature: Consecutive daily values.
        model: The model to test. It is not changed.
        n_days: Number of days to forecast each time.
        level: Probability covered by the interval.
        min_years: Years of data before the first forecast.

    Returns:
        One row per forecast and day, indexed by ``start`` (first day forecast) and
        ``days_ahead`` (from 1), with columns date, mean, std, lower, upper and
        actual.

    Raises:
        ValueError: If the data is too short for a single forecast.
    """
    check_count("n_days", n_days)
    check_count("min_years", min_years)
    check_level(level)
    if min_years < MIN_YEARS:
        raise ValueError(f"min_years must be at least {MIN_YEARS}, got {min_years}")
    dates = check_temperature(temperature)

    first = dates[0] + pd.DateOffset(years=min_years)
    last = dates[-1] - pd.Timedelta(days=n_days - 1)
    starts = pd.date_range(first, last, freq="MS", name="start")
    if starts.empty:
        raise ValueError("not enough data for min_years of history and n_days after")

    forecasts = {
        start: _one_forecast(temperature, model, start, n_days, level)
        for start in starts
    }
    return pd.concat(forecasts, names=["start", "days_ahead"])


def summarise(results: pd.DataFrame) -> pd.Series:
    """Return the accuracy of the forecasts from ``backtest``.

    bias is positive if actual temperatures were warmer than forecast. coverage is
    the fraction of days inside the interval.
    """
    error = results["actual"] - results["mean"]
    inside = results["actual"].between(results["lower"], results["upper"])
    one_day_ahead = error.xs(1, level="days_ahead")
    return pd.Series(
        {
            "forecasts": len(one_day_ahead),
            "coverage": inside.mean(),
            "bias": error.mean(),
            "rmse": _rmse(error),
            "rmse_one_day_ahead": _rmse(one_day_ahead),
        },
        name="value",
    )


def _one_forecast(
    temperature: pd.Series,
    model: TemperatureModel,
    start: pd.Timestamp,
    n_days: int,
    level: float,
) -> pd.DataFrame:
    history = temperature.loc[: start - pd.Timedelta(days=1)]
    forecast = copy.copy(model).fit(history).predict(n_days, level=level)
    forecast["actual"] = temperature.reindex(forecast.index)
    forecast = forecast.reset_index()
    forecast.index = pd.RangeIndex(1, n_days + 1, name="days_ahead")
    return forecast


def _rmse(error: pd.Series) -> float:
    return float((error**2).mean() ** 0.5)
