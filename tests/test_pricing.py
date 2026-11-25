from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tempmodel.model import FitResult, TemperatureModel
from tempmodel.pricing import DeliveryPeriod, Pricer, next_months, realised_indices
from tempmodel.pricing.indices import INDICES, Index

# The simulated data ends on 25 May 2006.
JUNE = DeliveryPeriod.month(2006, 6)
OCTOBER = DeliveryPeriod.month(2006, 10)
EXERCISE = date(2006, 9, 20)


@pytest.fixture(scope="module")
def pricer(result: FitResult) -> Pricer:
    return Pricer(result)


@pytest.fixture(scope="module")
def samples(model: TemperatureModel) -> pd.DataFrame:
    return model.sample(200, 40000, random_state=0)


def index_of(samples: pd.DataFrame, index: Index, period: DeliveryPeriod) -> pd.Series:
    """The index for each path, from its definition."""
    days = samples.loc[period.dates]
    if index == "HDD":
        return (18 - days).clip(lower=0).sum()
    if index == "CDD":
        return (days - 18).clip(lower=0).sum()
    return days.sum() if index == "CAT" else days.mean()


def test_month_has_all_its_days() -> None:
    february = DeliveryPeriod.month(2028, 2)
    assert (february.start, february.end) == (date(2028, 2, 1), date(2028, 2, 29))
    assert february.n_days == 29


def test_empty_period_is_rejected() -> None:
    with pytest.raises(ValueError, match="before it starts"):
        DeliveryPeriod(date(2027, 2, 1), date(2027, 1, 1))


@pytest.mark.parametrize("after", [date(2026, 11, 1), date(2026, 11, 30)])
def test_next_months_start_after_the_date(after: date) -> None:
    months = next_months(after, 3)
    starts = [month.start for month in months]
    assert starts == [date(2026, 12, 1), date(2027, 1, 1), date(2027, 2, 1)]


def test_realised_indices_by_hand() -> None:
    dates = pd.date_range("2027-01-31", "2027-03-01")  # only February is complete
    temperature = pd.Series([10.0, 20.0] * 15, index=dates)
    indices = realised_indices(temperature)
    assert [str(month) for month in indices.index] == ["2027-02"]
    february = indices.iloc[0]
    assert february["HDD"] == 14 * 8
    assert february["CDD"] == 14 * 2
    assert february["CAT"] == 14 * 30
    assert february["PRIM"] == 15


@pytest.mark.parametrize("index", INDICES)
@pytest.mark.parametrize("period", [JUNE, OCTOBER])
def test_futures_price_is_the_average_over_samples(
    pricer: Pricer, samples: pd.DataFrame, index: Index, period: DeliveryPeriod
) -> None:
    simulated = index_of(samples, index, period)
    error = simulated.std() / np.sqrt(len(simulated))
    assert pricer.futures_price(index, period) == pytest.approx(
        simulated.mean(), abs=4 * error
    )


def test_indices_satisfy_equations_10_and_11(pricer: Pricer) -> None:
    curve = pricer.futures_curve([JUNE, OCTOBER])
    n_days = np.array([30, 31])
    np.testing.assert_allclose(curve["PRIM"], curve["CAT"] / n_days)
    np.testing.assert_allclose(curve["HDD"], 18 * n_days - curve["CAT"] + curve["CDD"])


def test_cat_price_agrees_with_the_forecast(
    pricer: Pricer, model: TemperatureModel
) -> None:
    forecast = model.predict(200)["mean"].loc[OCTOBER.dates]
    assert pricer.futures_price("CAT", OCTOBER) == pytest.approx(forecast.sum())


def test_constant_market_price_of_risk_follows_equation_15(
    result: FitResult, pricer: Pricer
) -> None:
    theta, kappa = 0.3, result.reversion.kappa
    moved = Pricer(result, market_price_of_risk=theta)
    h = (JUNE.dates - result.last_date).days.to_numpy()
    expected = (theta * (1 - np.exp(-kappa * h)) / kappa).sum()
    change = moved.futures_price("CAT", JUNE) - pricer.futures_price("CAT", JUNE)
    assert change == pytest.approx(expected)


def test_market_price_of_risk_can_vary(result: FitResult) -> None:
    dates = pd.date_range(result.last_date, "2006-12-31")
    number = Pricer(result, market_price_of_risk=0.3)
    series = Pricer(result, market_price_of_risk=pd.Series(0.3, index=dates))
    for index in INDICES:
        assert series.futures_price(index, JUNE) == pytest.approx(
            number.futures_price(index, JUNE)
        )
    too_short = Pricer(result, market_price_of_risk=pd.Series(0.3, index=dates[:10]))
    with pytest.raises(ValueError, match="cover every day"):
        too_short.futures_price("CAT", JUNE)


def test_constant_volatility_changes_only_hdd_and_cdd(
    result: FitResult, pricer: Pricer
) -> None:
    constant = Pricer(result, constant_volatility=True)
    for index, same in (("CAT", True), ("PRIM", True), ("HDD", False)):
        a = pricer.futures_price(index, OCTOBER)  # type: ignore[arg-type]
        b = constant.futures_price(index, OCTOBER)  # type: ignore[arg-type]
        assert (a == pytest.approx(b)) is same


def test_cat_volatility_is_equation_25_for_daily_sums(
    result: FitResult, pricer: Pricer
) -> None:
    volatility = pricer.cat_volatility(OCTOBER)
    assert volatility.index[0] == result.last_date
    assert volatility.index[-1] == pd.Timestamp("2006-09-30")
    alpha, kappa = result.reversion.alpha, result.reversion.kappa
    t = result.clock.time(pd.DatetimeIndex(volatility.index))
    sigma = np.sqrt(result.variance(t)) / alpha
    to_start = np.arange(len(volatility), 0, -1)
    paper = (
        (np.exp(-kappa * to_start) - np.exp(-kappa * (to_start + 31))) * sigma / kappa
    )
    # A sum over days instead of an integral changes only the scale.
    np.testing.assert_allclose(volatility, paper * kappa / (1 - alpha))


def test_cat_volatility_adds_up_to_the_variance_of_the_index(
    pricer: Pricer, samples: pd.DataFrame
) -> None:
    """What is unknown at the start of delivery is added by hand."""
    before = (pricer.cat_volatility(OCTOBER) ** 2).sum()
    alpha = pricer.result.reversion.alpha
    days = OCTOBER.dates[:-1]
    left = np.arange(len(days), 0, -1)
    residual = pricer.result.variance(pricer.result.clock.time(days))
    during = (residual * ((1 - alpha**left) / (1 - alpha)) ** 2).sum()
    simulated = index_of(samples, "CAT", OCTOBER).var()
    assert before + during == pytest.approx(simulated, rel=0.03)


def option(pricer: Pricer, index: Index, kind: Any, strike: float) -> float:
    return pricer.option_price(index, kind, OCTOBER, strike=strike, exercise=EXERCISE)


@pytest.mark.parametrize("index", INDICES)
def test_put_call_parity(result: FitResult, index: Index) -> None:
    pricer = Pricer(result, interest_rate=0.05)
    price = pricer.futures_price(index, OCTOBER)
    for strike in (0.95 * price, price, 1.05 * price):
        difference = option(pricer, index, "call", strike) - option(
            pricer, index, "put", strike
        )
        years = (pd.Timestamp(EXERCISE) - result.last_date).days / 365
        assert difference == pytest.approx(np.exp(-0.05 * years) * (price - strike))


@pytest.mark.parametrize("index", INDICES)
def test_call_with_no_chance_of_loss_is_the_futures_price(
    pricer: Pricer, index: Index
) -> None:
    """The futures price is expected not to change (it is a martingale)."""
    low = -1e4
    call = option(pricer, index, "call", low)
    assert call + low == pytest.approx(pricer.futures_price(index, OCTOBER), rel=1e-6)


@pytest.mark.parametrize("index", ["CAT", "PRIM"])
def test_formula_agrees_with_the_grid(pricer: Pricer, index: Index) -> None:
    strike = pricer.futures_price(index, OCTOBER)
    at_exercise = pricer._price_at(index, OCTOBER, pd.Timestamp(EXERCISE))
    weights = np.exp(-(np.linspace(-8, 8, 2001) ** 2) / 2)
    grid = np.average(np.maximum(at_exercise - strike, 0), weights=weights)
    assert option(pricer, index, "call", strike) == pytest.approx(grid, rel=1e-4)


@pytest.mark.parametrize("index", INDICES)
@pytest.mark.parametrize("position", [600, 1000, 1400])  # cold, normal and warm
def test_price_at_exercise_is_the_average_over_samples_from_there(
    pricer: Pricer, index: Index, position: int
) -> None:
    result = pricer.result
    exercise = pd.Timestamp("2006-05-28")
    price = pricer._price_at(index, JUNE, exercise)[position]

    # The weather at that position of the grid, from the forecast of the model.
    ahead = pd.date_range("2006-05-26", exercise)
    before = result.clock.time(ahead - pd.Timedelta(days=1))
    std = np.sqrt(result.reversion.path_variance(result.variance(before))[-1])
    mean = result.reversion.expected_path(result.past_deviation(), len(ahead))[-1]
    weather = mean + std * np.linspace(-8, 8, 2001)[position]

    dates = pd.date_range("2006-05-29", JUNE.end)
    variance = result.variance(result.clock.time(dates - pd.Timedelta(days=1)))
    noise = np.random.default_rng(0).standard_normal((len(dates), 40000))
    paths = result.reversion.simulate(
        np.array([weather]), noise * np.sqrt(variance)[:, np.newaxis]
    )
    paths += result.seasonal_mean(dates).to_numpy()[:, np.newaxis]
    simulated = index_of(pd.DataFrame(paths, index=dates), index, JUNE)
    error = simulated.std() / np.sqrt(len(simulated))
    assert price == pytest.approx(simulated.mean(), abs=4 * error)


def test_option_is_discounted(result: FitResult, pricer: Pricer) -> None:
    years = (pd.Timestamp(EXERCISE) - result.last_date).days / 365
    discounted = option(Pricer(result, interest_rate=0.1), "CAT", "call", 200)
    assert discounted == pytest.approx(
        np.exp(-0.1 * years) * option(pricer, "CAT", "call", 200)
    )


def test_bad_requests_are_rejected(pricer: Pricer) -> None:
    with pytest.raises(ValueError, match="after the valuation date"):
        pricer.futures_price("HDD", DeliveryPeriod.month(2006, 5))
    unknown: Any = "SNOW"
    with pytest.raises(ValueError, match="index"):
        pricer.futures_price(unknown, JUNE)
    with pytest.raises(ValueError, match="kind"):
        option(pricer, "CAT", "straddle", 100)
    for exercise in (date(2006, 5, 25), date(2006, 10, 1)):
        with pytest.raises(ValueError, match="exercise"):
            pricer.option_price("CAT", "call", OCTOBER, strike=100, exercise=exercise)
