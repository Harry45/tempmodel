from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tempmodel.model import FitResult, NotFittedError, TemperatureModel
from tests import stockholm


def test_uses_as_many_records_as_the_paper(result: FitResult) -> None:
    assert len(result.history) == 16570


def test_recovers_the_papers_parameters(result: FitResult) -> None:
    parameters = result.parameters()
    assert parameters["a"] == pytest.approx(stockholm.A, abs=0.4)
    assert parameters["b"] == pytest.approx(stockholm.B, abs=0.00005)
    assert parameters["b1"] == pytest.approx(stockholm.B1, abs=0.3)
    assert parameters["g1"] == pytest.approx(stockholm.G1, abs=2)
    assert parameters["alpha"] == pytest.approx(stockholm.ALPHA, abs=0.015)
    assert parameters["kappa"] == pytest.approx(0.194, abs=0.02)
    assert parameters["residual_std"] == pytest.approx(2.04, abs=0.05)
    days = np.arange(365)
    np.testing.assert_allclose(
        result.variance(days), stockholm.daily_variance(days), rtol=0.2
    )


def test_parameters_are_named_as_in_table_1(result: FitResult) -> None:
    parameters = result.parameters()
    expected = ["c", *(f"{x}{i}" for x in "cd" for i in range(1, 5))]
    assert list(parameters.index[-9:]) == expected
    scaled = parameters["c"] * result.reversion.alpha**2
    assert scaled == pytest.approx(result.variance.constant)


def test_standardised_residuals_are_standard_normal(result: FitResult) -> None:
    residual = result.history["standardised_residual"]
    assert residual.mean() == pytest.approx(0, abs=0.03)
    assert residual.std() == pytest.approx(1, abs=0.02)


def test_residual_has_the_volatility_of_the_day_before(result: FitResult) -> None:
    t = result.clock.time(pd.DatetimeIndex(result.history.index))
    np.testing.assert_allclose(
        result.history["volatility"].iloc[1:] ** 2, result.variance(t[:-1])
    )


def test_prediction_follows_equation_5(
    model: TemperatureModel, result: FitResult, temperature: pd.Series
) -> None:
    forecast = model.predict(400)
    assert forecast.index[0] == temperature.index[-1] + pd.Timedelta(days=1)
    assert len(forecast) == 400

    alpha = result.reversion.alpha
    yesterday = pd.DatetimeIndex(temperature.index[-1:])
    deviation = temperature.iloc[-1] - result.seasonal_mean(yesterday).iloc[0]
    variance = 0.0
    for date, row in forecast.iterrows():
        t = result.clock.time(yesterday)
        deviation *= alpha
        variance = alpha**2 * variance + result.variance(t)[0]
        yesterday = pd.DatetimeIndex([date])
        mean = result.seasonal_mean(yesterday).iloc[0] + deviation
        assert row["mean"] == pytest.approx(mean)
        assert row["std"] ** 2 == pytest.approx(variance)


def test_interval_matches_level(model: TemperatureModel) -> None:
    forecast = model.predict(10, level=0.8)
    below = stats.norm.cdf(forecast["lower"], forecast["mean"], forecast["std"])
    above = stats.norm.sf(forecast["upper"], forecast["mean"], forecast["std"])
    np.testing.assert_allclose(below, 0.1)
    np.testing.assert_allclose(above, 0.1)


def test_samples_agree_with_prediction(model: TemperatureModel) -> None:
    forecast = model.predict(365)
    samples = model.sample(365, 20000, random_state=0)
    assert samples.shape == (365, 20000)
    assert samples.index.equals(forecast.index)
    np.testing.assert_allclose(samples.mean(axis=1), forecast["mean"], atol=0.15)
    np.testing.assert_allclose(samples.std(axis=1), forecast["std"], rtol=0.03)


def test_samples_have_the_right_persistence(
    model: TemperatureModel, result: FitResult
) -> None:
    samples = model.sample(1000, 200, random_state=0)
    deviation = samples.sub(model.predict(1000)["mean"], axis=0).iloc[100:]
    before, after = deviation.iloc[:-1].to_numpy(), deviation.iloc[1:].to_numpy()
    slope = (before * after).sum() / (before**2).sum()
    assert slope == pytest.approx(result.reversion.alpha, abs=0.01)


def test_samples_are_reproducible(model: TemperatureModel) -> None:
    first = model.sample(5, 3, random_state=1)
    pd.testing.assert_frame_equal(first, model.sample(5, 3, random_state=1))
    assert not first.equals(model.sample(5, 3, random_state=2))


def test_forecast_includes_29_february(temperature: pd.Series) -> None:
    forecast = TemperatureModel().fit(temperature.loc[:"2004-02-20"]).predict(20)
    assert pd.Timestamp("2004-02-29") in forecast.index
    assert len(forecast) == 20


def test_forecast_can_start_from_29_february(temperature: pd.Series) -> None:
    data = temperature.loc[:"2004-02-29"].copy()
    data.iloc[-1] = 30.0
    forecast = TemperatureModel().fit(data).predict(1)
    assert forecast.index[0] == pd.Timestamp("2004-03-01")
    assert forecast["mean"].iloc[0] > 20


def test_options_reach_the_parts_of_the_model(temperature: pd.Series) -> None:
    model = TemperatureModel(seasonal_harmonics=2, variance_harmonics=3, trend=False)
    result = model.fit(temperature).result_
    assert len(result.mean.amplitudes) == 2
    assert len(result.variance.sine) == 3
    assert result.mean.trend == 0


def test_refit_replaces_previous_fit(temperature: pd.Series) -> None:
    shorter = temperature.loc["1990-06-01":"2000-01-01"]
    refitted = TemperatureModel().fit(temperature).fit(shorter)
    fresh = TemperatureModel().fit(shorter)
    pd.testing.assert_frame_equal(refitted.predict(30), fresh.predict(30))


def test_failed_refit_keeps_previous_fit(temperature: pd.Series) -> None:
    model = TemperatureModel().fit(temperature)
    before = model.result_
    dates = pd.date_range("2000-01-01", periods=2000)
    with pytest.raises(ValueError, match="mean-reverting"):
        model.fit(pd.Series(np.tile([-1.0, 1.0], 1000), index=dates))
    assert model.result_ is before


def test_fit_does_not_share_data_with_the_caller(temperature: pd.Series) -> None:
    data = temperature.copy()
    model = TemperatureModel().fit(data)
    before = model.predict(5)
    data.iloc[-1] = 99.0
    pd.testing.assert_frame_equal(model.predict(5), before)


def test_use_before_fit_is_rejected() -> None:
    model = TemperatureModel()
    with pytest.raises(NotFittedError):
        model.predict(5)
    with pytest.raises(NotFittedError):
        model.sample(5)
    with pytest.raises(NotFittedError):
        _ = model.result_


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_bad_counts_are_rejected(
    model: TemperatureModel, temperature: pd.Series, count: Any
) -> None:
    with pytest.raises(ValueError, match="n_days"):
        model.predict(count)
    with pytest.raises(ValueError, match="n_days"):
        model.sample(count)
    with pytest.raises(ValueError, match="n_samples"):
        model.sample(5, count)
    with pytest.raises(ValueError, match="seasonal_harmonics"):
        TemperatureModel(seasonal_harmonics=count).fit(temperature)
    with pytest.raises(ValueError, match="variance_harmonics"):
        TemperatureModel(variance_harmonics=count).fit(temperature)


@pytest.mark.parametrize("level", [0, 1, 95])
def test_bad_level_is_rejected(model: TemperatureModel, level: float) -> None:
    with pytest.raises(ValueError, match="level"):
        model.predict(5, level=level)


def test_bad_data_is_rejected(temperature: pd.Series) -> None:
    data = temperature.iloc[:1500]
    with pytest.raises(ValueError, match="DatetimeIndex"):
        TemperatureModel().fit(data.reset_index(drop=True))
    with pytest.raises(ValueError, match="3 years"):
        TemperatureModel().fit(data.iloc[:1000])
    with pytest.raises(ValueError, match="every day"):
        TemperatureModel().fit(data.drop(data.index[700]))
    with pytest.raises(ValueError, match="every day"):
        TemperatureModel().fit(data.iloc[::-1])
    missing = data.copy()
    missing.iloc[5] = np.nan
    with pytest.raises(ValueError, match="missing"):
        TemperatureModel().fit(missing)
