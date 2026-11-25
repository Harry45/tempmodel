import pandas as pd
import pytest

from tempmodel.backtest import backtest, summarise
from tempmodel.model import NotFittedError, TemperatureModel


@pytest.fixture(scope="module")
def recent(temperature: pd.Series) -> pd.Series:
    return temperature.loc["1996-05-26":]


@pytest.fixture(scope="module")
def results(recent: pd.Series) -> pd.DataFrame:
    return backtest(recent, TemperatureModel())


def test_starts_each_month_with_enough_data_on_both_sides(
    results: pd.DataFrame,
) -> None:
    starts = results.index.unique("start")
    assert starts[0] == pd.Timestamp("2001-06-01")
    assert starts[-1] == pd.Timestamp("2005-05-01")
    assert len(starts) == 48
    assert len(results) == 48 * 365
    assert results.notna().all(axis=None)


def test_rows_match_a_forecast_made_by_hand(
    recent: pd.Series, results: pd.DataFrame
) -> None:
    by_hand = TemperatureModel().fit(recent.loc[:"2003-01-31"]).predict(365)
    rows = results.xs(pd.Timestamp("2003-02-01"), level="start")
    assert rows.index.tolist() == list(range(1, 366))
    assert rows["date"].tolist() == by_hand.index.tolist()
    for column in ("mean", "std", "lower", "upper"):
        assert rows[column].tolist() == pytest.approx(by_hand[column].tolist())
    assert rows["actual"].tolist() == recent.loc[by_hand.index].tolist()


def test_forecasts_do_not_use_later_data(recent: pd.Series) -> None:
    changed = recent.copy()
    changed.loc["2003-02-01":] += 50
    options = {"n_days": 30}
    before = backtest(recent, TemperatureModel(), **options)
    after = backtest(changed, TemperatureModel(), **options)
    start = pd.Timestamp("2003-02-01")
    pd.testing.assert_series_equal(
        before.xs(start, level="start")["mean"], after.xs(start, level="start")["mean"]
    )


def test_coverage_is_right_for_data_from_the_model(results: pd.DataFrame) -> None:
    summary = summarise(results)
    assert summary["forecasts"] == 48
    assert summary["coverage"] == pytest.approx(0.95, abs=0.02)
    assert summary["bias"] == pytest.approx(0, abs=0.5)
    assert summary["rmse_one_day_ahead"] < summary["rmse"]


def test_summary_by_hand() -> None:
    index = pd.MultiIndex.from_product([["a"], [1, 2]], names=["start", "days_ahead"])
    results = pd.DataFrame(
        {"mean": [0.0, 0], "lower": [-2.0, -2], "upper": [2.0, 2], "actual": [1.0, 3]},
        index=index,
    )
    summary = summarise(results)
    assert summary["coverage"] == 0.5
    assert summary["bias"] == 2
    assert summary["rmse"] == pytest.approx(5**0.5)
    assert summary["rmse_one_day_ahead"] == 1


def test_model_is_not_changed(recent: pd.Series) -> None:
    model = TemperatureModel()
    backtest(recent, model, n_days=10)
    with pytest.raises(NotFittedError):
        _ = model.result_


def test_too_little_data_is_rejected(recent: pd.Series) -> None:
    with pytest.raises(ValueError, match="not enough data"):
        backtest(recent.iloc[: 6 * 365], TemperatureModel())
    with pytest.raises(ValueError, match="min_years"):
        backtest(recent, TemperatureModel(), min_years=2)
