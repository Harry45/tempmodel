from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from tempmodel import figures
from tempmodel.figures import PAPER_FIGURES, plot_forecast
from tempmodel.model import FitResult, TemperatureModel
from tempmodel.plot_style import PlotStyle

HISTOGRAMS = [
    figures.plot_deviation_histogram,
    figures.plot_standardised_residual_histogram,
]
AUTOCORRELATIONS = [
    figures.plot_residual_autocorrelation,
    figures.plot_standardised_residual_autocorrelation,
]
SINGLE_PANEL = [figures.plot_seasonal_mean, figures.plot_daily_variance, *HISTOGRAMS]


@pytest.fixture(autouse=True)
def close_figures() -> Any:
    yield
    plt.close("all")


def bar_heights(fig: Figure) -> list[float]:
    bars = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)]
    return [bar.get_height() for bar in bars]


@pytest.mark.parametrize("plot", PAPER_FIGURES)
def test_paper_figures_draw(
    result: FitResult, plot: Callable[[FitResult], Figure]
) -> None:
    fig = plot(result)
    fig.canvas.draw()
    assert all(ax.get_title(loc="left").startswith("Figure") for ax in fig.axes)


def test_plot_forecast_draws(model: TemperatureModel, result: FitResult) -> None:
    samples = model.sample(365, 5)
    fig = plot_forecast(result, model.predict(365), samples)
    fig.canvas.draw()
    assert len(fig.axes[0].lines) == 5  # data, three paths and the mean
    fig = plot_forecast(result, model.predict(365), samples, n_paths=5, recent_days=30)
    lines = fig.axes[0].lines
    assert len(lines) == 7
    assert len(np.asarray(lines[0].get_xdata())) == 30


def test_hdd_curve_draws_both_prices() -> None:
    months = pd.date_range("2027-01-01", periods=3, freq="MS")
    seasonal = pd.Series([400.0, 300, 200], index=months)
    fig = figures.plot_hdd_curve(seasonal, seasonal - 20)
    fig.canvas.draw()
    labels = [label.get_text() for label in fig.axes[0].get_xticklabels()]
    assert labels == ["Jan\n2027", "Feb\n2027", "Mar\n2027"]
    assert len(fig.axes[0].collections) == 2


def test_cat_volatility_labels_the_delivery_month() -> None:
    dates = pd.date_range("2026-12-01", "2027-01-31")
    volatility = pd.Series(range(len(dates)), index=dates, dtype=float)
    fig = figures.plot_cat_volatility([volatility])
    fig.canvas.draw()
    assert [text.get_text() for text in fig.axes[0].texts] == ["Feb"]


def test_years_limits_the_history(result: FitResult) -> None:
    fig = figures.plot_seasonal_mean(result, years=2)
    assert len(np.asarray(fig.axes[0].lines[0].get_xdata())) == 730


@pytest.mark.parametrize("plot", HISTOGRAMS)
def test_bins_can_be_a_number_a_rule_or_edges(
    result: FitResult, plot: Callable[..., Figure]
) -> None:
    n_days = result.history["residual"].notna().sum()
    assert len(bar_heights(plot(result, bins=7))) == 7
    assert sum(bar_heights(plot(result, bins="auto"))) >= n_days
    assert len(bar_heights(plot(result, bins=[-20, -10, 0, 10, 20]))) == 4


@pytest.mark.parametrize("plot", HISTOGRAMS)
def test_unequal_bins_are_rejected(
    result: FitResult, plot: Callable[..., Figure]
) -> None:
    with pytest.raises(ValueError, match="equal widths"):
        plot(result, bins=[-20, 0, 1, 20])


@pytest.mark.parametrize("bins", [10, 40, np.linspace(-2, 2, 21).tolist()])
def test_curve_matches_the_bars(result: FitResult, bins: Any) -> None:
    fig = figures.plot_standardised_residual_histogram(result, bins=bins)
    highest_bar = max(bar_heights(fig))
    peak = np.max(np.asarray(fig.axes[0].lines[0].get_ydata()))
    assert peak == pytest.approx(highest_bar, rel=0.1)


@pytest.mark.parametrize("plot", AUTOCORRELATIONS)
def test_lag_and_level_can_be_set(
    result: FitResult, plot: Callable[..., Figure]
) -> None:
    fig = plot(result, max_lag=50, level=0.99)
    for ax in fig.axes:
        assert len(np.asarray(ax.lines[0].get_xdata())) == 50
        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_texts()[0].get_text().startswith("99% limits")
    n = result.history["residual"].notna().sum()
    band = fig.axes[0].patches[0].get_window_extent().transformed(
        fig.axes[0].transData.inverted()
    )  # fmt: skip
    assert band.y1 == pytest.approx(2.5758 / np.sqrt(n), rel=1e-3)


@pytest.mark.parametrize("plot", AUTOCORRELATIONS)
def test_lag_is_limited_by_the_data(
    result: FitResult, plot: Callable[..., Figure]
) -> None:
    n = result.history["residual"].notna().sum()
    fig = plot(result, max_lag=10**6)
    assert len(np.asarray(fig.axes[0].lines[0].get_xdata())) == n - 1


@pytest.mark.parametrize(
    ("plot", "options"),
    [
        (figures.plot_seasonal_mean, {"years": 0}),
        (figures.plot_residual_autocorrelation, {"max_lag": 0}),
        (figures.plot_residual_autocorrelation, {"level": 95}),
    ],
)
def test_bad_options_are_rejected(
    result: FitResult, plot: Callable[..., Figure], options: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="must be"):
        plot(result, **options)


@pytest.mark.parametrize("plot", SINGLE_PANEL)
def test_can_draw_on_an_existing_panel(
    result: FitResult, plot: Callable[..., Figure]
) -> None:
    fig, axes = plt.subplots(1, 2)
    assert plot(result, ax=axes[1]) is fig
    assert axes[1].lines
    assert not axes[0].lines
    assert not axes[1].spines["top"].get_visible()
    assert len(plt.get_fignums()) == 1


def test_forecast_can_draw_on_an_existing_panel(
    model: TemperatureModel, result: FitResult
) -> None:
    fig, ax = plt.subplots()
    assert plot_forecast(result, model.predict(5), model.sample(5, 5), ax=ax) is fig
    assert len(plt.get_fignums()) == 1


@pytest.mark.parametrize("plot", [*SINGLE_PANEL, *AUTOCORRELATIONS])
def test_size_can_be_set(result: FitResult, plot: Callable[..., Figure]) -> None:
    fig = plot(result, figsize=(3, 2))
    assert tuple(fig.get_size_inches()) == (3, 2)


def test_style_can_be_set(result: FitResult) -> None:
    style = PlotStyle(fit_color="#ff0000", fit_width=3)
    fig = figures.plot_seasonal_mean(result, style=style)
    line = fig.axes[0].lines[1]
    assert to_hex(line.get_color()) == "#ff0000"
    assert line.get_linewidth() == 3


def test_style_keeps_interactive_display_on() -> None:
    size = plt.rcParams["axes.titlesize"]
    with plt.rc_context({"interactive": False}):
        with PlotStyle().applied():
            plt.ion()  # as a notebook does while making its first figure
        assert plt.isinteractive()
        assert plt.rcParams["axes.titlesize"] == size
