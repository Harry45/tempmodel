"""Figures 1 to 8 of Benth & Šaltytė Benth (2007), and a forecast figure.

Every function takes a ``style``, and draws a new figure of size ``figsize``.
Those with a single panel can instead draw on an existing ``ax``.
"""

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from tempmodel.diagnostics import autocorrelation
from tempmodel.model import FitResult
from tempmodel.model.calendar import DAYS_PER_YEAR
from tempmodel.model.validation import check_count, check_level
from tempmodel.plot_style import PlotStyle

Size = tuple[float, float]
Bins = int | str | Sequence[float]

DEFAULT_STYLE = PlotStyle()

# Details of drawing that should not need changing.
CURVE_POINTS = 400
BAR_GAP_WIDTH = 0.5
MONTH_BAR_WIDTH = 0.8
LAG_AXIS_MARGIN = 0.015  # so that lag 1 clears the axis


def plot_seasonal_mean(
    result: FitResult,
    *,
    years: int = 10,
    ax: Axes | None = None,
    figsize: Size = (9, 4),
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Figure 1: temperature and the seasonal mean, over the last ``years``."""
    check_count("years", years)
    history = result.history.tail(years * DAYS_PER_YEAR)
    with style.applied():
        panel = _panel(ax, figsize, style)
        _plot_data(panel, history["temperature"], "Daily average temperature", style)
        _plot_fit(
            panel,
            history.index,
            history["seasonal_mean"],
            "Seasonal mean $s(t)$",
            style,
        )
        panel.set(title="Figure 1. Temperature and seasonal mean", ylabel="°C")
        _legend_above(panel, columns=2)
    return _figure_of(panel)


def plot_deviation_histogram(
    result: FitResult,
    *,
    bins: Bins = 20,
    ax: Axes | None = None,
    figsize: Size = (6, 4),
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Figure 2: histogram of the deviation, with a fitted normal.

    ``bins`` is a number of bins, a Matplotlib rule such as "auto", or a list of
    equally spaced edges.
    """
    deviation = result.history["deviation"].to_numpy()
    with style.applied():
        panel = _panel(ax, figsize, style)
        normal = stats.norm(deviation.mean(), deviation.std())
        _histogram(panel, deviation, bins, normal.pdf, "Fitted normal", style)
        panel.set(title="Figure 2. Temperature less trend and season", xlabel="°C")
    return _figure_of(panel)


def plot_residual_autocorrelation(
    result: FitResult,
    *,
    max_lag: int = 2000,
    level: float = 0.95,
    figsize: Size = (8, 6),
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Figure 3: autocorrelation of the residuals and of their squares.

    The band is where a fraction ``level`` of the values would fall if there were
    no autocorrelation. ``max_lag`` is reduced if there is not enough data.
    """
    return _plot_autocorrelations(
        result.history["residual"],
        "Figure 3. Residuals",
        "Figure 3. Squared residuals",
        max_lag=max_lag,
        level=level,
        figsize=figsize,
        style=style,
    )


def plot_daily_variance(
    result: FitResult,
    *,
    ax: Axes | None = None,
    figsize: Size = (7, 4),
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Figure 4: empirical daily variance of the residuals and the fitted function."""
    days = np.arange(DAYS_PER_YEAR)
    with style.applied():
        panel = _panel(ax, figsize, style)
        _plot_data(panel, result.variance.empirical, "Empirical", style)
        fitted = result.variance(days)
        _plot_fit(panel, days, fitted, r"Fitted $\tilde{\sigma}^2(t)$", style)
        panel.set(
            title="Figure 4. Daily variance",
            xlabel="Day of the year (0 = 1 January)",
            ylabel="°C²",
            xlim=(0, DAYS_PER_YEAR),
        )
        panel.set_ylim(bottom=0)
        _legend_above(panel, columns=2)
    return _figure_of(panel)


def plot_standardised_residual_autocorrelation(
    result: FitResult,
    *,
    max_lag: int = 2000,
    level: float = 0.95,
    figsize: Size = (8, 6),
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Figure 5: autocorrelation after dividing the residuals by the volatility.

    The options are those of ``plot_residual_autocorrelation``. The paper's caption
    describes both panels as squared residuals, before and after. Its text and its
    upper panel show the residuals themselves, so that is what is plotted here.
    """
    return _plot_autocorrelations(
        result.history["standardised_residual"],
        "Figure 5. Residuals divided by volatility",
        "Figure 5. Squared residuals divided by variance",
        max_lag=max_lag,
        level=level,
        figsize=figsize,
        style=style,
    )


def plot_standardised_residual_histogram(
    result: FitResult,
    *,
    bins: Bins = 20,
    ax: Axes | None = None,
    figsize: Size = (6, 4),
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Figure 6: histogram of the standardised residuals, with the standard normal.

    ``bins`` is as in ``plot_deviation_histogram``.
    """
    residual = result.history["standardised_residual"].dropna().to_numpy()
    with style.applied():
        panel = _panel(ax, figsize, style)
        _histogram(panel, residual, bins, stats.norm.pdf, "Standard normal", style)
        panel.set(
            title="Figure 6. Residuals divided by volatility",
            xlabel="Standardised residual",
        )
    return _figure_of(panel)


PAPER_FIGURES: tuple[Callable[[FitResult], Figure], ...] = (
    plot_seasonal_mean,
    plot_deviation_histogram,
    plot_residual_autocorrelation,
    plot_daily_variance,
    plot_standardised_residual_autocorrelation,
    plot_standardised_residual_histogram,
)


def plot_forecast(
    result: FitResult,
    forecast: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    n_paths: int = 3,
    recent_days: int = 180,
    ax: Axes | None = None,
    figsize: Size = (9, 4.5),
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Recent temperature, then the forecast mean, interval and some paths.

    Args:
        result: The fitted model.
        forecast: Output of ``predict``.
        samples: Output of ``sample``.
        n_paths: Number of sample paths to show.
        recent_days: Number of days of data to show.
        ax: Panel to draw on, instead of a new figure.
        figsize: Size of a new figure.
        style: Colours and line widths.
    """
    check_count("n_paths", n_paths)
    check_count("recent_days", recent_days)
    recent = result.history["temperature"].tail(recent_days)
    with style.applied():
        panel = _panel(ax, figsize, style)
        panel.plot(
            recent.index,
            recent,
            color=style.ink,
            linewidth=style.recent_width,
            label="Data",
        )
        panel.fill_between(
            forecast.index,
            forecast["lower"],
            forecast["upper"],
            color=style.fit_color,
            alpha=style.band_opacity,
            linewidth=0,
            label="Interval",
        )
        for i, path in enumerate(list(samples)[:n_paths]):
            _plot_data(panel, samples[path], "Samples" if i == 0 else None, style)
        _plot_fit(panel, forecast.index, forecast["mean"], "Forecast mean", style)
        panel.set(title=f"Forecast for the next {len(forecast)} days", ylabel="°C")
        _legend_above(panel, columns=4)
    return _figure_of(panel)


def plot_hdd_curve(
    seasonal: pd.Series,
    constant: pd.Series,
    *,
    ax: Axes | None = None,
    figsize: Size = (8, 4),
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Figure 7: HDD futures prices, with seasonal and with constant volatility.

    Args:
        seasonal: Prices by delivery month, with seasonal volatility.
        constant: The same with constant volatility.
        ax: Panel to draw on, instead of a new figure.
        figsize: Size of a new figure.
        style: Colours and line widths.
    """
    with style.applied():
        panel = _panel(ax, figsize, style)
        x = np.arange(len(seasonal))
        left, right = x - MONTH_BAR_WIDTH / 2, x + MONTH_BAR_WIDTH / 2
        panel.hlines(
            constant,
            left,
            right,
            color=style.ink,
            linestyles="--",
            linewidth=style.fit_width,
            label="Constant volatility",
        )
        panel.hlines(
            seasonal,
            left,
            right,
            color=style.fit_color,
            linewidth=style.fit_width,
            label="Seasonal volatility",
        )
        months = pd.DatetimeIndex(seasonal.index).strftime("%b\n%Y")
        panel.set_xticks(x, list(months))
        panel.set_ylim(bottom=0)
        panel.grid(visible=False, axis="x")
        panel.set(title="Figure 7. HDD futures curve", ylabel="HDD index points")
        _legend_above(panel, columns=2)
    return _figure_of(panel)


def plot_cat_volatility(
    volatility: Sequence[pd.Series],
    *,
    ax: Axes | None = None,
    figsize: Size = (8, 4),
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Figure 8: daily volatility of CAT futures up to delivery.

    Each Series is the output of ``cat_volatility`` for one delivery month, and is
    labelled with that month.
    """
    with style.applied():
        panel = _panel(ax, figsize, style)
        for series in volatility:
            panel.plot(
                series.index, series, color=style.fit_color, linewidth=style.fit_width
            )
            delivery = series.index[-1] + pd.Timedelta(days=1)
            panel.annotate(
                delivery.strftime("%b"),
                (series.index[-1], series.iloc[-1]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                color=style.muted_ink,
                fontsize="small",
            )
        panel.set(
            title="Figure 8. Volatility of CAT futures",
            ylabel="CAT index points per day",
        )
        panel.margins(y=0.1)  # room for the labels
        panel.set_ylim(bottom=0)
    return _figure_of(panel)


def _panel(ax: Axes | None, figsize: Size, style: PlotStyle) -> Axes:
    """Return ``ax``, or the panel of a new figure, ready to draw on."""
    panel = plt.subplots(figsize=figsize)[1] if ax is None else ax
    style.apply_to(panel)
    return panel


def _figure_of(panel: Axes) -> Figure:
    figure = panel.get_figure(root=True)
    assert figure is not None
    return figure


def _plot_data(ax: Axes, data: pd.Series, label: str | None, style: PlotStyle) -> None:
    ax.plot(
        data.index,
        data,
        color=style.data_color,
        linewidth=style.data_width,
        label=label,
    )


def _plot_fit(
    ax: Axes, x: ArrayLike, fit: ArrayLike, label: str, style: PlotStyle
) -> None:
    ax.plot(x, fit, color=style.fit_color, linewidth=style.fit_width, label=label)


def _legend_above(ax: Axes, columns: int) -> None:
    """Put the legend beside the title, clear of the data."""
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=columns)


def _histogram(
    ax: Axes,
    values: NDArray[np.float64],
    bins: Bins,
    density: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    label: str,
    style: PlotStyle,
) -> None:
    """Draw a histogram with a density scaled to the same area."""
    edges = np.histogram_bin_edges(values, bins)
    widths = np.diff(edges)
    if not np.allclose(widths, widths[0]):
        raise ValueError("bins must have equal widths, since the bars show counts")
    ax.hist(
        values,
        bins=list(edges),
        color=style.data_color,
        edgecolor="white",
        linewidth=BAR_GAP_WIDTH,
        label="Data",
    )
    x = np.linspace(edges[0], edges[-1], CURVE_POINTS)
    area = len(values) * widths[0]
    _plot_fit(ax, x, area * density(x), label, style)
    ax.set_ylabel("Number of days")
    ax.grid(visible=False, axis="x")
    ax.legend()


def _plot_autocorrelations(
    residual: pd.Series,
    title: str,
    squared_title: str,
    *,
    max_lag: int,
    level: float,
    figsize: Size,
    style: PlotStyle,
) -> Figure:
    """Draw the autocorrelation of the residuals above that of their squares."""
    check_count("max_lag", max_lag)
    check_level(level)
    values = residual.dropna().to_numpy()
    max_lag = min(max_lag, len(values) - 1)
    limit = float(stats.norm.ppf((1 + level) / 2)) / np.sqrt(len(values))

    with style.applied():
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        panels = ((axes[0], values, title), (axes[1], values**2, squared_title))
        for ax, panel_values, panel_title in panels:
            style.apply_to(ax)
            ax.plot(
                np.arange(1, max_lag + 1),
                autocorrelation(panel_values, max_lag),
                color=style.ink,
                linewidth=style.autocorrelation_width,
            )
            ax.axhspan(
                -limit,
                limit,
                color=style.fit_color,
                alpha=style.band_opacity,
                linewidth=0,
                label=f"{100 * level:g}% limits if uncorrelated",
            )
            ax.set(
                title=panel_title,
                ylabel="Autocorrelation",
                xlim=(-LAG_AXIS_MARGIN * max_lag, max_lag),
            )
            ax.legend(loc="upper right")
        axes[1].set_xlabel("Lag (days)")
    return fig
