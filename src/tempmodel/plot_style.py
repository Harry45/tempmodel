"""The appearance of the figures, in one place."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


@dataclass(frozen=True)
class PlotStyle:
    """Colours and line widths.

    Attributes:
        data_color: Data, and sample paths.
        fit_color: Fitted curves, and bands.
        ink: Titles, recent data and autocorrelations.
        muted_ink: Axes and their labels.
        grid_color: Grid lines.
        data_width: Width of data lines.
        fit_width: Width of fitted curves.
        recent_width: Width of the recent data in a forecast.
        autocorrelation_width: Width of autocorrelation lines.
        band_opacity: Opacity of bands, from 0 to 1.
    """

    data_color: str = "#a8a8a8"
    fit_color: str = "#257de0"
    ink: str = "#191919"
    muted_ink: str = "#666666"
    grid_color: str = "#e6e6e6"
    data_width: float = 0.8
    fit_width: float = 2
    recent_width: float = 0.9
    autocorrelation_width: float = 0.6
    band_opacity: float = 0.18

    @contextmanager
    def applied(self) -> Iterator[None]:
        """Use this style for figures and text made inside the ``with`` block.

        Unlike ``plt.rc_context``, this restores only the settings it changes. A
        notebook turns on interactive display while its first figure is being made,
        and ``rc_context`` would turn it off again, hiding all later figures.
        """
        settings = self._rc_params()
        previous = {key: plt.rcParams[key] for key in settings}
        plt.rcParams.update(settings)
        try:
            yield
        finally:
            plt.rcParams.update(previous)

    def apply_to(self, ax: Axes) -> None:
        """Style a panel that may have been made outside ``applied``."""
        ax.set_axisbelow(True)
        ax.grid(visible=True, color=self.grid_color)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color(self.muted_ink)
        ax.tick_params(colors=self.muted_ink)
        ax.xaxis.label.set_color(self.muted_ink)
        ax.yaxis.label.set_color(self.muted_ink)

    def _rc_params(self) -> dict[Any, Any]:
        return {
            "axes.titlecolor": self.ink,
            "axes.titlelocation": "left",
            "axes.titlesize": "medium",
            "figure.constrained_layout.use": True,
            "legend.fontsize": "small",
            "legend.frameon": False,
        }
