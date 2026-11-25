"""The seasonal mean s(t), equation (2)."""

from dataclasses import dataclass
from typing import Self

import numpy as np

from tempmodel.model.calendar import DAYS_PER_YEAR, FloatArray, IntArray
from tempmodel.model.fourier import angular_frequency, least_squares, sine_and_cosine


@dataclass(frozen=True, eq=False)
class SeasonalMean:
    """A linear trend plus yearly waves.

    s(t) = a + b t + sum_j b_j cos(2 pi j (t - g_j) / 365)

    Each wave is stored as a sine and a cosine. That makes the fit linear, so it
    has an exact solution and needs no starting values. The paper's separate sine
    terms are redundant, because each cosine already has a free phase.

    Attributes:
        level: a, the mean at t = 0.
        trend: b, the change per day.
        sine: Weight on the sine of each harmonic.
        cosine: Weight on the cosine of each harmonic.
    """

    level: float
    trend: float
    sine: FloatArray
    cosine: FloatArray

    @classmethod
    def fit(
        cls,
        t: IntArray,
        temperature: FloatArray,
        *,
        harmonics: int,
        trend: bool,
    ) -> Self:
        """Fit to temperature by least squares."""
        sine, cosine = sine_and_cosine(t, harmonics)
        constant = np.ones(len(t))
        years = t / DAYS_PER_YEAR  # similar in size to the other columns
        trend_column = [years] if trend else []
        weights = least_squares([constant, *trend_column, sine, cosine], temperature)
        return cls(
            level=float(weights[0]),
            trend=float(weights[1]) / DAYS_PER_YEAR if trend else 0.0,
            sine=weights[-2 * harmonics : -harmonics],
            cosine=weights[-harmonics:],
        )

    def __call__(self, t: IntArray) -> FloatArray:
        """Return s(t)."""
        sine, cosine = sine_and_cosine(t, len(self.sine))
        return np.asarray(
            self.level + self.trend * t + sine @ self.sine + cosine @ self.cosine
        )

    @property
    def amplitudes(self) -> FloatArray:
        """b_j, the size of each wave."""
        return np.hypot(self.sine, self.cosine)

    @property
    def phases(self) -> FloatArray:
        """g_j, the day on which each wave peaks."""
        angle = np.arctan2(self.sine, self.cosine)
        return np.asarray(angle / angular_frequency(len(self.sine)))
