"""Yearly waves shared by the seasonal mean and the seasonal variance."""

import numpy as np

from tempmodel.model.calendar import DAYS_PER_YEAR, FloatArray, IntArray


def angular_frequency(harmonics: int) -> FloatArray:
    """Return the frequencies of waves repeating 1, 2, ... times a year."""
    return np.asarray(2 * np.pi * np.arange(1, harmonics + 1) / DAYS_PER_YEAR)


def sine_and_cosine(t: IntArray, harmonics: int) -> tuple[FloatArray, FloatArray]:
    """Return sine and cosine waves, with one column per harmonic."""
    angle = np.outer(t, angular_frequency(harmonics))
    return np.sin(angle), np.cos(angle)


def least_squares(columns: list[FloatArray], target: FloatArray) -> FloatArray:
    """Return the weights on ``columns`` that best reproduce ``target``."""
    design = np.column_stack(columns)
    return np.asarray(np.linalg.lstsq(design, target, rcond=None)[0])
