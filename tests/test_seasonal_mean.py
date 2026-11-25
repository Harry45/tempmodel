import numpy as np
import pytest

from tempmodel.model import SeasonalMean

T = np.arange(3 * 365)


def test_recovers_an_exact_signal() -> None:
    signal = (
        6 + 0.002 * T + 10 * np.cos(2 * np.pi * (T + 150) / 365)
        + 2 * np.cos(4 * np.pi * (T - 30) / 365)
    )  # fmt: skip
    mean = SeasonalMean.fit(T, signal, harmonics=2, trend=True)
    assert mean.level == pytest.approx(6)
    assert mean.trend == pytest.approx(0.002)
    np.testing.assert_allclose(mean.amplitudes, [10, 2])
    np.testing.assert_allclose(mean.phases, [-150, 30])
    np.testing.assert_allclose(mean(T), signal)


def test_amplitudes_and_phases_give_the_papers_formula() -> None:
    noise = np.random.default_rng(0).standard_normal(len(T))
    mean = SeasonalMean.fit(T, noise, harmonics=1, trend=True)
    wave = mean.amplitudes[0] * np.cos(2 * np.pi * (T - mean.phases[0]) / 365)
    np.testing.assert_allclose(mean(T), mean.level + mean.trend * T + wave)


def test_without_trend_every_year_is_the_same() -> None:
    mean = SeasonalMean.fit(T, T.astype(np.float64), harmonics=1, trend=False)
    assert mean.trend == 0
    np.testing.assert_allclose(mean(T[:365]), mean(T[-365:]))
