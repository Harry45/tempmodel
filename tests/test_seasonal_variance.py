import numpy as np
import pytest

from tempmodel.model import SeasonalVariance
from tests import stockholm

T = np.arange(200 * 365)
DAYS = np.arange(365)


def test_recovers_the_papers_variance() -> None:
    noise = np.random.default_rng(0).standard_normal(len(T))
    variance = SeasonalVariance.fit(
        T, noise * np.sqrt(stockholm.daily_variance(T)), harmonics=4
    )
    np.testing.assert_allclose(variance(DAYS), stockholm.daily_variance(DAYS), rtol=0.1)


def test_empirical_is_the_variance_of_each_day_of_the_year() -> None:
    residual = np.random.default_rng(0).standard_normal(len(T))
    variance = SeasonalVariance.fit(T, residual, harmonics=4)
    by_year = residual.reshape(200, 365)
    np.testing.assert_allclose(variance.empirical, by_year.var(axis=0, ddof=1))


def test_missing_residuals_are_ignored() -> None:
    residual = np.random.default_rng(0).standard_normal(len(T))
    with_gap = np.concatenate([[np.nan], residual])
    np.testing.assert_allclose(
        SeasonalVariance.fit(np.arange(-1, len(T)), with_gap, harmonics=4)(DAYS),
        SeasonalVariance.fit(T, residual, harmonics=4)(DAYS),
    )


def test_negative_variance_is_rejected() -> None:
    noise = np.random.default_rng(0).standard_normal(len(T))
    short_burst = np.where(T % 365 < 10, 10, 0.1)
    with pytest.raises(ValueError, match="not positive"):
        SeasonalVariance.fit(T, noise * short_burst, harmonics=4)
