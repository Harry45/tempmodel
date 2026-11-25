import numpy as np
import pytest

from tempmodel.diagnostics import autocorrelation, normality_summary


def test_autocorrelation_matches_the_definition() -> None:
    x = np.random.default_rng(0).standard_normal(300).cumsum()
    y = x - x.mean()
    expected = [(y[:-lag] * y[lag:]).sum() / (y * y).sum() for lag in range(1, 51)]
    np.testing.assert_allclose(autocorrelation(x, 50), expected, atol=1e-12)


@pytest.mark.parametrize("max_lag", [0, 10])
def test_bad_lag_is_rejected(max_lag: int) -> None:
    with pytest.raises(ValueError, match="max_lag"):
        autocorrelation(np.arange(10), max_lag)


def test_normality_summary_of_normal_values() -> None:
    summary = normality_summary(np.random.default_rng(0).standard_normal(20000))
    assert summary["skewness"] == pytest.approx(0, abs=0.06)
    assert summary["kurtosis"] == pytest.approx(0, abs=0.12)
