import numpy as np
import pytest

from tempmodel.model import MeanReversion

HALF = MeanReversion(alpha=0.5)


def test_recovers_exact_decay() -> None:
    reversion = MeanReversion.fit(np.power(0.8, np.arange(20), dtype=np.float64))
    assert reversion.alpha == pytest.approx(0.8)
    assert reversion.kappa == pytest.approx(-np.log(0.8))


def test_residuals_start_on_the_second_day() -> None:
    residual = HALF.residuals(np.array([4.0, 3.0, 1.0]))
    np.testing.assert_allclose(residual, [np.nan, 1.0, -0.5])


def test_expected_path_decays_from_the_last_deviation() -> None:
    path = HALF.expected_path(np.array([100.0, 8.0]), 3)
    np.testing.assert_allclose(path, [4, 2, 1])


def test_path_variance_accumulates() -> None:
    np.testing.assert_allclose(
        HALF.path_variance(np.array([4.0, 1.0, 1.0])), [4, 2, 1.5]
    )


def test_simulate_without_noise_is_the_expected_path() -> None:
    past = np.array([8.0])
    paths = HALF.simulate(past, np.zeros((3, 2)))
    for path in paths.T:
        np.testing.assert_allclose(path, HALF.expected_path(past, 3))


def test_simulate_inverts_residuals() -> None:
    deviation = np.random.default_rng(0).standard_normal(50)
    residual = HALF.residuals(deviation)[1:, np.newaxis]
    paths = HALF.simulate(deviation[:1], residual)
    np.testing.assert_allclose(paths[:, 0], deviation[1:])


@pytest.mark.parametrize("ratio", [-0.5, 1.1])
def test_no_mean_reversion_is_rejected(ratio: float) -> None:
    with pytest.raises(ValueError, match="mean-reverting"):
        MeanReversion.fit(ratio ** np.arange(20.0))
