import inspect
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from tempmodel import figures
from tempmodel.config import Settings, load_settings
from tempmodel.model import TemperatureModel


def test_shipped_config_is_valid() -> None:
    settings = load_settings()
    assert settings.period.years == 10
    assert settings.location.name == "London Heathrow"


def test_overrides_are_applied() -> None:
    assert load_settings(["period.years=3"]).period.years == 3


@pytest.mark.parametrize(
    "override",
    [
        "period.years=0",
        "period.years=-1",
        "period.years=2.5",
        "period.years=true",
        "period.years='10'",
        "location.latitude=91",
        "location.longitude=-181",
        "location.timezone=Europe/Nowhere",
        "location.timezone=../etc",
        "api.url=not-a-url",
        "api.variables=[]",
        "api.variables=[a,a]",
        "api.variables=['a,b']",
        "api.timeout_seconds=0",
        "api.max_retries=-1",
        "+period.months=2",
        "model.column=other",
        "model.seasonal_harmonics=0",
        "model.variance_harmonics=1.5",
        "forecast.n_days=0",
        "forecast.level=1",
        "forecast.n_samples=0",
        "forecast.seed=-1",
        "api.retry_backoff_seconds=-1",
        "plots.bins=0",
        "plots.bins=2.5",
        "plots.bins=many",
        "plots.history_years=0",
        "plots.max_lag=0",
        "plots.level=1",
        "plots.n_paths=0",
        "plots.recent_days=0",
    ],
)
def test_invalid_values_are_rejected(override: str) -> None:
    with pytest.raises(ValidationError):
        load_settings([override])


def test_settings_are_immutable() -> None:
    settings: Any = load_settings()
    with pytest.raises(ValidationError):
        settings.period.years = 5


def test_missing_section_is_rejected() -> None:
    raw = load_settings().model_dump()
    del raw["period"]
    with pytest.raises(ValidationError):
        Settings.model_validate(raw)


def test_bins_can_be_a_rule() -> None:
    assert load_settings(["plots.bins=auto"]).plots.bins == "auto"


def default(function: Callable[..., Any], name: str) -> Any:
    return inspect.signature(function).parameters[name].default


def test_config_agrees_with_the_defaults_in_the_code() -> None:
    settings = load_settings()
    model, forecast, plots = settings.model, settings.forecast, settings.plots
    for name in ("seasonal_harmonics", "variance_harmonics", "trend"):
        assert getattr(model, name) == default(TemperatureModel, name)
    assert forecast.level == default(TemperatureModel.predict, "level")
    assert forecast.n_samples == default(TemperatureModel.sample, "n_samples")
    assert plots.history_years == default(figures.plot_seasonal_mean, "years")
    for histogram in (
        figures.plot_deviation_histogram,
        figures.plot_standardised_residual_histogram,
    ):
        assert plots.bins == default(histogram, "bins")
    for autocorrelation in (
        figures.plot_residual_autocorrelation,
        figures.plot_standardised_residual_autocorrelation,
    ):
        assert plots.max_lag == default(autocorrelation, "max_lag")
        assert plots.level == default(autocorrelation, "level")
    assert plots.n_paths == default(figures.plot_forecast, "n_paths")
    assert plots.recent_days == default(figures.plot_forecast, "recent_days")
