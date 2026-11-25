"""Typed, validated settings, loaded from the Hydra configuration."""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    HttpUrl,
    NonNegativeFloat,
    PositiveFloat,
    StringConstraints,
    field_validator,
    model_validator,
)

from tempmodel.model.validation import MIN_YEARS

VariableName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]
Probability = Annotated[float, Field(gt=0, lt=1)]
BinRule = Literal["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class _Base(BaseModel):
    """Immutable settings that reject unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ApiSettings(_Base):
    """How to query the Open-Meteo archive."""

    url: HttpUrl
    model: NonEmptyStr
    variables: Annotated[tuple[VariableName, ...], Field(min_length=1)]
    timeout_seconds: PositiveFloat
    max_retries: Annotated[int, Field(strict=True, ge=0)]
    retry_backoff_seconds: NonNegativeFloat

    @field_validator("variables")
    @classmethod
    def _unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("variables must not contain duplicates")
        return value


class LocationSettings(_Base):
    """Where to fetch data for."""

    name: NonEmptyStr
    latitude: Annotated[float, Field(ge=-90, le=90)]
    longitude: Annotated[float, Field(ge=-180, le=180)]
    timezone: NonEmptyStr

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError, OSError) as error:
            raise ValueError(f"unknown timezone {value!r}") from error
        return value


class PeriodSettings(_Base):
    """How much history to fetch."""

    years: PositiveStrictInt


class OutputSettings(_Base):
    """Where to save the data."""

    path: Path


class ModelSettings(_Base):
    """The temperature model and the data it is fitted to."""

    column: VariableName
    seasonal_harmonics: PositiveStrictInt
    variance_harmonics: PositiveStrictInt
    trend: bool


class ForecastSettings(_Base):
    """How to forecast."""

    n_days: PositiveStrictInt
    level: Probability
    n_samples: PositiveStrictInt
    seed: Annotated[int, Field(strict=True, ge=0)]


class BacktestSettings(_Base):
    """How to check past forecasts."""

    min_years: Annotated[int, Field(strict=True, ge=MIN_YEARS)]


class PlotSettings(_Base):
    """What the figures show; their appearance is in ``PlotStyle``."""

    bins: PositiveStrictInt | BinRule
    history_years: PositiveStrictInt
    max_lag: PositiveStrictInt
    level: Probability
    n_paths: PositiveStrictInt
    recent_days: PositiveStrictInt


class PricingSettings(_Base):
    """How to price futures and options."""

    n_months: PositiveStrictInt
    base_temperature: FiniteFloat
    market_price_of_risk: FiniteFloat
    interest_rate: FiniteFloat


class Settings(_Base):
    """All settings."""

    api: ApiSettings
    location: LocationSettings
    period: PeriodSettings
    output: OutputSettings
    model: ModelSettings
    forecast: ForecastSettings
    backtest: BacktestSettings
    plots: PlotSettings
    pricing: PricingSettings

    @model_validator(mode="after")
    def _column_is_downloaded(self) -> Self:
        if self.model.column not in self.api.variables:
            raise ValueError(f"model.column {self.model.column!r} not in api.variables")
        return self

    @classmethod
    def from_hydra(cls, config: DictConfig) -> Self:
        """Validate a Hydra configuration."""
        return cls.model_validate(OmegaConf.to_container(config, resolve=True))


def load_settings(overrides: Sequence[str] = ()) -> Settings:
    """Load settings outside the command line, e.g. in a notebook or test.

    Args:
        overrides: Hydra overrides such as ``["period.years=5"]``.
    """
    with initialize_config_module("tempmodel.conf", version_base="1.3"):
        return Settings.from_hydra(compose("config", overrides=list(overrides)))
