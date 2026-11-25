import matplotlib as mpl
import pandas as pd
import pytest

from tempmodel.model import FitResult, TemperatureModel
from tests import simulated_stockholm

mpl.use("Agg")


@pytest.fixture(scope="session")
def temperature() -> pd.Series:
    return simulated_stockholm.simulate()


@pytest.fixture(scope="session")
def model(temperature: pd.Series) -> TemperatureModel:
    return TemperatureModel().fit(temperature)


@pytest.fixture(scope="session")
def result(model: TemperatureModel) -> FitResult:
    return model.result_
