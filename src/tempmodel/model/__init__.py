"""Temperature model of Benth & Šaltytė Benth (2007), sections 2 and 3.

Daily temperature is a seasonal mean plus a mean-reverting deviation whose
volatility varies over the year::

    T(t)   = s(t) + X(t)                            seasonal_mean.py, equation (2)
    X(t+1) = alpha X(t) + volatility(t) eps(t)      mean_reversion.py, equation (5)
    volatility(t)^2 = daily_variance(t)             seasonal_variance.py, equation (3)

``temperature_model.py`` puts them together.

Reference: Quantitative Finance 7(5), 553-561, doi:10.1080/14697680601155334.
"""

from tempmodel.model.fit_result import FitResult
from tempmodel.model.mean_reversion import MeanReversion
from tempmodel.model.seasonal_mean import SeasonalMean
from tempmodel.model.seasonal_variance import SeasonalVariance
from tempmodel.model.temperature_model import NotFittedError, TemperatureModel

__all__ = [
    "FitResult",
    "MeanReversion",
    "NotFittedError",
    "SeasonalMean",
    "SeasonalVariance",
    "TemperatureModel",
]
