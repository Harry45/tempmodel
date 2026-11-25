# tempmodel

A model of daily temperature that forecasts the mean, a prediction interval and
sample paths for the days ahead.


## Setup

Requires Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Then open `playground.ipynb` with the `.venv` kernel. It goes through the model one
equation at a time and reproduces the paper's Figures 1 to 6 for London.

## The model

$$T(t) = s(t) + X(t), \qquad X(t+1) = \alpha X(t) + \tilde\sigma(t)\,\epsilon_t, \qquad \epsilon_t \sim N(0, 1)$$

Temperature is a seasonal mean plus a deviation. Each day keeps a fraction $\alpha$ of
the previous day's deviation and adds noise whose size depends on the time of year.
The model is fitted in three steps:

| Step | Fitted by | Code in `src/tempmodel/model/` |
|---|---|---|
| 1. Trend and seasonal mean $s(t)$, equation (2) | Least squares on temperature | `seasonal_mean.py` |
| 2. Mean reversion $\alpha = e^{-\kappa}$, equation (5) | Regressing each deviation on the previous day's | `mean_reversion.py` |
| 3. Daily variance $\tilde\sigma^2(t)$, equation (3) | Least squares on the residual variance of each day of the year | `seasonal_variance.py` |

As in the paper, 29 February is left out of the fit. Forecasts do include it.

## Fit and forecast

```python
from tempmodel.config import load_settings
from tempmodel.data import get_daily_temperature
from tempmodel.model import TemperatureModel

settings = load_settings()
temperature = get_daily_temperature(settings)["temperature_2m_mean"]  # up to yesterday

model = TemperatureModel().fit(temperature)
forecast = model.predict(365, level=0.95)          # mean, std, lower, upper
samples = model.sample(365, 1000, random_state=0)  # one column per path

result = model.result_
result.parameters()  # named as in the paper
result.history       # deviations, residuals and volatility for each day
```

`temperature` can be any pandas Series with one value per day, at least three years
long and with no gaps.

To repeat an analysis on the same data, save a copy with `tempmodel-download`, then
read it with `load_csv` from `tempmodel.data`.

## Figures

```python
from tempmodel import figures

figures.plot_seasonal_mean(result)                          # Figure 1
figures.plot_deviation_histogram(result, bins=40)           # Figure 2
figures.plot_residual_autocorrelation(result)               # Figure 3
figures.plot_daily_variance(result)                         # Figure 4
figures.plot_standardised_residual_autocorrelation(result)  # Figure 5
figures.plot_standardised_residual_histogram(result)        # Figure 6
figures.plot_forecast(result, forecast, samples)
```

All take `figsize` and `style`.

## Settings

The location, number of years and model options are in
`src/tempmodel/conf/config.yaml`. Change them there, or when loading:

```python
settings = load_settings(["period.years=20"])
```

`TemperatureModel` takes its options directly:

```python
TemperatureModel(seasonal_harmonics=2, variance_harmonics=3, trend=False)
```

## Limitations

- **Trend.** Ten years is too short to estimate it well. For London it comes out near
  1.5 °C per decade, which is too steep to trust, and makes the forecast for the next
  year about 0.8 °C warmer than with no trend. Use `trend=False` to leave it out.
- **Data.** Values come from weather models, not a station. Open-Meteo's default
  source changes over time, with a small step at the start of 2025.
- **Interval.** It is where each day's temperature should fall. It leaves out
  uncertainty in the parameters. In a backtest on London, the 95% interval contained
  93% of days.
- **Short-term dependence.** The residuals are still slightly correlated over the
  first two days. The paper finds the same and suggests a higher-order model.
