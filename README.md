# tempmodel

A model of daily temperature that forecasts the mean, a prediction interval and
sample paths for the days ahead, and prices futures and options on temperature.


## Setup

Requires Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Then open a notebook with the `.venv` kernel:

- `playground.ipynb` goes through the model one equation at a time, with the paper's
  Figures 1 to 6 for London.
- `pricing.ipynb` prices futures and options, with Figures 7 and 8.

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

## Pricing

Futures on HDD, CDD, CAT and PRIM, as of the last day of data, in index points:

```python
from datetime import timedelta

from tempmodel.pricing import Pricer, next_months

pricer = Pricer(result, market_price_of_risk=0.0, interest_rate=0.0)
months = next_months(pricer.valuation_date, 12)

pricer.futures_curve(months)  # a row per month, a column per index
pricer.cat_volatility(months[2])  # for each day up to delivery
pricer.option_price(
    "HDD",
    "call",
    months[2],
    strike=330,
    exercise=months[2].start - timedelta(days=1),
)
```

The market price of risk and the interest rate are assumptions, as there are no market
quotes here to fit them to. Prices use the daily model, with sums in place of the
paper's integrals, because contracts settle on daily values.

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

## Backtest

```python
from tempmodel.backtest import backtest, summarise

summarise(backtest(temperature, model))
```

This forecasts from the first day of each past month, using only earlier data, and
compares with what happened. On data up to 24 November 2026, over 48 forecasts of 365
days:

| | |
|---|---|
| Days inside the 95% interval | 93% |
| Error (RMSE), one day ahead | 1.75 °C |
| Error (RMSE), over the year | 2.94 °C |
| Bias (positive: warmer than forecast) | +0.36 °C |

## Limitations

- **Trend.** Ten years is too short to estimate climate change. For London it comes
  out near 1.5 °C per decade, partly because the data source changed. It is kept
  because it helps the forecast: without it, forecasts tend to be too cold. Use
  `trend=False` to leave it out.
- **Prices.** HDD prices depend strongly on the trend. For London the next 12 months
  add up to about 2160 with it and 2410 without it. The past average is 2435 and the
  latest 12 months 2110 (data up to 24 November 2026).
- **Figure 7.** The paper reports that seasonal volatility nearly doubles summer HDD
  prices. Here it lowers them, as it does with the paper's own parameters in July and
  August. See `pricing.ipynb`.
- **Data.** Values come from weather models, not a station. Open-Meteo's default
  source changes over time, with a small step at the start of 2025.
- **Interval.** It is where each day's temperature should fall. It leaves out
  uncertainty in the parameters.
- **Short-term dependence.** The residuals are still slightly correlated over the
  first two days. The paper suggests a higher-order model, but for London extra lags
  lowered the one-day error by only about 1%.

## License

MIT, see [LICENSE](LICENSE). Weather data by [Open-Meteo](https://open-meteo.com/).
