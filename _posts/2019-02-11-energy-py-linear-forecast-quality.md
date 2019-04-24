---
title: 'Measuring forecast quality using linear programming'
date: 2019-02-11
categories:
 - Energy
excerpt: Using energy-py-linear to measure the economic value of using a forecast.

---

The ideal forecast quality measurement directly aligns with a key business metric.  Models are not often able to be trained in this way - often models are trained using error measures that will look familiar to anyone who does gradient based optimization, such as mean squared error.

This post uses [energy-py-linear](https://github.com/ADGEfficiency/energy-py-linear) to measure forecast quality in terms of a key business metric - cost.  energy-py-linear is a Python library for optimizing energy systems using mixed integer linear programming - [read the introductory post here](https://adgefficiency.com/intro-energy-py-linear/).

A battery operating in price arbitrage is optimized using actual prices and forecast prices.  The forecast error can then be quantified by how much money dispatching the battery using the forecast leaves on the table versus dispatching with perfect foresight of prices.

[A Jupyter notebook with the code used in this blog post is available in the energy-py-linear GitHub repository](https://github.com/ADGEfficiency/energy-py-linear/blob/master/notebooks/forecast_quality.ipynb).

# NEM electricity prices & forecast data

The dataset used is a single sample of prices and forecasts supplied by AEMO for the National Electricity Market (NEM) in Australia.  The price is the South Australian trading price and the forecast is the AEMO supplied predispatch price.

```python
import pandas as pd

dataset = pd.read_csv('.notebooks/data/forecast_sample.csv', index_col=0, parse_dates=True)

Timestamp           Trading Price [$/MWh]    Predispatch Forecast [$/MWh]
2018-07-01 17:00:00                177.11                        97.58039
2018-07-01 17:30:00                135.31                       133.10307
2018-07-01 18:00:00                143.21                       138.59979
2018-07-01 18:30:00                116.25                       128.09559
2018-07-01 19:00:00                 99.97                       113.29413
```

A copy of the data is available in [energy-py-linear/notebooks/data/forecast_sample.csv](https://github.com/ADGEfficiency/energy-py-linear/blob/master/notebooks/data/forecast_sample.csv).  A simple plot of the price and forecast is show below in Figure 1.

![]({{ "/assets/linear-forecast/forecast.png" }})

**Figure 1 - a sample of the South Australian trading price and predispatch forecast from July 2018**

## Using battery storage to measure electricity price forecast quality

First we create an instance of the `Battery` class.  We use a large capacity battery so that the battery will chase after all possible arbitrage opportunities with a roundtrip efficiency of 100%.

```python
import energypylinear

model = energypylinear.Battery(capacity=1000, power=2, efficiency=1.0)
```

We then dispatch the battery using perfect foresight of prices:

```python
perfect_foresight = model.optimize(
    prices=dataset.loc[:, 'Trading Price [$/MWh]'],
    timestep='30min'
)
```

Next we dispatch using the forecast:

```python
forecast = model.optimize(
    prices=dataset.loc[:, 'Trading Price [$/MWh]'],
    forecasts=dataset.loc[:, 'Predispatch Forecast [$/MWh]'],
    timestep='30min'
)
```

Finally after some massaging of the `perfect_foresight` and `forecast` objects (which are lists of dictionaries) into dataframes, we end up with:

```python
#  we multiply by -1 to convert net costs into net benefits
perfect_total = -1 * perfect_foresight.loc[:, 'Actual [$/30min]'].sum()
forecast_total = -1 * forecast.loc[:, 'Actual [$/30min]'].sum()
forecast_error = perfect_total - forecast_total

"""
Optimal dispatch is a benefit of $ 429.48
Dispatching under the forecast gave a benefit of $ 302.59

Forecast error is $ 126.89
Forecast error is 29.55 %
"""
```

## Thoughts on using linear programming to measure forecast quality

Any optimization algorithm can be used in place of linear programming.  If there is any error in the optimization (say to a local minima) then the final quality measurement combines the error from both forecasting and from the optimization that used the forecast.

Linear programming is great if your system can be modelled as linear - you end up with the global optimum every time.  As we are interested in energy, battery storage is an application where we can build this linear program.

This can can be specific to a certain battery configuration - for example a battery developer could measure how much using a certain forecast costs them.  This can also be extended to other energy systems such as combined heat and power which can be modelled as linear programs.  The ability to get measurements of forecast quality as a function of specific asset configurations is attractive for energy engineers.

A challenge with using this measurement of forecast error is what happens when the net benefit of dispatching the battery to a forecast - i.e. when the forecast quality is so bad that using it ends up losing money.  Unlike other error measures such as mean squared error it's not appropriate to simply take the absolute.

A final consideration is how this method can be used for other time series, such as stock prices or weather prediction.  A large capacity battery operating in price arbitrage does somewhat resemble arbitrage of stocks, so the error measurement might be useful for comparing forecasts.  It's less clear how useful this model would be for a temperature prediction.

---

In summary - this post introduced a method for measuring forecast quality using linear optimization of electric battery storage.  It's suitable for measuring the economic impact of using a forecast for specific energy assets - it's applicability to other time series is an open question.

Thanks for reading!
