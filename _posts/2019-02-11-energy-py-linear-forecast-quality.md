---
title: 'Measuring Forecast Accuracy with Linear Programming'
date_created: 2019-02-11
date_updated: 2023-02-23
date: 2023-02-23
categories:
 - Energy
excerpt: Using energy-py-linear to measure the accuracy of a forecast.
classes: wide2
toc: true
toc_sticky: true

---

This post introduces a methodology to measure the accuracy of an electricity price forecast using linear programming.

## Predictive Accuracy vs. Business Value

The ideal forecast quality measurement directly aligns with a key business metric.  Models are not often able to be trained in this way - often models are trained using error measures that will look familiar to anyone who does gradient based optimization, such as mean squared error.

This post uses a linear programming to measure forecast quality in terms of a key business metric - cost.

A battery operating in price arbitrage is optimized using actual prices and forecast prices.  

The forecast error can then be quantified by how much money dispatching the battery using the forecast leaves on the table versus dispatching with perfect foresight of prices.

# Data

This work uses [energy-py-linear](https://github.com/ADGEfficiency/energy-py-linear) for the battery linear program - you can find the code & data in [examples/forecast-accuracy.py](https://github.com/ADGEfficiency/energy-py-linear/blob/main/examples/forecast-accuracy.py) - the full source code is also available at the bottom of this post.

```python
$ pip install energypylinear
```

The dataset used is a single sample of the South Australian trading price and the AEMO predispatch price forecast.

Both the price and forecast are supplied by AEMO for the National Electricity Market (NEM) in Australia.  

A simple plot of the price and forecast is show below in Figure 1:

![]({{ "/assets/linear-forecast/forecast.png" }})

<center>
  <em>Figure 1 - South Australian trading price and predispatch forecast from July 2018.</em>
</center>

# Method

First we create an instance of the `Battery` class.  We use a large capacity battery so that the battery will chase after all possible arbitrage opportunities with a roundtrip efficiency of 100%.

```python
import energypylinear as epl

asset = epl.battery.Battery(
    power_mw=2,
    capacity_mwh=4,
    efficiency=0.9
)
```

We then dispatch the battery using perfect foresight of prices:

```python
actuals = asset.optimize(
    electricity_prices=data['Trading Price [$/MWh]'],
    freq_mins=30,
)
```

Next we dispatch using the forecast:

```python
forecasts = asset.optimize(
    electricity_prices=data['Predispatch Forecast [$/MWh]'],
    freq_mins=30,
)
```

We can then create `epl.Account` objects to represent the financials for these two simulations.

The trick is using the actuals interval data with the forecast simulation in `forecast_account` - this evaluates the economics with actual prices but dispatch optimized for forecasts:

```python
#  calculate the variance between accounts
actual_account = epl.get_accounts(actuals.interval_data, actuals.simulation)
forecast_account = epl.get_accounts(actuals.interval_data, forecasts.simulation)
variance = actual_account - forecast_account

print(f"actuals: {actual_account}")
print(f"forecasts: {forecast_account}")
print(f"variance: {variance}")
print(f"\nforecast error: $ {-1 * variance.cost:2.2f} pct: {100 * variance.cost / actual_account.cost:2.1f} %")
```

# Discussion

## Extend to Different Domains

The method above is specific to using batteries for wholesale price arbitrage.

The idea of using variance between two optimization runs with different inputs can be extended to many business problems.

If there is any error in the optimization (say to a local minima) then the final quality measurement combines the error from both forecasting and from the optimization that used the forecast.

A large capacity battery operating in price arbitrage does somewhat resemble arbitrage of stocks, so the error measurement might be useful for comparing forecasts.  It's less clear how useful this model would be for a temperature prediction.

## Negative Value

A challenge with using this measurement of forecast error is what happens when the net benefit of dispatching the battery to a forecast - i.e. when the forecast quality is so bad that using it ends up losing money.  Unlike other error measures such as mean squared error it's not appropriate to simply take the absolute.

# Summary

This post introduces a method for measuring forecast accuracy using linear optimization of electric battery storage, by looking at the difference between two optimization runs given actual and forecast prices as input.

---

Thanks for reading!
