---
title: 'Introducing energy-py-linear'
date: 2019-02-06
categories:
- Energy
excerpt: A library for optimizing energy systems using mixed integer linear programming.

---

This post introduces `energy-py-linear` - [a Python library for optimizing energy assets using mixed integer linear programming (MILP)](https://github.com/ADGEfficiency/energy-py-linear).

MILP guarantees convergence to the global optimum of a cost function for linear systems.  In energy battery storage and combined heat and power can be modelled and optimized using linear programming.

## Optimizing battery dispatch into a wholesale market

energy-py-linear can be used to optimize a battery that uses price arbitrage in a wholesale market ([see the source code here](https://github.com/ADGEfficiency/energy-py-linear/blob/master/energypylinear/battery/battery.py):

```python
import pandas as pd
import energypylinear as epl

prices = [10, 50, 10, 50, 10]
model = epl.Battery(power=2, capacity=4)
info = model.optimize(prices, timestep='30min')
pd.DataFrame().from_dict(info)

   Import [MW]  Export [MW]  Power [MW]  Charge [MWh]
0          2.0          0.0         2.0      0.000000
1          0.0          2.0        -2.0      0.066667
2          2.0          0.0         2.0      0.000000
3          0.0          2.0        -2.0      0.066667
4          NaN          NaN         NaN      0.000000
```

The battery charges during the low price periods and discharges during the high price periods.  The battery is fully discharged during the last period (a sign of optimal behaviour).  

The last `NaN` row is given because the `Charge` is the battery level at the start of each interval.  This last row tells us what the `Charge` level is for the battery after the optimization is finished.

The dispatch above is for perfectly forecast prices - the library can also be used to measure forecast quality by passing in a forecast along with true prices:

```python
#  a forecast that is the inverse of the prices we used above
forecasts = [50, 10, 50, 10, 50]

info = model.optimize(prices, forecasts=forecasts, timestep='30min')
```

## Optimizing combined heat and power

energy-py-linear can be used to optimize CHP systems.  First a list of the assets in the plant is made, then this plant configuration is optimized for given prices ([see the source code here](https://github.com/ADGEfficiency/energy-py-linear/blob/master/energypylinear/chp/chp.py)):

```python
from energypylinear import chp

assets = [
    chp.GasTurbine(size=10, name='gt1'),
    chp.Boiler(size=100, name='blr1'),
    chp.Boiler(size=100, name='blr2', efficiency=0.9),
    chp.SteamTurbine(size=6, name='st1')
]

info = chp.optimize(
    assets,
    gas_price=20,
    electricity_price=1000,
    site_steam_demand=100,
    site_power_demand=100,
)

print(info)
    total steam generated 130.0 t/h
    total steam consumed 30.0 t/h
    steam to site 100.0 t/h
    total power generated 16.0 MWe
    total power consumed 0.2 MWe
    net grid 84.2 MWe
    power to site 100.0 MWe
```

Check out this other post where I demonstrate how to [use energy-py-linear to measure forecast quality](https://adgefficiency.com/energy-py-linear-forecast-quality/).

Thanks for reading!
