---
title:  ""
layout: archive
permalink: /my-work/
classes: wide

---

![]({{"/assets/teaching.jpg"}})

I'm an energy engineer / data scientist who likes to build models to control energy systems.  I've worked on industrial & district energy projects at a international energy utility and on demand side flexibility at an early stage company.

Currently I'm the director at Data Science Retreat - an advanced Berlin bootcamp.  I'm also doing a bit of teaching on topics I am passionate about - [backpropagation](https://github.com/ADGEfficiency/teaching-monolith/tree/master/backprop), [bash + Git](https://github.com/ADGEfficiency/teaching-monolith/tree/master/bash-and-git), [test driven development](https://github.com/ADGEfficiency/teaching-monolith/tree/master/test-driven-development) and [reinforcement learning](https://github.com/ADGEfficiency/rl-course).

I also maintain repositories of [programming](https://github.com/ADGEfficiency/programming-resources), [reinforcement learning](https://github.com/ADGEfficiency/rl-resources) and [machine learning](https://github.com/ADGEfficiency/ml-resources) resources.

Contact me on [LinkedIn](https://www.linkedin.com/in/adgefficiency/) or [via email](adam.green@adgefficiency.com). [Check out my work on GitHub](https://github.com/ADGEfficiency).

## Reinforcement learning for energy systems 

[blog post](https://www.adgefficiency.com/energy_py-reinforcement-learning-for-energy-systems/) - [github](https://github.com/ADGEfficiency/energy-py) - [DQN debugging](https://www.adgefficiency.com/dqn-debugging/), [hyperparameter tuning](https://www.adgefficiency.com/dqn-tuning/) and [solving](https://www.adgefficiency.com/dqn-solving/).

![]({{"/assets/dqn_solving/fig1.png"}})

## Mixed integer linear programming of battery storage and combined heat and power 

[blog post](https://adgefficiency.com/intro-energy-py-linear/) - [github](https://github.com/ADGEfficiency/energy-py-linear) - [measuring forecast quality](https://adgefficiency.com/energy-py-linear-forecast-quality/)

```python
import energypylinear as epl
model = epl.Battery(power=2, capacity=4, efficiency=1.0)
prices = [10, 50, 10, 50, 10]
info = model.optimize(prices, timestep='30min')
```

## UK and Australian grid data

The Australian grid is a unique combination of high coal penetrations, quality potential renewable resources (and high penetration in South Australia) and a deregulated, volatile electricity market.  It also has good data availability - if you know where to look for it.

[A hackers guide to AEMO data](https://www.adgefficiency.com/hackers-aemo/) - [Elexon API Web Scraping using Python](https://www.adgefficiency.com/elexon-api-web-scraping-using-python/) - [What is the UK Imbalance Price?](http://localhost:4000/what-is-the-uk-imbalance-price/)

## Combined heat and power

I spent four years working as an industrial energy engineer, and worked with a lot of CHP plant.  [energy-py-linear](https://github.com/ADGEfficiency/energy-py-linear) has a CHP model that can be configured with a number of gas and steam turbines, then optimized as a function of gas and electricity prices.

```python
from energypylinear.chp import Boiler, GasTurbine, SteamTurbine

assets = [
	GasTurbine(size=10, name='gt1'),
	Boiler(size=100, name='blr1'),
	Boiler(size=100, name='blr2', efficiency=0.9),
	SteamTurbine(size=6, name='st1')
]

info = optimize(
	assets,
	gas_price=20,
	electricity_price=1000,
	site_steam_demand=100,
	site_power_demand=100,
)
```

[CHP Cheat Sheet - Gas Engines & Gas Turbines](https://www.adgefficiency.com/cheat-sheet-gas-engine-gas-turbine-chp-energy-basics/) - [Four Negative Effects of High Return Temperatures](https://www.adgefficiency.com/energy-basics-four-negative-effects-of-high-return-temperatures/)

## Energy

I'm an energy engineer at heart.  Some of my most popular work is the *Energy Basics* series - such as [the heat equation](http://localhost:4000/energy-basics-q-m-cp-dt/) and [kW versus kWh](http://localhost:4000/energy-basics-kw-vs-kwh/).

I've also written about [Average versus Marginal Carbon Emissions](https://www.adgefficiency.com/energy-basics-average-vs-marginal-carbon-emissions/), the [Four Inconvenient Truths of the Clean Energy Transition](https://www.adgefficiency.com/four-inconvenient-truths-clean-energy-transition/) and the [intersection of energy and machine learning](http://localhost:4000/machine-learning-in-energy-part-one/).

## Parallelized Cross Entropy Method

[github](https://github.com/ADGEfficiency/cem)

CEM on CartPole and Pendulum.  Parallelized across processes and through batch.

```bash
$ python cem.py cartpole --num_process 6 --epochs 8 --batch_size 4096

$ python cem.py pendulum --num_process 6 --epochs 15 --batch_size 4096
```

## Talks

April 3 2017 - Berlin Machine Learning Group - A Glance at Q-Learning - [meetup page](https://www.meetup.com/berlin-machine-learning/events/234989414/) - [youtube](https://www.youtube.com/watch?v=25NPjJ6hBmI)

June 21 2017 - Data Science Festival - A Glance at Q-Learning - [meetup page](https://www.datasciencefestival.com/adam-green-glance-q-learning/) - [youtube](https://www.youtube.com/watch?v=25NPjJ6hBmI)

September 3 2018 - Berlin Machine Learning Group - energy-py - [meetup page](https://www.meetup.com/berlin-machine-learning/events/246637693/) - [slides](https://gitpitch.com/ADGEfficiency/energy-py-talk) - [GitHub repo](https://github.com/ADGEfficiency/energy-py-talk)
