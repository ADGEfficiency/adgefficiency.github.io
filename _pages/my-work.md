---
title: My Work
layout: archive
permalink: /my-work/
classes: wide
sidebar:
  nav: "top"

---

Hi - I'm Adam - an energy engineer turned data professional.

I work on using data & machine learning to combat climate change, and like to build models to optimize the control of energy systems. 

I am particularly interested in [the space that exists between](https://adgefficiency.com/space-between-money-and-the-planet/) economic and carbon optimization - the different outcomes we get when we design and optimize systems for money or for the environment.

Currently I'm a data engineer at [Gridcognition](https://gridcognition.com/), working on building software to navigate the transition to a decentralised and decarbonised future.  I occasionally do some consulting, speaking and mentoring.

[LinkedIn](https://www.linkedin.com/in/adgefficiency/) - [Twitter](https://twitter.com/ADGEfficiency) - [email](adam.green@adgefficiency.com) - [GitHub](https://github.com/ADGEfficiency) - [personalized mentoring](https://mentorcruise.com/mentor/AdamGreen/) - [CV](https://adgefficiency.com/cv.pdf)

# Projects

## Data Science South

A sister website focusing on educating data professionals.

[website](https://www.datasciencesouth.com/)

## climate-news-db

A tool for downloading climate change newspaper articles.

[app](https://www.climate-news-db.com/) - [blog post](https://www.datasciencesouth.com/blog/climate-news-db) - [source code](https://github.com/ADGEfficiency/climate-news-db)

![]({{"/assets/my-work/db.png"}})

## Soft Actor Critic (SAC) Reimplementation

[source code](https://github.com/ADGEfficiency/sac)

![]({{"/assets/my-work/sac.png"}})

## World Models Reimplementation

[blog post](https://adgefficiency.com/world-models/) - [source code](https://github.com/ADGEfficiency/world-models) - [references & resources](https://github.com/ADGEfficiency/rl-resources/tree/master/world-models)

![]({{"/assets/my-work/world.png"}})

## Teaching Monolith

[data science teaching materials](https://github.com/ADGEfficiency/teaching-monolith)

![]({{"/assets/my-work/monolith.png"}})

## Reinforcement Learning for Energy Systems

[blog post](https://www.adgefficiency.com/energy_py-reinforcement-learning-for-energy-systems/) - [github](https://github.com/ADGEfficiency/energy-py) - [DQN debugging](https://www.adgefficiency.com/dqn-debugging/), [hyperparameter tuning](https://www.adgefficiency.com/dqn-tuning/) and [solving](https://www.adgefficiency.com/dqn-solving/).

![]({{"/assets/dqn_solving/fig1.png"}})

## Mixed Integer Linear Programming of Battery Storage and Combined Heat & Power

[blog post](https://adgefficiency.com/intro-energy-py-linear/) - [github](https://github.com/ADGEfficiency/energy-py-linear) - [measuring forecast quality](https://adgefficiency.com/energy-py-linear-forecast-quality/)

```python
import energypylinear as epl
model = epl.Battery(power=2, capacity=4, efficiency=1.0)
prices = [10, 50, 10, 50, 10]
info = model.optimize(prices, timestep='30min')
```

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


## UK and Australian Grid Data

The Australian grid is a unique combination of high coal penetrations, quality potential renewable resources (and high penetration in South Australia) and a deregulated, volatile electricity market.  It also has good data availability - if you know where to look for it.

[A hackers guide to AEMO data](https://www.adgefficiency.com/hackers-aemo/) - [Elexon API Web Scraping using Python](https://www.adgefficiency.com/elexon-api-web-scraping-using-python/) - [What is the UK Imbalance Price?](http://www.adgefficiency.com/what-is-the-uk-imbalance-price/)


## Writing on Energy

I'm an energy engineer at heart.  Some of my most popular work is the *Energy Basics* series - such as [the heat equation](http://www.adgefficiency.com/energy-basics-q-m-cp-dt/) and [kW versus kWh](http://www.adgefficiency.com/energy-basics-kw-vs-kwh/).

I've also written about [Average versus Marginal Carbon Emissions](https://www.adgefficiency.com/energy-basics-average-vs-marginal-carbon-emissions/), the [Four Inconvenient Truths of the Clean Energy Transition](https://www.adgefficiency.com/four-inconvenient-truths-clean-energy-transition/) and the [intersection of energy and machine learning](http://localhost:4000/machine-learning-in-energy/).

## Parallelized Cross Entropy Method

[github](https://github.com/ADGEfficiency/cem) - [blog post](https://adgefficiency.com/cem/)

CEM on CartPole and Pendulum.  Parallelized across processes and through batch.

```bash
$ python cem.py cartpole --num_process 6 --epochs 8 --batch_size 4096

$ python cem.py pendulum --num_process 6 --epochs 15 --batch_size 4096
```

## Talks

April 3 2017 - Berlin Machine Learning Group - A Glance at Q-Learning - [meetup](https://www.meetup.com/berlin-machine-learning/events/234989414/)

June 21 2017 - Data Science Festival - A Glance at Q-Learning - [meetup](https://www.datasciencefestival.com/adam-green-glance-q-learning/) - [youtube](https://www.youtube.com/watch?v=25NPjJ6hBmI)

September 3 2018 - Berlin Machine Learning Group - energy-py - [meetup](https://www.meetup.com/berlin-machine-learning/events/246637693/) - [slides](https://gitpitch.com/ADGEfficiency/energy-py-talk) - [repo](https://github.com/ADGEfficiency/energy-py-talk)

September 24 2019 - Data Science Retreat Demo Day - Mistakes Data Scientists Make - [meetup](https://www.meetup.com/Data-Science-Retreat/events/264686728/) - [blog post](http://www.adgefficiency.com/mistakes-talk/) - [DSR talk](https://www.canva.com/design/DADlQld9yF0/share/preview?token=DoG2rySn8x8KGT5xMyoe6A&role=EDITOR&utm_content=DADlQld9yF0&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton) - [mistakes talk](https://www.canva.com/design/DADl9pRJd0c/share/preview?token=ptRfgqrLSz5BSZHgLXYTgA&role=EDITOR&utm_content=DADl9pRJd0c&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton)

April 10 2020 - Data Science Retreat Demo Day - [meetup](https://www.meetup.com/Data-Science-Retreat/events/269691369/) - [slides](https://www.canva.com/design/DAD1Z-Tx6n0/qZ1W579ElkdOifKzMOn1Og/view?utm_content=DAD1Z-Tx6n0&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink) - [recording](https://drive.google.com/open?id=1XyfRXAdNhh0zz6MWmPWRbeXgizSuLfbA)

June 9 2020 - AI Guild #datacareer workshop

May 26 2021 - AI / ML / DL - [slides](https://docs.google.com/presentation/d/1T0Kbf63yf_nAiNJar8pS8xgFL1yJTu3MbRk9cj-D1oQ/edit?usp=sharing)
