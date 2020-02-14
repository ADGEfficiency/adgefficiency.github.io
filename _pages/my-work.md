---
title:  ""
layout: archive
permalink: /my-work/
classes: wide

---

![]({{"/assets/teaching.jpg"}})

*Teaching backpropagation*

An energy engineer turned data scientist.  [Currently I am the Director at Data Science Retreat](https://datascienceretreat.com/) - Europe's most advanced data science bootcamp.

I like building models to optimize the control of energy systems. I am particularly interested in the space that exists between economic and carbon optimization.

I have been an energy engineer on industrial & district energy projects at ENGIE, as well as a data scientist at Tempus Energy.  I am currently rounding out a solid technical base with management experience running Data Science Retreat.

Contact me on [LinkedIn](https://www.linkedin.com/in/adgefficiency/) or [via email](adam.green@adgefficiency.com).

[All of my work is open source](https://github.com/ADGEfficiency). [See my photos on flickr](https://www.flickr.com/photos/37628582@N00/). [I sometimes stream on Twitch](https://www.twitch.tv/climatecoder).

## Data science teaching

[You can see the resources I use to teach data science in the teaching monolith](https://github.com/ADGEfficiency/teaching-monolith).

[I'm offering personalized mentoring on Mentor Cruise](https://mentorcruise.com/mentor/AdamGreen/).

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

[A hackers guide to AEMO data](https://www.adgefficiency.com/hackers-aemo/) - [Elexon API Web Scraping using Python](https://www.adgefficiency.com/elexon-api-web-scraping-using-python/) - [What is the UK Imbalance Price?](http://www.adgefficiency.com/what-is-the-uk-imbalance-price/)

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

I'm an energy engineer at heart.  Some of my most popular work is the *Energy Basics* series - such as [the heat equation](http://www.adgefficiency.com/energy-basics-q-m-cp-dt/) and [kW versus kWh](http://www.adgefficiency.com/energy-basics-kw-vs-kwh/).

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

September 24 2019 - Data Science Retreat Demo Day - Mistakes Data Scientists Make - [blog post](http://www.adgefficiency.com/mistakes-talk/) - [DSR talk](https://www.canva.com/design/DADlQld9yF0/share/preview?token=DoG2rySn8x8KGT5xMyoe6A&role=EDITOR&utm_content=DADlQld9yF0&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton) - [mistakes talk](https://www.canva.com/design/DADl9pRJd0c/share/preview?token=ptRfgqrLSz5BSZHgLXYTgA&role=EDITOR&utm_content=DADl9pRJd0c&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton)
