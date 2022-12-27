---
title: My Work
layout: archive
permalink: /my-work/
classes: wide
sidebar:
  nav: "top"
author_profile: true

---

Energy engineer turned data professional.  My work is focused on using data to combat climate change.   I am passionate & enjoy modelling energy systems, especially for optimization & control.

[I am interested in the space that exists between economic and carbon optimization](https://adgefficiency.com/space-between-money-and-the-planet/) - the different outcomes we get when we design and optimize systems for money or for the environment.

I'm currently working as a data scientist at [Orkestra Energy](https://www.orkestra.energy), working on building software to navigate the transition to a decentralised and decarbonised future.

[LinkedIn](https://www.linkedin.com/in/adgefficiency/) - [Twitter](https://twitter.com/ADGEfficiency) - [email](adam.green@adgefficiency.com) - [GitHub](https://github.com/ADGEfficiency) - [mentoring](https://mentorcruise.com/mentor/AdamGreen/) - [CV](https://adgefficiency.com/cv.pdf)


# Projects

## Data Science South

[A sister website](https://www.datasciencesouth.com/) focusing on data professional (analyst, engineer, scientist) education.

## climate-news-db

A tool for downloading climate change newspaper articles - [app](https://www.climate-news-db.com/) - [source code](https://github.com/ADGEfficiency/climate-news-db).

![]({{"/assets/my-work/db.png"}})


## [nem-data](https://github.com/ADGEfficiency/nem-data)

CLI for downloading data for Australia's National Energy Market (NEM):

```bash
$ pip install nemdata
```

/Users/adam/adgefficiency.github.io/_pages/my-work.md

## Soft Actor Critic (SAC) Reimplementation

Implementing the SAC Reinforcement learning algorithm - an off policy algorithm capable of both continuous and discrete policies - [source code](https://github.com/ADGEfficiency/sac).

![]({{"/assets/my-work/sac.png"}})

## World Models Reimplementation

Implementing the 2017 classic paper - using a variational auto-encoder, mixed density networks and evolutionary optimization to learn to race a car from pixels - [blog post](https://adgefficiency.com/world-models/) - [source code](https://github.com/ADGEfficiency/world-models) - [references & resources](https://github.com/ADGEfficiency/rl-resources/tree/master/world-models).

![]({{"/assets/my-work/world.png"}})

## Teaching Monolith

[Data science teaching materials](https://github.com/ADGEfficiency/teaching-monolith).

![]({{"/assets/my-work/monolith.png"}})

## Reinforcement Learning for Energy Systems

[blog post](https://www.adgefficiency.com/energy_py-reinforcement-learning-for-energy-systems/) - [github](https://github.com/ADGEfficiency/energy-py) - [DQN debugging](https://www.adgefficiency.com/dqn-debugging/), [hyperparameter tuning](https://www.adgefficiency.com/dqn-tuning/) and [solving](https://www.adgefficiency.com/dqn-solving/).

![]({{"/assets/dqn_solving/fig1.png"}})

## Mixed Integer Linear Programming of Battery Storage and Combined Heat & Power

[blog post](https://adgefficiency.com/intro-energy-py-linear/) - [github](https://github.com/ADGEfficiency/energy-py-linear) - [measuring forecast quality](https://adgefficiency.com/energy-py-linear-forecast-quality/)

## UK and Australian Grid Data

The Australian grid is a unique combination of high coal penetrations, quality potential renewable resources (and high penetration in South Australia) and a deregulated, volatile electricity market.  It also has good data availability - if you know where to look for it.

[A hackers guide to AEMO data](https://www.adgefficiency.com/hackers-aemo/) - [Elexon API Web Scraping using Python](https://www.adgefficiency.com/elexon-api-web-scraping-using-python/) - [What is the UK Imbalance Price?](http://www.adgefficiency.com/what-is-the-uk-imbalance-price/)

## Parallelized Cross Entropy Method

[github](https://github.com/ADGEfficiency/cem) - [blog post](https://adgefficiency.com/cem/)

CEM on CartPole and Pendulum.  Parallelized across processes and through batch.

```bash
$ python cem.py pendulum --num_process 6 --epochs 15 --batch_size 4096
```

