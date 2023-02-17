---
title: My Work
layout: archive
permalink: /my-work/
classes: wide
sidebar:
  nav: "top"
author_profile: true

---

Energy engineer turned data professional.  My work is focused on using data to combat climate change.   I enjoy modelling energy systems.

I am interested in the space that exists between [economic and carbon optimization](https://adgefficiency.com/space-between-money-and-the-planet/) - the different outcomes we get when we design and optimize energy systems for money or for the environment.

I love mentoring data professionals - you can sign up for [mentoring with me here](https://mentorcruise.com/mentor/AdamGreen/). See my [CV here](https://adgefficiency.com/cv.pdf).

# Projects

## Data Science South

[A sister website](https://datasciencesouth.com/) focusing on data professional (analyst, engineer, scientist) education.

![]({{"/assets/my-work/dss.png"}})

## Space Between Money and the Planet

Demonstrating the existence of a tradeoff between monetary gain and carbon emissions reduction in the dispatch of electric batteries:

![](/assets/space-between-2023/annual.png)

[Blog post](https://adgefficiency.com/space-between-money-and-the-planet/) - [source code](https://github.com/ADGEfficiency/space-between-money-and-the-planet).

## Linear Programming for Energy Systems 

[energy-py-linear](https://github.com/ADGEfficiency/energy-py-linear) - A Python library for optimizing the dispatch of energy assets with mixed-integer linear programming, including batteries, electric vehicles and CHP generators:

```bash
$ pip install energypylinear
```

[Introductory blog post](https://adgefficiency.com/intro-energy-py-linear/) - [github](https://github.com/ADGEfficiency/energy-py-linear) - [measuring forecast quality](https://adgefficiency.com/energy-py-linear-forecast-quality/) - [source code](https://github.com/ADGEfficiency/energy-py-linear).

## climate-news-db

A dataset of climate change newspaper articles - [app](https://www.climate-news-db.com/) - [source code](https://github.com/ADGEfficiency/climate-news-db).

![]({{"/assets/my-work/db.png"}})

## Creative Writing with GPT2

Fine-tune a base GPT2 model for your favourite authors - [source code](https://github.com/ADGEfficiency/creative-writing-with-gpt2).

![]({{"/assets/my-work/creative.png"}})

## Australian Electricity Grid Data

[nem-data](https://github.com/ADGEfficiency/nem-data) - a CLI for downloading data for Australia's National Energy Market (NEM):

```bash
$ pip install nemdata
```

[A Hackers Guide to AEMO Data](https://www.adgefficiency.com/hackers-aemo/) is a developer focused guide to the electricity market data provide by AEMO for the NEM - [source code](https://github.com/ADGEfficiency/nem-data).

## UK Electricity Grid Data

[Elexon API Web Scraping using Python](https://www.adgefficiency.com/elexon-api-web-scraping-using-python/) and [What is the UK Imbalance Price?](http://www.adgefficiency.com/what-is-the-uk-imbalance-price/).


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

[Blog post](https://www.adgefficiency.com/energy_py-reinforcement-learning-for-energy-systems/) - [github](https://github.com/ADGEfficiency/energy-py) - [DQN debugging](https://www.adgefficiency.com/dqn-debugging/), [hyperparameter tuning](https://www.adgefficiency.com/dqn-tuning/) and [solving](https://www.adgefficiency.com/dqn-solving/).

![]({{"/assets/dqn_solving/fig1.png"}})


## Parallelized Cross Entropy Method

[Github](https://github.com/ADGEfficiency/cem) - [blog post](https://adgefficiency.com/cem/).

CEM to learn CartPole and Pendulum control problems.  Parallelized across processes and in `numpy`.

```bash
$ python cem.py pendulum --num_process 6 --epochs 15 --batch_size 4096
```
