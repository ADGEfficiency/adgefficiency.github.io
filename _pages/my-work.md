---
title: My Work
layout: archive
permalink: /my-work/
classes: wide
sidebar:
  nav: "top"
author_profile: true

---

Energy engineer turned data professional.  My work is focused on using data to prevent climate change.

I am interested in the different outcomes we get when we design and optimize energy systems for money or for the environment - the space that exists between economic and carbon optimization.

Currently I'm working as a Lead Data Scientist at Meridian Energy.

I'm also passionate about teaching & mentoring data professionals -- [I offer mentoring here](https://mentorcruise.com/mentor/AdamGreen/). 

## Energy Data Science

### Linear Programming for Energy Systems 

[energy-py-linear](https://github.com/ADGEfficiency/energy-py-linear) -- A Python library for optimizing the dispatch of energy assets with mixed-integer linear programming, including batteries, electric vehicles and CHP generators:

```bash
$ pip install energypylinear
```

[Introductory blog post](https://adgefficiency.com/intro-energy-py-linear/) -- [measuring forecast quality](https://adgefficiency.com/energy-py-linear-forecast-quality/) -- [source code](https://github.com/ADGEfficiency/energy-py-linear).

### Australian Electricity Grid Data

[nem-data](https://github.com/ADGEfficiency/nem-data) -- a Python CLI for downloading data for Australia's National Energy Market (NEM):

```bash
$ pip install nemdata
```

[A Hackers Guide to AEMO Data](https://www.adgefficiency.com/hackers-aemo/) is a developer focused guide to the electricity market data provide by AEMO for the NEM.

### UK Electricity Grid Data

[Elexon API Web Scraping using Python](https://www.adgefficiency.com/elexon-api-web-scraping-using-python/) and [What is the UK Imbalance Price?](http://www.adgefficiency.com/what-is-the-uk-imbalance-price/).

### Space Between Money and the Planet

[Blog post](https://adgefficiency.com/space-between-money-and-the-planet/) -- [source code](https://github.com/ADGEfficiency/space-between-money-and-the-planet).

Demonstrating the existence of a tradeoff between monetary gain and carbon emissions reduction in the dispatch of electric batteries.

![](/assets/space-between-2023/annual.png)

### Reinforcement Learning for Energy Systems

[Blog post](https://www.adgefficiency.com/energy_py-reinforcement-learning-for-energy-systems/) -- [Github](https://github.com/ADGEfficiency/energy-py) -- [DQN debugging](https://www.adgefficiency.com/dqn-debugging/), [hyperparameter tuning](https://www.adgefficiency.com/dqn-tuning/) and [solving](https://www.adgefficiency.com/dqn-solving/).

![]({{"/assets/dqn_solving/fig1.png"}})

## Optimization

### Soft Actor Critic Reimplementation

[Source code](https://github.com/ADGEfficiency/sac) - implementation of the Soft Actor Critic (SAC) reinforcement learning algorithm. SAC is an off policy algorithm capable of both continuous and discrete policies.

### World Models Reimplementation

[Blog post](https://adgefficiency.com/world-models/) -- [source code](https://github.com/ADGEfficiency/world-models) - [references & resources](https://github.com/ADGEfficiency/rl-resources/tree/master/world-models).

Implementation of the 2017 classic paper.  Variational auto-encoder, mixed density networks and evolutionary optimization learn to race a car from pixels.

![]({{"/assets/my-work/world.png"}})

### Parallelized Cross Entropy Method

[Source code](https://github.com/ADGEfficiency/cem) -- [blog post](https://adgefficiency.com/cem/).

CEM to learn CartPole and Pendulum control problems.  Parallelized across processes and in `numpy`.

```bash
$ python cem.py pendulum --num_process 6 --epochs 15 --batch_size 4096
```

### Evolutionary Optimization

[Source code](https://github.com/ADGEfficiency/evolution) -- evolutionary optimization on 2D optimization problems.

<center>
    <img src="/assets/my-work/rastrigin-pycma.gif" style="width: 50%;">
</center>

### Dynamic Programming

[Source code](https://github.com/ADGEfficiency/gridworld) -- visualizing dynamic programming using `pygame`.

## Data Science Education

### Data Science South

[A sister website](https://datasciencesouth.com/) focusing on data professional (analyst, engineer, scientist) education.

![]({{"/assets/my-work/dss.png"}})

### Teaching Monolith

[Data science teaching materials](https://github.com/ADGEfficiency/teaching-monolith).

![]({{"/assets/my-work/monolith.png"}})

## NLP

### climate-news-db

A dataset of climate change newspaper articles -- [app](https://www.climate-news-db.com/) -- [source code](https://github.com/ADGEfficiency/climate-news-db).

![]({{"/assets/my-work/db.png"}})

### Creative Writing with GPT2

Fine-tune a base GPT2 model for your favourite authors -- [source code](https://github.com/ADGEfficiency/creative-writing-with-gpt2).

![]({{"/assets/my-work/creative.png"}})
