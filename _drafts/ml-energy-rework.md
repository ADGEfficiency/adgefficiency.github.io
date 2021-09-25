---
title: 'Machine Learning in Energy New'
date: 2021-08-21
classes: wide2
toc: true
toc_sticky: true
categories:
  - Energy
  - Machine Learning
excerpt:  A guide to machine learning for energy professionals.
redirect_from: 
  - /machine-learning-in-energy-part-one/
  - /machine-learning-in-energy-part-two/

---

> The data may not contain the answer - JOHN TUKEY

This post is what I want professionals to know about machine learning and energy:

- **energy professionals** wanting to understand how machine learning can solve energy industry problems,
- **data professionals** wanting to understand how their skills can be used to solve energy problems.

This is the third major rework of this post. I've refactored some of the content out of this post - some of it has ended up in [AI, Machine Learning and Deep Learning ](https://adgefficiency.com/ai-ml-dl/).


# What I want energy professionals to know

## Importance of data

The most important part of any machine learning project is the data - energy machine learning projects are no different.

Data is the fuel of machine learning - no matter how powerful an engine, fuel is required to make it work.

A common form of data in energy is time series data (often called interval data).  

Energy professionals will already be familiar with the importance of time - energy demand and prices all depend heavily on the time of day, day of week and month of the year.

Data quality is not only about quantity of data.  More data is almost always better (because you can choose not to use it), but data quantity is not the only thing to get right.

In addition to the amount of data, data quality is important.  Alongside common data quality issues such as missing values or duplicates, time series data quality is dependent on seasonality and stationarity.

Seasonality is the regular, cyclic patterns in time series data. Energy time series data often exhibits multiple levels of seasonality:

- monthly,
- weekly,
- daily.

Stationarity refers to whether a time series is fundamentally changing or not.  A more technical way to describe a non-stationary time series is to say that the data generating process has changed.

A good example of this is the Australian electricity market switching from 30 minute settlement to 5 minute settlement.  This kind of fundamental change in the market structure

More examples of a non-stationary time series?  Any energy system undergoing a transition to renewables, or a warming planet.


## What can machine learning do?

If you aren't familiar with ML -> [AI, Machine Learning and Deep Learning ](https://adgefficiency.com/ai-ml-dl/).

It's always important to remember that the value of data is more than only machine learning.

For many business problems, doing putting data into a database and doing analytics is enough to provide value.  If this is the case for you - lucky you!

It's important to maximize the value of data - data can often be used in many different ways - machine learning is only one.

Two common uses of machine learning are prediction and control.

The first group of tasks machine learning excels at is prediction.

Examples of using machine learning for prediction in energy include:
- forecasting 

Another group of tasks machine learning excels at is control.

Examples of using machine learning for control in energy include:
- batteries
- data centers


# What I want machine learning people to know


## The energy transition

The energy industry is defined by transitions.  

Our current energy transitions is towards distributed, small scale and intermittent renewable generation.

The intermittent side introduces uncertainty on generatiors

The distributed nature introduces uncertainty on the demand side along with a higher dimensional action space.

More challenging control problems - storage, flexible demand - on top of a vast number of existing control 



## Importance of time + sequence models

- lots of data, but much still not digitized (5 min frequency)

Time important in energy

Applying attention is a goldmine

- heavy seasonality & trends (non stationary)


## Availability of cost functions

One of the best things about working in energy is the availability of cost functions. 

Technical people in energy are lucky to have access to cost functions that they can calculate (or at least estimate) - two of the most useful cost functions are money and carbon.

The importance of a cost function won't be lost on machine learning people - optimizing a cost function is how neural networks learn.

Compare energy to domains such as driving or language - which lack clear cost functions.  It is not trivial on how to quantify:

- the quality of driving (taking into account speed and safety),
- the quality of speech (taking into account clarity and accuracy),
- the quality of government policy.

Energy is lucky because we have good cost functions that can be quantified that correlate well with what businesses want (more money or less carbon).

Businesses are naturally focused on money - the only hope for proper carbon optimization is a carbon tax.


# What I want everyone to know

## Non-linear optimization on top of the linear optimization cake

Cost functions form the basis of many optimization algorithms - including the mixed integer linear programming that is used heavily throughout the energy industry.

We are again lucky to have many systems that can be modelled (or approximated well) as mixed integer linear programs.  Linear programming can be used to optimize the dispatch of assets like gas generation, hydro or battery storage.

Take your existing optimization work flows and feed them better inputs/forecasts

Linear programs offer access to the global optimum - convergence to the best solution, everytime. Non-linear optimization is iterative, messy and only locally optimal.

An excellent example of the power of linear programming is the work of Kantor et al (2020) in [A Mixed-Integer Linear Programming Formulation for Optimizing Multi-Scale Material and Energy Integration](https://www.frontiersin.org/articles/10.3389/fenrg.2020.00049/full), which optimzes both the sizing and dispatch of energy assets like ? and ?.

However linear methods can only be pushed so far - at some point the fundamental non-linearity of the world gets in the way.  

Take for example mass & energy balances - enforcing constraints on one is fine, but when you start to build models that balance both mass and energy, you will end up with a fundamentally bilinear system.

```
mass * temperature
```

A classic example of this is dispatch of a battery operating in a price arbitrage scenario.

While it's quite easy to use linear methods with perfect foresight of prices, if 

Optimization or prediction problems
- ml = non-linear = big opportunity,
- both things ML does well
- time series forecasting
- sequence based deep learning,

Prediction + linear optimization = a natural way to fit ml into existing energy optimization problems

Case for ML
- more data in future
- compute cheap


## Not a silver bullet

Despite investors, AI/ML is not solution to everything

Data = more than ML

Marginal gains in some areas - unlikely to revolutionize anything

Expectation management

The unfortunate truth for technical people is that the climate problem is much larger than technology 

- policy (carbon tax) still remains the most powerfurl and most underutilized tool we have

For many businesses, both economic and environmental gains can occur with the implementation of automation or basic data collection.

Futures that are difficult to imagine:

- a 100% robot operated power station

Possibilities that are open today:

- use deep learning for prediction


# Further

https://www.climatechange.ai/summaries

- [AI, Machine Learning and Deep Learning](https://adgefficiency.com/ai-ml-dl/)

Optimization of Energy Systems, Victor Zavala https://youtu.be/ngiQCRJWZws

Warren Powell, "Stochastic Optimization Challenges in Energy" - https://youtu.be/dOIoTFX8ejM
