---
title: 'The Space Between Money and the Planet'
date: 2021-10-05
categories:
  - Energy
excerpt: What is the opportunity cost of saving carbon with battery storage?

---

<center>
<img src="/assets/space-between/hero.png">
</center>

---

I've had this question in the back of my mind for a while - **is their an environmental cost of making money from battery storage?**

We care both about making money and saving carbon - ideally we can do both at the same time.

A world where we made the most amount of money would ideally also be a world where we saved the most amount of carbon - clean electricity is cheap, and dirty electricity is expensive.  We can operate our battery focusing on making money, and know that we will save carbon.

In the opposite world, where dirty electricity is cheap and clean electricity is expensive, there would be an opportunity cost for saving carbon.  There would be situations where you could reduce the environmental benefit of operating your battery in order to make more money.

The delta between these two worlds can be measured in what we care about - money and carbon.  

The table below shows a scenario where the there is a trade off between optimizing for money or carbon.

Choosing to prioritize money over carbon means we make `$150` more than if we optimized for carbon.  

|                 | Optimize for Money   | Optimize for Carbon   | Delta |
| --------------  | -------------------- | --------------------- | ----- |
| Money saved $   | 200                  | 50                    | 150   |
| Carbon saved tC | 10                   | 20                    | 10    |
|                 |                      | **Carbon Price $/tC**      | 15    |

Estimating the size of this delta also allows us to calculate a carbon price of `15 $/tC` - the ratio of the money gained by optimizing for money and the carbon saving gained by optimizing for carbon.  

This carbon price gives some indication about the level of support (via a carbon tax on electricity market participants) required to counteract the misalignment between our price and carbon signals.

The purpose of this post is to estimate the space between two worlds - one where we optimize for money, the other where we optimize for carbon. 


## Reproduce these results

You can grab the results from the [adgefficiency-public](https://s3.console.aws.amazon.com/s3/buckets/adgefficiency-public) S3 bucket.

I recommend installing the `awscli` Python package and syncing to a folder named `notebooks/results`:

```bash
$ pip install awscli
$ aws s3 sync s3://adgefficiency-public/space-between/results ./notebooks/results
```

You can use the codebase that generated these results by cloning energypy-linear codebase at commit [1d19e3e1](https://github.com/ADGEfficiency/energy-py-linear/tree/1d19e3e11d2df48f3007c682afd437159651d062), and running a `make` command:

```bash
$ git clone https://github.com/ADGEfficiency/energy-py-linear
$ cd energy-py-linear
$ git checkout 1d19e3e11d2df48f3007c682afd437159651d062
$ make space-between
```

This `make` command will:

If you can't get any of this working feel free to email me at [adam.green@adgefficiency.com](mailto:adam.green@adgefficiency.com).

## Methods

This work uses two of my tools:

1. [nem-data](https://github.com/ADGEfficiency/nem-data) - a Python CLI for downloading Australian electricity market data,
2. [energypy-linear](https://github.com/ADGEfficiency/energy-py-linear) - a Python library for optimizing the dispatch of batteries operating in price arbitrage.

The battery model is a mixed-integer linear program built in PuLP, that optimizes the dispatch of a battery with perfect foresight.  The only value stream available to the battery is the arbitrage of electricity from one interval to another.


### Optimize for price or carbon

The battery model can be optimized on two objectives - either price or carbon. 

Optimizing for price means the battery will import electricity from the grid at low prices and export it during high prices.

Optimizing for carbon means the battery will import electricity from the grid at low marginal carbon intensity and export it during high marginal carbon intensity, leading to a carbon saving.

Below is are two examples - the left optimizing a battery for money, on the right optimizing a battery for carbon:

![](/assets/space-between/panel.png)

<center><figcaption>Comparing the optimization for price (left) and carbon (right).</figcaption></center>
<br />

An important sense check when looking at optimized battery profiles is that the battery ends the period on zero charge - as we do for both above.


### Datasets

Our two objectives of price and carbon require two signals - a price signal and a carbon signal.

For this study I'm using data from 2014 to end of 2020:

- price signal = 30 minute trading price in South Australia,
- carbon signal = 5 minute NEMDE data + NEM generator carbon intensity in South Australia.

The 30 minute price data is upsampled to 5 minutes to align with the carbon data. The battery is optimized in monthly blocks.


## Results

The chart below shows the results grouped by month:

- the price delta - the difference between the optimize for money and optimize for carbon worlds in thousands of Australian dollars per month,
- the carbon delta - the difference between the optimize for money and optimize for carbon worlds in term of tons of carbon savings per month,
- the monthly carbon price - the ratio of our price to carbon deltas.

![](/assets/space-between/monthly.png)

<center><figcaption>Monthly deltas from 2014 to end of 2020.</figcaption></center>

The chart below shows the results grouped by year:

![]({{ '/assets/space-between/annual.png' }})

<center><figcaption>Annual deltas from 2014 to end of 2020.</figcaption></center>

The key takeaway from the plot above is that a carbon price of below `100 $/tC` would be enough to fully incentivize batteries to maximize both their economic and carbon savings.

This carbon price would be applied in proportion to the carbon intensity of the electricity produced by each market participant.

<p align="center"><img src="/assets/space-between/scatter.png"></p>

<center><figcaption>Monthly deltas from 2014 to end of 2020..</figcaption></center>

## Exploring this carbon price metric

Imagine we have a system where our deltas are `$500` and `50 tC`, giving a carbon price of `$/tC 10`.


What this means is that if we could receive `$500` of income by dispatching in a carbon friendly way, this would be enough to cancel out the `$500` we could have made ignoring carbon and optimizing for price.

|                   | Scenario One | Scenario 2 |
| --------------    | -----        | ---------- |
| Money saved $     | 500          | 400        |
| Carbon saved tC   | 50           | 50         |
| Carbon Price $/tC | 10           | 8          |


This carbon price is a break-even carbon price for the battery - it is what we would have to pay the market to offset the lost revenue of `$500`.


### Effect of efficiency & forecast error on carbon price

If we make mistakes on the dispatch of our battery, we may end up with a delta of `$400` and `50 tC`, giving a carbon price of `$/tC 8`.

Assuming that we still have a delta of `50 tC` may be a stretch (roundtrip efficiency would affect price & carbon performance equally).

I initially found this counter-intuitive - for years I had always thought that high electricity prices would be a good thing for helping to achieve carbon saving from battery storage - there is a reason that the

I'm so used to the idea that high electricity prices are a good thing - however here we see that the more expensive the electricity, the higher of carbon price to counter the value of the that electricity.

The dirtier the electricity, the lower carbon price we need to incentive to save the same amount of money.  This is similar to the value of energy efficiency - highest when replacing dirty or inefficient plant.


## Criticism

There are lot's of things wrong with this work - below are a few that I've thought of.


### Choice of data

For this study I used the 30 minute South Australia trading price for a price signal, and the 5 minute NEMDE data for a carbon signal.

Marginal carbon - assuming we wouldn't affect the market

Price - only looking at the trading price, not including behind the meter benefits like network chargces

Using different price and carbon signals will change the results of this study - this isn't a fatal criticism but it should reinforce that this study is heavily dependent on the choice of data.

We can add to this the super generic but always relevant criticism of anything empirical - you can't use the past to predict the future.


### Lack of forecast error / perfect foresight

Optimizing with perfect foresight allows us to put an upper limit on both money and carbon savings.  In reality, a battery will be operated with imperfect foresight of future prices.

Because we are interested in the ratio between carbon & economic savings, taking the ratio of maximum carbon to maximum economic savings is hopefully useful. 

We are essentially assuming that the relative dispatch error (in % lost carbon or money) is the same for both objectives.


### Simplistic battery model, incorrect battery size

Only a very simple roundtrip efficiency applied onto battery export - in reality efficiency is proportinal

This study uses a battery configuration of 1 MW power rating with 2 MWh of capacity - other batteries have different ratios of power to energy.

### No network effects

Batteries often have access to many value streams - arbitrage of wholesale electricity is only one of them.

---

Thanks for reading!

If you enjoyed this post, check out [Measuring Forecast Quality using Linear Programming](https://adgefficiency.com/energy-py-linear-forecast-quality/), where I show how to use this same battery model to measure the quality of a forecast.
