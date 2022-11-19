---
title: Typical Year Forecasting of Electricity Prices
date: 2022-11-19
categories:
  - Energy
excerpt: Improve your energy project models with this simple & flexible forecasting technique.
toc: true
toc_sticky: true

---

```
created: 2022-11-19, updated: 2022-11-19
```

Energy prices are volatile - the price of gas, oil and electricity all vary widely year on year.  

Energy prices assumptions are crucial for investing & building new energy projects - the economic viability of solar, battery and energy efficiency projects depend heavily on energy prices used in business case models.

This post will show why the standard industry approaches to using energy prices for investment decisions in energy projects are hiding massive amounts of error.


# What is a Typical Year Forecast?

A typical year forecast uses historical data to create a **single, synthetic year of data**.  

This single year of data is suitable for use in **business case modelling of energy projects** - it's not suitable for short term dispatch of energy assets.

A typical year forecast has the following advantages:

- **simple to create** - no machine learning, gradients or iterative calculations,
- **interpretable** - easy to understand why one sample is selected over others,
- **realistic** - the forecast is made from actual historical data,
- **domain flexible** - can be used with any time series (not just electricity prices),
- **statistically flexible** - can use a range of statistics to define what typical means.

A typical year forecast has the following disadvantages:

- **data quantity** - requires at least 2 years of historical data,
- **domain knowledge** - requires selecting & weighting of statistics based on problem understanding.

An example is a typical metrological year (TMY) forecast, used to create a dataset of typical year of weather.  TMY forecasts are commonly used in predicting solar generation or building energy use. 

The idea & inspiration for this post came from using the [TMY forecast produced by Solcast](https://solcast.com/tmy) - thanks Solcast for the inspiration!

# Motivation - The Problem with the Standard Industry Approach

Estimating the economic performance (simple payback, IRR, NPV or rate of return on capital) of an investment in an energy project requires combining two models - **a technical model and a financial model**.

Commonly the technical model will model a single year in isolation, and is used as an input to the financial model.

The financial model must model **multiple years over time** (to model economic return over time), using the technical results as the basis for the first year with the financial inputs (such as prices) forecasted forward alongside the single year of technical results.

In the absence of forecasted energy prices across the future project lifetime, **energy prices are often modelled in a similar way to the technical model** - taking a single reference year in isolation, with these prices being forecasted forward with some assumption of inflation.

A simple example of a technical & financial model of an energy project is given below:

- a technical model outputs annual savings of `150 MWh`,
- we assume prices at `100 $/MWh` 
- captial investment estimated at `$ 25,000`.  

The technical inputs & price assumptions are then forecast forward (here without inflation) to calculate cumulative savings:

|   year |   capex |   savings_mwh |   price |   savings_$ |   cumulative_savings_$ |
|-------:|--------:|--------------:|--------:|------------:|-----------------------:|
|      0 |   25000 |           150 |     100 |       15000 |                 -10000 |
|      1 |       0 |           150 |     100 |       15000 |                   5000 |
|      2 |       0 |           150 |     100 |       15000 |                  20000 |
|      3 |       0 |           150 |     100 |       15000 |                  35000 |

It's not common to see both the project capex and savings in the same year (usually you need to build something before it gives value) - for this simple example please forgive this!

## Why Using The Most Recent Prices is Wrong

Choosing the reference year for prices is commonly done by:

- taking the most recent prices,
- taking the most full calendar year of prices,
- taking the prices that align with the technical model.

If we were setting up our model in November 2022 with a technical model based on 2019 data, we could choose:

- the most recent prices - October 2021 to September 2022,
- the most recent calendar year - January 2021 to December 2021,
- align with the technical model - January 2019 to December 2019.

Below we will demonstrate why **all of these commonly used methodologies introduce error** - variance that changes the results dramatically.

## Error of Using Recent Prices

Let's expand on our example above - instead of assuming prices of `100 $/MWh`, let's use the actual annual average electricity prices for South Australia.

The figure below shows the same financial model we introduced above with actual annual average prices:

![Project savings versus annual average electricity prices.]({{ "/assets/typical-year/f1.png" }})

**Look at the variance of these results!**  Around half of our projects lose money, with the other half being profitable.

This variance in project performance is only occurring based on *when we do our modelling* - not based on the fundamental, underlying economics of the project.  We can do better!


# Creating a Typical Year Forecast

Creating a typical year forecast requires defining what typical means.  We will **define typical as similarity** - our typical year forecast will be made of *samples of data that are most similar to all the other data*.

We can **quantify similarity by defining error** - error between **statistics measured across all our data and statistics measured across a candidate sample**.  The samples that minimize error will be selected and used in our forecast.

For our first typical year forecast, we will create a forecast based on a single statistic - **the average price within a month**.

The basic idea is as follows:

```python
#  example - creating a typical year forecast based on the mean with 5 years of historical data

#  iterate across each month in a year (12 months in total)
for each month in a year (Jan, Feb ... Nov, Dec)
 
  #  calculate one long term statistic across all 5 years for this one month
  long_term_mean = historical_data[month].mean()

  #  iterate across our historical data, selecting this one month 
  #  5 months across 5 years, all the same month
  for year in historical_data
    sample_mean = year[month].mean()

    #  calculate the error of this month versus the long term statistic
    sample_error = absolute(sample_mean - long_term_mean)

  #  select the sample with the lowest sample error
  #  this is the historical month we will use in our typical year forecast
  selected_sample = argmin(sample_errors)
```

After following this procedure, we end up with 12 monthly samples - one for each month, creating our typical year forecast.

## Typical Year Forecast for South Australian Electricity Prices

To demonstrate the idea, we will first limit ourselves to forecasting a single month - January, for electricity prices in South Australia, using 10 years of historical data.

Let's first start by calculating our long term statistic - the average price in January across the entire dataset, which is `85.45 $/MWh`.

We can then look at what the average price was in each January and calculate the error versus the long term statistic.

This leads us to selecting January 2017 as our typical month of electricity prices:

|   year | month   |   price-mean |   long-term-mean |   error-mean |
|-------:|:--------|-------------:|-----------------:|-------------:|
|   2012 | January |      25.6153 |           85.449 |     59.8337  |
|   2013 | January |      59.1246 |           85.449 |     26.3244  |
|   2014 | January |      88.8675 |           85.449 |      3.41845 |
|   2015 | January |      34.68   |           85.449 |     50.769   |
|   2016 | January |      50.2573 |           85.449 |     35.1917  |
|   2017 | January |      84.2589 |           85.449 |      **1.19009** |
|   2018 | January |     158.757  |           85.449 |     73.3081  |
|   2019 | January |     241.025  |           85.449 |    155.576   |
|   2020 | January |      83.2037 |           85.449 |      2.24526 |
|   2021 | January |      28.7008 |           85.449 |     56.7482  |

We can then forecast the remaining 11 months of the year, ending up with 12 months that make up our typical year forecast:

| month     |   year |   price-mean |   long-term-mean |   error-mean |
|:----------|-------:|-------------:|-----------------:|-------------:|
| January   |   2017 |      84.2589 |          85.449  |     1.19009  |
| February  |   2020 |      64.1771 |          71.2239 |     7.04685  |
| March     |   2021 |      68.7727 |          66.6858 |     2.08692  |
| April     |   2021 |      52.1361 |          64.1214 |    11.9854   |
| May       |   2016 |      70.6976 |          70.1316 |     0.565976 |
| June      |   2021 |      84.3886 |          81.6753 |     2.71335  |
| July      |   2021 |      91.1873 |          94.7737 |     3.58638  |
| August    |   2016 |      66.2397 |          64.8625 |     1.37717  |
| September |   2012 |      53.7977 |          54.7594 |     0.961707 |
| October   |   2012 |      50.9616 |          52.3186 |     1.35705  |
| November  |   2016 |      61.8883 |          57.3279 |     4.56045  |
| December  |   2015 |      66.8321 |          67.2765 |     0.444369 |

![Typical year forecast using the mean as a statistic.]({{ "/assets/typical-year/f2.png" }})

We can compare this typical year forecast to actual historical prices:

![Comparing our typical year forecast using the mean as a statistic to historical data.]({{ "/assets/typical-year/f3.png" }})

## Extending the Forecast With More Statistics

Above we only considered the mean when selecting a month.  The mean is a measurement of the *central tendency* of a distribution - using the mean to select a month will mean our forecast has a similar central point to the long term average.

For some energy assets, **the variance can be more important than the average**.  Variance is how *spread out* or variable prices are.  The variance of prices is important for batteries operating in wholesale arbitrage - this spread puts an upper limit on how profitable shifting of electricity from low to high price intervals can be.

Our procedure for creating a typical year forecast based on both the mean and the variance is similar to only considering the mean - we calculate two additional statistics (the long term standard deviation and the sample standard deviation), and include them in our sample error:

```python
#  creating a typical year forecast based on the mean & standard deviation

#  iterate across each month in a year
for month in (Jan, Feb ... Nov, Dec):
  #  calculate two statistics - long term mean & standard deviation
  long_term_mean = data.mean()
  long_term_std = data.std()
  #  iterate across historical data & calculate sample errors
  #  using both long term statistics
  for year in (historical data):
    sample_mean = month.year.mean()
    sample_std = month.year.std()
    sample_error = absolute(long_term_mean - sample_mean) + absolute(long_term_std - sample_std)
  #  select sample that minimizes error
  selected_sample = argmin(sample_errors)
```

Taking this approach again, we end up with our typical year forecast - different from our previous forecast where we only used the mean:

| month     |   year |   price-mean |   long-term-mean |   price-std |   long-term-std |   error |
|:----------|-------:|-------------:|-----------------:|------------:|----------------:|--------:|
| January   |   2020 |      83.2037 |          85.449  |    519.785  |        504.705  | 17.3251 |
| February  |   2018 |     109.17   |          71.2239 |    290.873  |        300.955  | 48.0282 |
| March     |   2020 |      46.9517 |          66.6858 |    225.829  |        271.301  | 65.2057 |
| April     |   2015 |      39.9493 |          64.1214 |    100.387  |         99.2508 | 25.3085 |
| May       |   2016 |      70.6976 |          70.1316 |    132.686  |        133.63   |  1.5091 |
| June      |   2021 |      84.3886 |          81.6753 |     96.1186 |        130.305  | 36.8999 |
| July      |   2015 |      73.5053 |          94.7737 |    226.191  |        236.491  | 31.5684 |
| August    |   2013 |      71.2364 |          64.8625 |     88.1036 |        103.648  | 21.9185 |
| September |   2012 |      53.7977 |          54.7594 |     62.1015 |         75.617  | 14.4772 |
| October   |   2019 |      67.3398 |          52.3186 |     92.2279 |        108.001  | 30.7947 |
| November  |   2019 |      50.8623 |          57.3279 |     88.3317 |        109.014  | 27.1474 |
| December  |   2013 |      79.5734 |          67.2765 |    372.848  |        318.756  | 66.3892 |

We can compare our two typical year forecasts directly:

![Typical year forecast using the mean as a statistic.]({{ "/assets/typical-year/f4.png" }})

We can see that this typical year forecasting is selecting months with higher prices - including more of the tasty price spikes that makes the NEM so interesting for battery storage.

# Evaluating the Typical Year Forecast

Let's return to our original motivating example, with our additional estimate using our typical year forecast based on using the mean (show as 2052 in green):

![Typical year forecast using the mean as a statistic.]({{ "/assets/typical-year/f5.png" }})

**How great is that!** Our typical year forecast does a fantastic job of cutting through the variance - modelling our project right in the middle of the high variance estimates we get when taking the traditional, industry standard approaches of using historical price data.

No longer are we slaves to the cruel master of time - as the years go by, our estimation of project economics will stay stable and consistent, rather than varying wildly based on when we are doing our modelling.

# Discussion

## Challenges

### Data Quantity

This methodology requires multiple years of data - if we only have access to a single year, this method is not appropriate.

### Alignment

One problem that arises when concatenating interval data from different time periods together is alignment at the intersection - the sample below from the typical year forecast produced above shows the issue - our forecast jumps from Tuesday in January 2017 to Friday 2020:

| interval-start      | original-timestamps   |   price |   day-of-week |
|:--------------------|:----------------------|--------:|--------------:|
| 2052-01-31 23:50:00 | 2017-01-31 23:50:00   |   39.52 |             1 |
| 2052-01-31 23:55:00 | 2017-01-31 23:55:00   |   39.52 |             1 |
| 2052-02-01 00:00:00 | 2020-02-01 00:00:00   |  299.2  |             5 |
| 2052-02-01 00:05:00 | 2020-02-01 00:05:00   |  299.2  |             5 |

This misalignment will cause issues with the incorrect number of weekdays or weekends in a year - important as energy demand and price has strong weekly seasonality.

This alignment problem also occurs when you don't use a typical year forecast - for example if you use price data from 2022 with technical data from 2010.

### Domain Expertise

There is some amount of domain expertise required to setup a typical year forecast - primarily in defining the appropriate statistics.  

Using multiple statistics can also require weighting - for example if the standard deviation is orders of magnitude higher than the mean, we may want to weight the mean higher.

## Extensions & Improvements

### Higher Frequency Sampling

In the examples above we have selected samples on a monthly basis - it is possible to instead select samples on a different frequency, such as week of the year (52 weeks) or day of the year (365 days).

### More Statistics

One advantage of this methodology are flexibility of statistics we choose - unlike a loss function for a neural network, they do not need to be differentiable. 

For example, we could use statistics like:

- mean, median, mode,
- number of time periods above a threshold price,
- number of negative prices.

This is an exciting feature of typical year forecasting - the flexibility and simplicity of using any statistic that aligns with the features your technical and financial models need to be accurate on.

# Summary

In this post we introduced *typical year forecasting* - a flexible, powerful forecasting method suitable for use in energy project business case modelling.

Typical year forecast address a hidden flaw in the price assumptions commonly used in industry - the large errors introduced by using recent price data.

A typical year forecast addresses these issues by selecting historical price data that is most similar to all the historical data.

Typical year forecasts have the following advantages:

Typical year forecasts have the following disadvantages:

Further extensions on the methods shown above include:

- higher frequency sampling,
- using a variety of statistics to define similarity.

---

Thanks for reading!  If you enjoyed this post, make sure to check out [Measuring Forecast Quality using Linear Programming](https://adgefficiency.com/energy-py-linear-forecast-quality/).

You can find the source code to reproduce the typical year forecasting results at [adgefficiency/typical-year-electricity-price-forecasting]().
