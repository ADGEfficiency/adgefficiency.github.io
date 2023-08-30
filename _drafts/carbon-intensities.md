---
title: Interval Level Average Carbon Intensity in the NEM
date: 2022-12-04
categories:
  - Energy
excerpt: Estimating the interval level average carbon intensity of electricity in the NEM.
toc: true
toc_sticky: true

---



## Part One - Code

polars
- first do one month in isolation, to show the features of polars
- build up to a single query
- then use many queries with select

then do with demand

behind meter generation???

then do interconnectors


## Part Two Analysis

- averages versus time of day / week
- best / worst days
- compare difference between generation + demand (table)
- interconnectors


## Part Three Discussion

Ignored some generators

Ignored storage

Behind the meter generation



Future
- compare dispatch methods
- overlay colors onto the time periods you would dispatch from (dirtiest by average, dirtiest by absolute carbon etc)
- side by side for each


---

This post shows how to estimate the interval level average carbon intensity of electricity in the NEM:

- by interval level, we mean the carbon intensity will be calculated on a 5 minute basis - one value for each 5 minute interval,
- by average, we mean that the carbon intensity will be averaged across all generators in the grid, weighted by their generation.

# Calculating Carbon Intensity

The fundamental way to calculate a carbon intensity is:

$$intensity=carbon/energy=tC/MWh$$

This simple equation underpins all calculations of carbon intensity.  The complexity comes in how we choose to quantify `carbon` and `energy`.

# Average Carbon Intensity in the NEM

After installing the packages in [repo](), we can download the raw generation data for the NEM using `nemdata`, which will download the UNITSCADA dataset into `~/nem-data/data`:

```shell-session
$ nemdata -t unit-scada -s 2019-01 -e 2021-12
$ ls ~/nem-data/data/unit-scada
```

We can then load this data with `pandas`, do some column renaming and take a look:

```python
import nemdata
data = nemdata.loader(['unit-scada'])['unit-scada']
data = data.rename({
    'SCADAVALUE': 'power_mw',
})
data = data['interval-start', 'DUID', 'power_mw']
print(data.shape)
print(data.sample())
```


