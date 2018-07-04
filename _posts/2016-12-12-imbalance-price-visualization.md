---
title: Imbalance Price Visualization
date: 2016-12-12
categories:
  - Imbalance Price Forecasting
excerpt: Plotting using matplotlib.

---

This post is part of a series applying machine learning techniques to an energy problem.  The goal of this series is to develop models to forecast the UK Imbalance Price.  
- Introduction - [What is the Imbalance Price](http://adgefficiency.com/what-is-the-uk-imbalance-price/)
- Getting Data - [Scraping the ELEXON API](http://adgefficiency.com/elexon-api-web-scraping-using-python/)
- Visualization - [Imbalance Price Visualization](https://adgefficiency.com/imbalance-price-visualization/)

---

Visualization is a crucial first step in data analysis.  In this post we use the visualization library in the forecasting_energy package.  The notebook where this work was done is here.

First step is to load up the clean Elexon data that we downloaded and cleaned in previous posts.

```python
elexon = pd.read_csv('~/git/forecasting_energy/data/processed/elexon/clean.csv', index_col=0, parse_dates=True)
```

The first and most simple plot is to simply plot the series.  I do this using the `plot_time_series` function.

```
plot_time_series(elexon, 'Imbalance_price [£/MWh]', fig_name='fig1.png')
```

![fig1]({{ "/assets/imba_vis/fig1.png"}}) 

**Figure 1 - A plot of the imbalance price from 2015 to 2017**

Next we look at how statistics such as mean, median and standard deviation have changed ver time.  We use the `plot_grouped` function to show how these statistics change month by month.

![fig2]({{ "/assets/imba_vis/fig2.png"}}) 

**Figure 2 - Monthly statistics of the imbalance price across 2015-2017**

```python 
plot_grouped(elexon, 'Imbalance_price [£/MWh]', fig_name='fig2.png')
```

This function can also be used with different grouping.  In Figure 3 we can see how the prices changes for each month.

```python
plot_grouped(elexon, 'Imbalance_price [£/MWh]', group_type='month', fig_name='fig3.png')
```

![fig3]({{ "/assets/imba_vis/fig3.png"}}) 

**Figure 3 - Monthly statistics of the imbalance price across 2015-2017** 

Figure 3 shows some seasonality - the price tends to be higher and more volatile in the winter and lower in the summer.  The higher level of the price is expected - in the UK demand peaks in the winter.

The `plot_grouped` function can also be used to show how the price changes across the day - shown in Figure 4 below.

```python
plot_grouped(elexon, 'Imbalance_price [£/MWh]', group_type='hour', fig_name='fig4.png')
```

![fig4]({{ "/assets/imba_vis/fig4.png"}}) 

**Figure 4 - Daily statistics of the imbalance price across 2015-2017** 

Figure 4 also shows seasonality - this time on a daily basis.  Interestingly the price is most volatile in the afternoon - although it is likely that an outlier is distoriting this (seen by the maximum price also occurs in this time period).

It can also be useful to look at the distribution of time series.  Figure 5 shows 

```python
plot_distribution(elexon, 'Imbalance_price [£/MWh]', fig_name='fig5.png')
```

![fig5]({{ "/assets/imba_vis/fig5.png"}}) 

**Figure 5 - Histogram and kernel density plot of the imbalance price** 

## correaltions to other varaibles

## autocorrealtions


