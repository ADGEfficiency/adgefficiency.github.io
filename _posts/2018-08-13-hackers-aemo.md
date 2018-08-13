---
title: 'A hackers guide to AEMO data'
date: 2018-08-13
categories:
  - Energy
excerpt:  A simple guide to the Australian electricity grid data provided by AEMO.

---

This is a short guide to Australian electricity grid data supplied by AEMO (the market operator) for the NEM (the grid in Queensland, New South Wales, Victoria, South Australia, and Tasmania).

## AEMO timestamping

AEMO timestamp their data with the time **at the end of the interval**.  This means that `01/01/2018 14:00` refers to the time period `01/01/2018 13:30 - 01/01/2018 14:00`.

Personally I shift the AEMO time stamp backwards by one step of the index frequency (i.e. 5 minutes).  This allows the following to be true

```python
dispatch_prices.loc['01/01/2018 13:30': '01/01/2018 14:00'].mean() == trading_price.loc['01/01/2018 13:30']
```

If the AEMO timestamp is not shifted, then the following is true

```python
dispatch_prices.loc['01/01/2018 13:35': '01/01/2018 14:05'].mean() == trading_price.loc['01/01/2018 14:00']
```

Which personally I find less inutitive.  The shifting also allows eaiser alignment with weather data which is stamped as measured.

## Price structure

The wholesale electricity price is known as the **trading price** - a half hourly price for electricity.  The trading price is the average of the 6 **dispatch prices** that occur within a half hour.

AEMO provide both actual data and forecasts.  

Data from AEMO is supplied in three forms

- [CURRENT](http://www.nemweb.com.au/REPORTS/CURRENT/) - last 24 hours
- [ARCHIVE](http://www.nemweb.com.au/REPORTS/ARCHIVE/) - last 13 months
- [MMSDM](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/) - from 2009 until present

## Useful reports

The MMSDM links are for `2018_05`.  There are many more useful reports that have data for interconnector flows, demand and market settlement.

### actual data

- trading price (30 min electricity price) - [MMSDM](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_TRADINGPRICE_201805010000.zip)
- dispatch price (5 min electricity price) - [MMSDM](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_DISPATCHPRICE_201805010000.zip)
- generation of market participants - [MMSDM](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_DISPATCH_UNIT_SCADA_201805010000.zip)

### forecasts

- trading price forecast - [MMSDM](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/PUBLIC_DVD_PREDISPATCHPRICE_201805010000.zip)
- dispatch price forecast - [MMSDM](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_P5MIN_REGIONSOLUTION_201805010000.zip)

## Ecosystem

A major benefit of the large AEMO dataset is the ecosystem of third parties who can build useful tools from it.  

### [AEMO dashboard](https://www.aemo.com.au/Electricity/National-Electricity-Market-NEM/Data-dashboard)

![]({{"/assets/hacker_aemo.png"}})

### [electricityMap](https://www.electricitymap.org/)

![]({{"/assets/elect_map.png"}})

### [nemlog](http://nemlog.com.au/)

![]({{"/assets/nem_log.png"}})

### [opennem](https://opennem.org.au/#/all-regions) - [github](https://github.com/opennem/)

![]({{"/assets/opennem.png"}})

### [nemsight](http://analytics.com.au/energy-analysis/nemsight-trading-tool/)

![]({{"/assets/nemsight.png"}})

## Further reading

- [Winds of change: An analysis of recent changes in the South Australian electricity market - University of Melbourne](https://energy.unimelb.edu.au/news-and-events/news/winds-of-change-an-analysis-of-recent-changes-in-the-south-australian-electricity-market)
- [Li, Zili (2016) Topics in deregulated electricity markets. PhD thesis, Queensland University of Technology](https://eprints.qut.edu.au/98895/)
