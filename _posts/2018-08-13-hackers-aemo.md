---
title: 'A Hackers Guide to AEMO & NEM Electricity Market Data'
date: 2018-08-13
categories:
  - Energy
  - Machine Learning
excerpt:  A simple guide to data provided by AEMO for the Australia's National Electricity Market (NEM).

---

```
created: 2018-08-13, updated: 2022-12-10
```

This is a short guide to the electricity grid & market data supplied by AEMO (the market operator) for the Australian National Electricity Market (NEM).  

The NEM is Australia's electricity grid in Queensland, New South Wales, Victoria, South Australia, and Tasmania.

# Data

Information about the participants in the NEM is given in the [NEM Registration and Exemption List](https://www.aemo.com.au/-/media/Files/Electricity/NEM/Participant_Information/NEM-Registration-and-Exemption-List.xls).  

The carbon intensities for generators are given in the [Available Generators CDEII file](http://www.nemweb.com.au/Reports/CURRENT/CDEII/CO2EII_AVAILABLE_GENERATORS.CSV).

Interval data for the NEM is provided in two sources the NEM Dispatch Engine [NEMDE](http://nemweb.com.au/Data_Archive/Wholesale_Electricity/NEMDE/) and the Market Management System Data Model [MMSDM](http://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/).

## NEMDE

The NEMDE dataset provides infomation about how the grid is dispatched and price are set (including infomation about the marginal generator) in the `NemPriceSetter` XML files.  Data for each day is provided in a single ZIP file ([Example ZIP - NemPriceSetter_20220101_xml.zip](https://nemweb.com.au/Data_Archive/Wholesale_Electricity/NEMDE/2022/NEMDE_2022_01/NEMDE_Market_Data/NEMDE_Files/NemPriceSetter_20220101_xml.zip)), which contains many XML files:

```xml
# NemPriceSetter_20220101_xml/NEMPriceSetter_2022010100100.xml

<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<SolutionAnalysis>
	<PriceSetting PeriodID="2022-01-01T04:05:00+10:00" RegionID="NSW1" Market="Energy" Price="87.69011" Unit="LBBG1" DispatchedMarket="R5RE" BandNo="6" Increase="1" RRNBandPrice="23.7" BandCost="23.7" />
	<PriceSetting PeriodID="2022-01-01T04:05:00+10:00" RegionID="NSW1" Market="Energy" Price="87.69011" Unit="BW04" DispatchedMarket="R5RE" BandNo="1" Increase="-0.47368" RRNBandPrice="1" BandCost="-0.473684" />
	<PriceSetting PeriodID="2022-01-01T04:05:00+10:00" RegionID="NSW1" Market="Energy" Price="87.69011" Unit="BW03" DispatchedMarket="R5RE" BandNo="1" Increase="-0.52632" RRNBandPrice="1" BandCost="-0.526316" />
```

## MMSDM

The MMSDM provides both actual data and forecasts for a range of variables - including prices, demand and electricity flows.  

Data in the MMSDM is supplied from three different, overlapping sources:

- [CURRENT](http://www.nemweb.com.au/REPORTS/CURRENT/) - last 24 hours,
- [ARCHIVE](http://www.nemweb.com.au/REPORTS/ARCHIVE/) - last 13 months,
- [MMSDM](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/) - from 2009 until the end of last month.

Some report names can be different across sources - for example `DISPATCH_SCADA` versus `UNIT_SCADA`.

## Price Structure

The settlement price in the NEM is known as the **trading price** - it is the price that matters for what generators get paid and what customers pay.

Historically (before October 2021) it was settled on a 30 minute basis, as the average of the six 5 minute **dispatch prices** for the same interval.

## AEMO Timestamping

**AEMO timestamp with the time at the end of the interval**.  This means that `01/01/2018 14:00` refers to the time period `01/01/2018 13:30 - 01/01/2018 14:00`.  This will be true for columns like `SETTLEMENTDATE`, which refer to an interval.  Columns like `LASTCHANGED`, which refer to a single instant in time are not affected by this.

I prefer shifting the AEMO time stamp backwards by one step of the index frequency (i.e. 5 minutes).  This allows the following to be true:

```python
dispatch_prices.loc['01/01/2018 13:30': '01/01/2018 14:00'].mean() == trading_price.loc['01/01/2018 13:30']
```

The shifting also allows easier alignment with external data sources such as weather, which is usually stamped with the timestamp at the beginning of the interval.

If the AEMO timestamp is not shifted, then the following is true:

```python
dispatch_prices.loc['01/01/2018 13:35': '01/01/2018 14:05'].mean() == trading_price.loc['01/01/2018 14:00']
```

## Useful MMSDM Reports

The MMSDM links are for the reports linked below are all for [2018_05](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/DATA/).

## Actual Data

Examples for [MMSDM May 2015](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/DATA/):

- trading price (30 & 5 min electricity price) - TRADINGPRICE - [example ZIP - PUBLIC_DVD_TRADINGPRICE_201805010000.zip](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_TRADINGPRICE_201805010000.zip),
- dispatch price (5 min electricity price) - DISPATCHPRICE - [example ZIP - PUBLIC_DVD_DISPATCHPRICE_201805010000.zip](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_DISPATCHPRICE_201805010000.zip),
- generation of market participants - UNIT_SCADA - [example ZIP - PUBLIC_DVD_DISPATCH_UNIT_SCADA_201805010000.zip](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_DISPATCH_UNIT_SCADA_201805010000.zip),
- market participant bid volumes - BIDPEROFFER - [example ZIP - PUBLIC_DVD_BIDPEROFFER_201805010000.zip](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_BIDPEROFFER_201805010000.zip),
- market participant bid prices - BIDAYOFFER - [example ZIP - PUBLIC_DVD_BIDDAYOFFER_201805010000.zip](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_BIDDAYOFFER_201805010000.zip),
- demand - DISPATCHREGIONSUM - [example ZIP - PUBLIC_DVD_DISPATCHREGIONSUM_201805010000.zip),](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_DISPATCHREGIONSUM_201805010000.zip),
- interconnectors - INTERCONNECTORRES - [example ZIP - PUBLIC_DVD_DISPATCHINTERCONNECTORRES_201805010000.zip](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_DISPATCHINTERCONNECTORRES_201805010000.zip).

## Forecasts

- trading price forecast - [example ZIP](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/PUBLIC_DVD_PREDISPATCHPRICE_201805010000.zip),
- dispatch price forecast - [example ZIP](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2018/MMSDM_2018_05/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_P5MIN_REGIONSOLUTION_201805010000.zip).

# Ecosystem

A major benefit of the large AEMO dataset is the ecosystem of third parties who can build useful (and often open source) tools on top of it.

## [nem-data](https://github.com/ADGEfficiency/nem-data)

A simple CLI for downloading NEMDE & MMSDM data:

```shell-session
$ pip install nem-data
```

## [AEMO Dashboard](https://www.aemo.com.au/Electricity/National-Electricity-Market-NEM/Data-dashboard) - [interactive map](http://www.aemo.com.au/aemo/apps/visualisations/map.html)

![]({{"/assets/hacker_aemo/aemo_dashboard.png"}})

## [ElectricityMap](https://www.electricitymap.org/)

![]({{"/assets/hacker_aemo/elect_map.png"}})

## [AREMI](https://nationalmap.gov.au/renewables/)

![]({{"/assets/hacker_aemo/aremi.png"}})

## [NEMLog](http://nemlog.com.au/)

![]({{"/assets/hacker_aemo/nemlog.png"}})

## [OpenNEM](https://opennem.org.au/#/all-regions)

![]({{"/assets/hacker_aemo/opennem.png"}})

## [NEMSight](http://analytics.com.au/energy-analysis/nemsight-trading-tool/)

![]({{"/assets/hacker_aemo/nemsight.png"}})

## [gas & coal watch](https://cdn.knightlab.com/libs/timeline3/latest/embed/index.html?source=1k0rmFKexrYUBbHSb2opLO2y-f3lGx2vOUsx8uIFygro&amp;font=Default&amp;lang=en&amp;start_at_end=true&amp;initial_zoom=2&amp;height=650)

![]({{"/assets/hacker_aemo/gas_coal_watch.png"}})

# Further Reading

- [NEM on the AEMO website](https://www.aemo.com.au/Electricity/National-Electricity-Market-NEM)
- [Winds of change: An analysis of recent changes in the South Australian electricity market - University of Melbourne](https://energy.unimelb.edu.au/news-and-events/news/winds-of-change-an-analysis-of-recent-changes-in-the-south-australian-electricity-market)
- [Li, Zili (2016) Topics in deregulated electricity markets. PhD thesis, Queensland University of Technology](https://eprints.qut.edu.au/98895/)
- [Dungey et. al (2018) Strategic Bidding of Electric Power Generating Companies: Evidence from the Australian National Energy Market](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3126673)

Thanks for reading!
