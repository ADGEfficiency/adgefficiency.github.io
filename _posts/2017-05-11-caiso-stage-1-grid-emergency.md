---
title: 'CAISO Stage 1 Grid Emergency'
date: 2017-05-11T23:25:29+00:00
author: Adam Green
layout: post
permalink: /caiso-stage-1-grid-emergency-energy-insights/
categories:
  - Energy Insights
---

[On May 3rd 2017 the California grid experienced its first Stage 1 grid emergency in nearly a decade.](https://www.rtoinsider.com/caiso-grid-emergency-natural-gas-demand-42802/)

The reasons for this emergency notice were:
- a 330 MW gas-fired plant outage
- 800 MW of imports that were unavailable
- a demand forecasting error of 2 GW

A Stage 1 grid emergency doesn't mean a blackout - it forces the ISO (Independent System Operator) to dip into reserves and slip below required reserve margins.  It allows CAISO to access interruptible demand side managment programs.

## demand forecasting error of 2 GW

This is a massive absolute error - equivalent to a large power station!

Demand on the 11th of May for the same time period was [around 28 GW](http://www.caiso.com/outlook/SystemStatus.html) - giving a relative error of around 7%.

This isn't actually an error in forecasting demand - it's the effect of distributed & small scale solar appearing to the ISO as reduced demand.  This is one of the challenges of our energy transition - distributed generation introducing uncertainty in electricity demand. 

Improving time series forecasting is one of the key areas where [machine learning can contribute towards decarbonization.](http://adgefficiency.com/machine-learning-in-energy-part-one/)  In markets where generators must forecast their output ahead of time, errors in forecasting can lead to imbalance charges.  The risk of imbalance exists for traditional fossil fuel generators with unplanned maintenance, and for renewable generators with both maintenance and variable weather conditions. 

## lack of flexibility

> It was unusual that the issues began developing around the peak, and demand wasn’t ramping down much, but solar was ramping off faster than what the thermal units online at the time could keep up with in serving load - CAISO spokesperson Steven Greenlee

In a previous post I highlighted the [concept of flexibility](http://adgefficiency.com/complexity-of-a-zero-carbon-grid/).  This event demonstrates why flexibility is important for managing a modern electric grid.

Even if you have the capacity `MW` you might not have the flexibility `MW/min` to cope with the intermittent nature of renewables.

Interruptible demand side management programs are only called upon in a Stage 1 emergency.  Prior to this thermal units are used to balance the system.  Using flexible assets as a first step can be a cleaner and cheaper way to support the grid.  Flexible demand requires no fuel and is essentially a capex-lite power station, as it's already been built.

Thanks for reading!
