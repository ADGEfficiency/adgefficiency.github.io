---
title: 'Q = m Cp dT'
date: 2016-10-31
categories:
  - Energy Basics
excerpt: The equation I used the most as an energy engineer.

---
Heat transfer is a fundamental energy engineering operation.  Hot water loops are commonly used to transfer heat in district heating networks and on industrial sites.  

The capital & operating cost of many hot water loops are higher than they should be.  This post will explain why this is happening in the context of the foundational energy engineering equation `Q = m * Cp * dT`.

![Figure 1 – A simple hot water loop]({{ "/assets/q_mcdt/hot-water-loop-1.png"}})
**Figure 1 – A simple hot water loop**

This equation shows how to calculate heat transfer in our hot water loop:
```
Q = m * Cp * dT

heat = mass flow * specific heat capacity * temperature difference

kW = kg/s * kJ/kg/°C * °C
```
The mass flow rate `m [kg/s]` is a measurement of the amount of water flowing around the hot water loop.

The specific heat capacity `Cp [kJ/kg/°C]` is a thermodynamic property specific of the fluid used to transfer heat. We could manipulate the specific heat capacity only by changing the fluid used in the loop.  

Water is a good fluid choice for cost and safety considerations.  The specific heat capacity of water does vary with temperature but for the scope of a hot water loop it is essentially constant.

The temperature difference `dT [°C]` is the difference in temperature before and after heat transfer.

Optimization of a hot water loop requires correctly setting the flow rate and temperature.  We could use a high mass flow rate and low temperature difference.  We could also use a low mass flow rate with a high temperature difference.

A low mass flow with high temperature difference is optimal and will reduce our capital  & operating costs.   A low mass flow rate minimizes the amount of electricity required to pump water around the loop.

A high temperature difference leads to:

- increase in the maximum capacity of the loop to deliver heat.  Pipe size limits the capacity of the loop by limiting the maximum flow rate.  More heat can be transferred at the maximum flow rate by using a larger temperature difference.

- maximises heat recovery from CHP heat sources such as jacket water or exhaust.

- maximises electric output from steam turbine based systems by allowing a lower condenser pressure.

The capital cost benefit comes from being able to either transfer more heat for the same amount of investment or to install smaller diameter pipework.

The operating cost benefit arises from reduced pump electricity consumption and increased CHP system efficiency.

Thanks for reading!
