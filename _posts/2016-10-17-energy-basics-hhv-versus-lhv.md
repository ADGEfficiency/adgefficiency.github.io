---
title: 'HHV vs LHV - GCV vs NCV'
date: 2016-10-17
categories:
  - Energy Basics
---

A British engineer argues with a European engineer about what to assume for a gas boiler efficiency.

The European engineer demands they assume 89% –  the British engineer disagrees and wants to assume 80%.

Who is correct?

## Higher Heating Value vs. Lower Heating Value

In the same way that two different currencies can value the same thing with a different amount of the currency, two conventions exist for quantifying the amount of heat produced in fuel combustion `[kWh/kg]`.  These two conventions are

- higher heating value (HHV) aka gross calorific value (GCV)

- lower heating value (LHV) aka net calorific value (NCV)

Note that I use HHV/GCV and LHV/NCV interchangeably as they are in industry.

![Figure 1 - A fire-tube shell boiler]({{ "/assets/hhv_lhv/steam-boiler.jpg"}})
**Figure 1 – A fire-tube shell boiler**

These conventions arise from a practical engineering reality.  **It’s about the water vapour produced during combustion**.  In one of natures most beautiful symmetries combustion produces water vapour.  Condensing this water vapour releases a lot of energy.

The high heating value includes this energy.  The lower calorific value doesn’t include the energy released in condensing water.  This is why a gross calorific value is higher than a net calorific value.

- HHV = water vapour is condensed = more heat is recovered

- LHV = water vapour remains as vapour – less heat is recovered

The reason for the distinction is that **water vapour in combustion products is not often condensed in practice.**  

- Steam or hot water boilers – condensing water requires reducing the flue gas temperature low enough where acids present in the flue gas will also condense out and cause stack corrosion and potential failure.

- Power generation – water usually remains as a vapour as the temperature within the power turbine or pistons of the engine is too high.

## The two engineers

We return now to the argument between our European and British engineers.  Which efficiency (89% or 80%) is the correct assumption?

|Table 1 – Typical HHV and LHV efficiencies | **% HHV** | **% LHV**|
|-------|
|Gas boiler | 80 | 89|
|Gas engine (2 MWe) | 38	| 42|
|Gas turbine (5 MWe) | 28 | 31|

The answer is that it depends on how the efficiency will be used.  A common calculation is to calculate the gas consumption associated with supply heat from a gas boiler.  If we then want calculate the cost of this gas, we multiply by a gas price.  

```
annual gas consumption = annual heat consumption / gas boiler

annual gas cost = annual gas consumption * gas price
```

The gas price will be of the form `cost / energy [£/MWh]`.  The MWh can be given on either a HHV or LHV basis.  The correct way to specify a gas price is therefore either `£/MWh HHV` or `£/MWh LHV`.  This leaves no room for misunderstanding.

To calculate the cost using a UK gas price we would want to have assumed an efficiency of 80 % HHV.  This is because UK gas prices are given on an HHV basis.  Either convention can be used as long as all of our fuel consumptions, efficiencies and energy prices are given on the same basis.  Consistency is crucial.

Most data sheets will specify gas consumption or efficiency on an LHV basis.  If you are working in a country that prices fuel on an HHV basis (such as the UK) you will need convert this gas consumption to an HHV basis before you can calculate the cost.

## Summary

Using a fuel consumption on an LHV basis with a HHV gas price can lead to a significant underestimation of fuel costs.

Using an efficiency or gas consumption straight off a data sheet can easily wipe out the typical margins expected on energy sale contracts.  It can also tip a project IRR below the hurdle rate.

Best practice is to always be specific HHV or LHV when working with fuel consumptions, efficiencies and prices.  **Be the engineer who always writes ‘MWh HHV’ and £/MWh HHV!**

Thanks for reading!
