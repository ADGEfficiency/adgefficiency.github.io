---
title: 'Average versus Marginal Carbon Emissions'
date: 2017-02-02
categories:
  - Energy Basics
excerpt: Many professionals are getting this wrong. 

---

Calculating carbon savings is fundamental to fighting climate change.  Knowing a project's carbon saving allows us estimate how effective (measured in `$/tC`) a project is at saving carbon.  

Many professionals are getting the estimation of carbon savings wrong.  **I know because I used to make this mistake.**

This allows different technologies to be compared on the same basis.  This value can also be compared with a carbon price (also measured in `$/tC`) to determine if a project is worth doing (relative to what society thinks the cost of carbon in `$/tC` is).

So where are people going wrong?  **The mistakes is misunderstanding the difference between average and marginal carbon emissions**.

Average carbon emissions are calculated using the total carbon emissions and total amount of electricity generated.  This average is taken across all generators.

|Table 1 – Calculation of average carbon intensity for Base Case|
|---|
|Carbon emissions|	tC|	83,330|
|Electricity generated|	MWh|	182,827|
|Carbon intensity|	tC/MWh|	0.456|

This average carbon intensity can be used to calculate carbon savings.  If a project saved `2 MWh`, we would calculate a saving of`2 * 0.456 = 0.912 tC`.  This is wrong!

```
electricity_saved = 2 MWh

carbon_intensity = 0.456 tC/MWh

carbon_saving = 0.912 tC
```

In a market based electricity system, the reduction in generation is not averaged across each generator.  The reduction occurs in one plant – the marginal generator.  Let’s run through an example.

Suppose we have a grid where electricity is supplied by either wind or coal (the Base Case).  If we save 1 GW of electricity, the generation of the coal plant will reduce by 1 GW (Case 1).

An ideal electricity market will reduce output on the most expensive plant, and keep cheaper plants running at full load.  

![Figure 1 & 2]({{ "/assets/avg_marginal/fig1_2.png"}})
*Figure 1 & 2 – The effect of saving 1 GW of electricity.  Note that the generation from wind is unchanged.*

|Table 2 – Results for the Base Case & Case 1|
|---|
|||Base Case|	Case 1|	Saving|
|Wind|	MWh|	91,256|	91,256|	0|
|Coal|	MWh|	91,571|	67,571|	24,000|
|Total|	MWh|	182,827|	158,827|	24,000|
|Carbon emissions|	tC|	83,329|	61,489|	21,840|
|Carbon intensity|	tC/MWh|	0.456|	0.387|	0.910|

Our carbon saving is equal to `1 GW` multiplied by the carbon intensity of the marginal plant.  If we were to use the average grid carbon intensity (`0.456 tC/MWh`) we calculate a daily carbon saving of only `21,840 tC`.

How do we know what the marginal generator is?  It’s likely to be the most expensive generator at that time.  It may not be if the plant needs to be kept on for technical reasons.  

Taking the most expensive generator is a good approximation of the marginal grid carbon intensity.  Only if your project saved more than the output of this generator would you need average across multiple marginal generators.

As renewables are characterized by low marginal costs they are the unlikely to be pushed off the grid.  

High marginal cost generators like open cycle gas turbines are usually also dirty – so your saved electricity is likely doing valuable work – and potentially more than you previously thought.  This is one case where the common mistake is actually an underestimate.

[You can download a copy of the model here](https://github.com/ADGEfficiency/adgefficiency.github.io/blob/master/assets/avg_marginal/average-vs-marginal-emissions-2017-02-02-1.xlsx).

## Other methods for accounting for carbon 

The excellent [electricityMap](https://www.electricitymap.org/) has a great figure showing different methods for accounting for carbon.  This post focused on the carbon value of projects, so covers consequential accounting.

![Figure 3]({{ "/assets/avg_marginal/fig3.jpg"}})
*Figure 3 - Comparison of carbon accounting methods from [Tomorrow.com](http://www.tmrow.com/)*

In the 2018 paper *Creative accounting: A critical perspective on the market-based method for reporting purchased electricity (scope 2) emissions*, Brander et. al note that  the locational grid average should be the only method used to calculate average carbon intensities, except in the cases where actions result in additional renewable generation. 

electricityMap also have a [technical blog post on calculating marginal carbon intensities](https://medium.com/electricitymap/using-machine-learning-to-estimate-the-hourly-marginal-carbon-intensity-of-electricity-49eade43b421) where they detail their algorithm for calculating marginal carbon intensity. 

Thanks for reading!
