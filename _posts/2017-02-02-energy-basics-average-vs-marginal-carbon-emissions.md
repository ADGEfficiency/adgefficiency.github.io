---
id: 562
title: 'Average vs Marginal Carbon Emissions'
date: 2017-02-02T01:27:04+00:00
author: Adam Green
layout: post
guid: http://adgefficiency.com/?p=562
permalink: /energy-basics-average-vs-marginal-carbon-emissions/
categories:
  - Energy Basics
---

Carbon savings might seem like a simple calculation – yet many professionals are getting it wrong. I know because I was making this mistake in my previous job!

Accurate calculation of carbon savings is crucial in the fight against climate change.  Knowing how much we save from a project can be compared to other projects such as renewable generation.

So where are people going wrong?  The key is to understand the difference between average and marginal carbon emissions.
Average carbon emissions are calculated using the total carbon emissions and total amount of electricity generated.

|Table 1 – Calculation of average carbon intensity for Base Case|
|---|
|Carbon emissions|	tC|	83,330|
|Electricity generated|	MWh|	182,827|
|Carbon intensity|	tC/MWh|	0.456|

This average intensity can be used to calculate carbon savings.  For example if we had a project that saved 2 MWh we would calculate 2 * 0.456 = 0.912 tC as the saving.  This is wrong!

To understand why we need to the concept of the marginal generator.  In reality as electricity is saved the reduction in generation is not spread across each generator.  The reduction occurs in one plant – the marginal generator.  Let’s run through an example.

Suppose we have a grid where electricity is supplied by either wind or coal (the Base Case).  If we save 1 GW of electricity, the generation of the coal plant will reduce by 1 GW (Case 1).

The wholesale mechanism operating in most electricity markets will reduce output on the most expensive plant, not reduce the output of all plants equally.

![Figure 1]({{ "/assets/avg_marginal/fig1_2.png"}})
**Figure 1 & 2 – The effect of saving 1 GW of electricity.  Note that the generation from wind is unchanged.**

|Table 2 – Results for the Base Case & Case 1|
|---|
|||Base Case|	Case 1|	Saving|
|Wind|	MWh|	91,256|	91,256|	0|
|Coal|	MWh|	91,571|	67,571|	24,000|
|Total|	MWh|	182,827|	158,827|	24,000|
|Carbon emissions|	tC|	83,329|	61,489|	21,840|
|Carbon intensity|	tC/MWh|	0.456|	0.387|	0.910|

Our carbon saving is equal to 1 GW multiplied by the carbon intensity of the marginal plant.
If we were to use the average grid carbon intensity (0.456 tC/MWh) we calculate a daily carbon saving of only 21,480 tC.

You might be asking – how do we know what the marginal generator will be?  It’s likely to be the most expensive generator at that time (it may not be if the plant needs to be kept on for technical reasons).   As renewables are characterized by low marginal costs they are the unlikely to be pushed off the grid.

Luckily high marginal cost generators like open cycle gas turbines are usually also carbon intense – so your saved electricity is likely doing valuable work – and potentially more than you previously thought!

You can download a copy of the model here.
