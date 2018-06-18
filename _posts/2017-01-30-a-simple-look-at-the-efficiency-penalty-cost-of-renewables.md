---
title: A Simple Look at the Efficiency Penalty Cost of Renewables
date: 2017-01-30
author: Adam Green
categories:
  - Energy
excerpt: Can a fossil fuel efficiency penalty offset the benefits of renewables?

---

This article is a response to [Wind Integration vs. Air Emission Reductions: A Primer for Policymakers](https://www.masterresource.org/integrationfirming/wind-integration-and-emissions/), which claims the efficiency penalty of turning down fossil fuel power stations offsets the benefit of renewable power generation.  This post details some simple modelling to determine if this is true.

Renewable power generation brings a carbon benefit by avoiding fossil fuel generation.  Most generation (machines in general) is designed for optimum efficiency at maximum (or near maximum) load.  Operation of the machine above or below the optimum load will incurr an efficiency penatly.

Below I look at what the breakeven efficiency penalty would be to offset the benefit of renewable generation  
- total electricity demand of 1 GW
- renewable output is varied from 0 GW to 0.5 GW 
- coal supplies the balance
- coal operates at a 50 % HHV efficiency at full load

I then looked at various efficiency penalty factors in the form of reduced efficiency per reduction in load (`% HHV / % load`).   The efficiency penalty was modeled as linear.  [You can download a copy of the model here](https://github.com/ADGEfficiency/adgefficiency.github.io/blob/master/assets/renewable_effy_penalty/coal-efficiency-penalty-2017-01-30.xlsx).

![Figure 1]({{ "/assets/renewable_effy_penalty/fig1.png"}})
**Figure 1 – The effect of various assumed efficiency penalties on fossil fuel consumption**

For this simple model `5 % HHV / % load` is the break even.  If the efficiency really reduces at this rate then generating electricity from renewables is giving us no carbon benefit.

The real question is what is the actual relationship between fossil fuel power station output and efficiency.   It’s likely to be non-linear.  I also expect it would not be as harsh as 5 % HHV/% load – so likely renewables are providing a carbon benefit.

Is it is useful to know that this is a carbon penalty we could be paying somewhere on the system as renewables penetration increases.  This penalty will net off some of the maximum benefit that renewable generation could supply.

This comparision is between renewables and coal.  Because natural gas is cleaner than coal, a comparison with gas would be more in favour of renewables.  This is because a reduction in efficiency doesn't hurt as much.

Permanently shutting down fossil fuel power stations means this effect doesn’t occur - so lets turn them off completely.

Thanks for reading!
