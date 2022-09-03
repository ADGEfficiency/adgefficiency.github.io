# What I want everyone to know

## Non-linear optimization on top of the linear optimization cake

Cost functions form the basis of many optimization algorithms - including the mixed integer linear programming that is used heavily throughout the energy industry.

We are again lucky to have many systems that can be modelled (or approximated well) as mixed integer linear programs.  Linear programming can be used to optimize the dispatch of assets like gas generation, hydro or battery storage.

Take your existing optimization work flows and feed them better inputs/forecasts

Linear programs offer access to the global optimum - convergence to the best solution, everytime. Non-linear optimization is iterative, messy and only locally optimal.

An excellent example of the power of linear programming is the work of Kantor et al (2020) in [A Mixed-Integer Linear Programming Formulation for Optimizing Multi-Scale Material and Energy Integration](https://www.frontiersin.org/articles/10.3389/fenrg.2020.00049/full), which optimzes both the sizing and dispatch of energy assets like ? and ?.

However linear methods can only be pushed so far - at some point the fundamental non-linearity of the world gets in the way.  

Take for example mass & energy balances - enforcing constraints on one is fine, but when you start to build models that balance both mass and energy, you will end up with a fundamentally bilinear system.

```
mass * temperature
```

A classic example of this is dispatch of a battery operating in a price arbitrage scenario.

While it's quite easy to use linear methods with perfect foresight of prices, if 

Optimization or prediction problems
- ml = non-linear = big opportunity,
- both things ML does well
- time series forecasting
- sequence based deep learning,

Prediction + linear optimization = a natural way to fit ml into existing energy optimization problems

Case for ML
- more data in future
- compute cheap


# Further

https://www.climatechange.ai/summaries

- [AI, Machine Learning and Deep Learning](https://adgefficiency.com/ai-ml-dl/)

Optimization of Energy Systems, Victor Zavala https://youtu.be/ngiQCRJWZws

Warren Powell, "Stochastic Optimization Challenges in Energy" - https://youtu.be/dOIoTFX8ejM
