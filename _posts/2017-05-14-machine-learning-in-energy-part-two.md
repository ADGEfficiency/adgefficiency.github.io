---
title: 'machine learning in energy - part two'
date: 2017-05-14
categories:
  - Energy
  - Machine Learning
excerpt: Applications of machine learning in energy - forecasting, disaggregation and reinforcement learning.

---

This is the second of a two part series on the intersection of machine learning and the energy industry.  [The first part introduces what machine learning is, why it’s a fundamental paradigm shift, what’s driving performance and what some of the challenges are](https://adgefficiency.com/machine-learning-in-energy-part-one/).

This second part will detail three applications of machine learning in energy - time series forecasting, disaggregation and reinforcement learning.

## forecasting of electricity generation, consumption and price

### what's the problem

The first major lesson I learnt as an energy engineer was the importance of time in energy systems.  The economic and environmental impact of energy generation or consumption changes massively depending on the time when they occur.

Large scale storage of electricity is only just becoming economic - without storage, generation must match consumption.  When demand is high the grid can be forced to use expensive and inefficient peaking plants.  In periods of low demand electricity can be so abundant that the price becomes negative.

The more accurately system operators and market participants understand what generation and demand will be in the future, the better they can manage the grid.  The major uncertainty with dispatchable fossil fuel generators is unscheduled maintenance - usually expected to cause around 5-10% downtime annually.

Our current energy transition is making the forecasting problem more difficult.  **Our transition towards intermittent and distributed generation is introducing more uncertainty on both the generation and demand side**.

![]({{"/assets/ml_energy/energy_transition.png"}})

Intermittent, renewable generation is powered by the weather - making it hard to forecast.  Wind generation depends on forecasting wind speeds over vast areas. Solar power is more predictable but can still see variation as cloud cover changes.  As grid scale wind & solar penetration increase balancing the grid becomes difficult. Higher levels of renewables can lead to more fossil fuel backup kept in reserve in case forecasts are wrong.

The distributed and small scale of renewables (solar in particular) also makes demand forecasting more difficult.  A solar panel sitting on a residential home is not directly metered - the system operator has no idea it is there. As this solar panel generates throughout the day it appears to the grid as reduced consumption. 

The most famous example of this the 'duck curve'.  In California the loss of distributed solar power as the sunsets appears to the system operator as a massive increase in demand, requiring dispatchable generation to quickly ramp up to meet the loss of solar.

![]({{"/assets/ml_energy/duck_curve.png"}})

**Our current energy transition is a double whammy for grid balancing.** Forecasting of both generation and consumption is becoming more challenging.

This has a big impact on electricity prices. In a wholesale electricity market price is set by the intersection of generation and consumption. Volatility and uncertainty on both sides spill over into more volatile electricity prices.  South Australia is a prime example of this - the combination of high penetrations of wind and solar with little out of market subsidies makes the South Australian wholesale electricity market one of the most volatile commodities markets in the world.

### how machine learning will help

Classical time series forecasting models such as SARIMA are well developed for forecasting energy time series.  Machine learning adds additional tools to the time series forecasting toolbox.

Both regression and classification supervised machine learning models can be used for time series forecasting.  Regression models can directly forecast electricity generation, consumption and price. Classification models can forecast the probability of a spike in electricity prices.

Well trained random forests, support vector machines and neural networks can all be used to solve these problems. Of particular interest are recurrent neural networks - networks that model the temporal structure of features and targets implicitly by feeding in and generating sequences.

A key challenge is data. As renewables are weather driven forecasts of weather can be useful exogenous variables. It's key that we only train models on data that will be available at the time of the forecast. This means that historical information about weather forecasts can be more useful than the actual weather data.

Other useful variables for forecasting electricity generation, demand or prices include:
- lagged values of the target (i.e. what the price was last half hour
- prices in neighbouring markets
- prices of primary fuels such as natural gas or coal
- interconnector flows or flows through constraints in the grid
- externally supplied forecasts of the target variable (and the errors of those forecasts)

### what's the value to the world

Improving forecasts allows us to better balance the grid, reduce reliance on fossil fuel peaking or backup plants and reduce curtailment of renewables.

It's not only the economic & environmental cost of keeping backup plant spinning. Incorrect forecasts can lead to fossil fuel generators paid to reduce output - known as redispatching. This increases the cost to supply electricity to customers.

There are benefits for consumers of electricity as well. Improved prediction can also allow flexible electricity consumption to respond to market signals.  More accurate forecasts that can look further ahead will allow more electricity consumers to be flexible. Using flexible assets to manage the grid will reduce our reliance on fossil fuels for grid balancing.

### sources and further reading

- [Machine Learning in Energy (Fayadhoi Ibrahima)](http://large.stanford.edu/courses/2015/ph240/ibrahima2/)
- [7 reasons why utilities should be using machine learning](https://blogs.oracle.com/utilities/utilities-machine-learning)
- [Germany enlists machine learning to boost renewables revolution](http://www.nature.com/news/germany-enlists-machine-learning-to-boost-renewables-revolution-1.20251)
- [Weron (2014) Electricity price forecasting: A review of the state-of-the-art with a look into the future](http://www.sciencedirect.com/science/article/pii/S0169207014001083)
- [Re-dispatch costs in the German power grid](https://www.cleanenergywire.org/factsheets/re-dispatch-costs-german-power-grid)
- [This “duck curve” is solar energy’s greatest challenge](https://www.vox.com/2018/5/9/17336330/duck-curve-solar-energy-supply-demand-problem-caiso-nrel)

## energy disaggregation

### what's the problem

Imagine if every time you went to the restaurant you only got the total bill, with no breakdown of what you spent on your main and dessert.  Having a line by line breakdown of your spending is valuable.  This is the idea behind energy disaggregation - giving customers a breakdown of where their electricity went.

In an ideal world we would have visibility of each individual consumer of energy. We would know when a TV is turned on in a home or a pump is running in an industrial process. One solution would be to install metering on every consumer - an expensive, complex and impractical process.

Energy disaggregation is a more elegant solution. **A good energy disaggregation model can estimate appliance level consumption through a single aggregate meter.**

### how machine learning will help

Supervised machine learning is all about learning patterns in data. Many supervised machine learning algorithms can learn the patterns in the total consumption. Kelly & Knottenbelt (2015) used recurrent and convolutional neural networks to disaggregate residential energy consumptions.

A key challenge is data. Supervised learning requires labeled training data. Measurement and identification of sub-consumers forms training data for a supervised learner. Data is also required at a very high temporal frequency - ideally less than one second.

### what's the value to the world

Energy disaggregation has two benefits - it can identify & verify savings opportunities, and can increase customer engagement.

Imagine if you got an electricity bill that told you how much it cost you to run your dishwasher that month. The utility could help customers understand what they could have saved if they ran their dishwasher at different times.  **This kind of feedback can be very effective in increasing customer engagement** - a key challenge for utilities around the world.

### sources and further reading

- [7 reasons why utilities should be using machine learning](https://blogs.oracle.com/utilities/utilities-machine-learning) 
- [Neural NILM: Deep Neural Networks Applied to Energy Disaggregation](https://arxiv.org/pdf/1507.06594.pdf) 
- [Energy Disaggregation: The Holy Grail (Carrie Armel)](https://web.stanford.edu/group/peec/cgi-bin/docs/events/2011/becc/presentations/3%20Disaggregation%20The%20Holy%20Grail%20-%20Carrie%20Armel.pdf)
- [Putting Energy Disaggregation Tech to the Test](https://www.greentechmedia.com/articles/read/putting-energy-disaggregation-tech-to-the-test)

## reinforcement learning

### what's the problem

Optimal control of energy systems is hard. Key variables such as price and energy consumption exhibit seasonality and are non-stationary. Operators control systems with a large number of actions, with the optimal action changing throughout the day.

Our current energy transition makes this problem harder. Increased uncertainty on the generation and demand side leads to more volatility in key variables such as electricity prices.  The need for smarter ways of managing energy systems, such as demand flexibility, introduce more actions that need to be considered.

Today deterministic sets of rules or abstract models are commonly used to dispatch plant. Deterministic rules for operating any non-stationary system can't guarantee optimality. Changes in key variables can turn a profitable operation to one that loses money.

Abstract models (such as linear programming) can account for changes in key variables. But abstract models often force the use of unrealistic models of energy systems. More importantly the performance of the model is limited by the skill and experience of the modeller.

### how machine learning will help

Reinforcement learning gives a machine the ability to learn to take actions. The machine takes actions in an environment to optimize a reward signal. In the context of an energy system that reward signal could be energy cost, carbon or safety - whatever behaviour we want to incentivize.

![]({{"/assets/ml_energy/rl_energy.png"}})

The potential and promise of reinforcement learning is for an agent to learn to control a system better than any human can.  This superhuman level of performance has been demonstrated in environments such as Atari games, Chess and Go.  

A key problem with modern reinforcement learning is sample efficiency - agents need vast amounts of experience to learn high quality actions.  This limits reinforcement learning to environments that can be simulated - note that all of the landmark achievements have been with virtual environments.  The environments we want to control are real - meaning that either we need a human designed simulation model or to learn simulation models, in order to sample enough experience to learn from.

### what's the value to the world

Better control of our energy systems will allow us to reduce cost, reduce environmental impact and improve safety. Reinforcement learning allows us to do this at superhuman levels of performance.

Letting machines make decisions in energy systems allows operators to spend more time doing routine or preventative maintenance.  It also allows more time to be spent on upgrading existing plant.

### sources and further reading

- [energy_py – reinforcement learning in energy systems](http://adgefficiency.com/energy_py-reinforcement-learning-for-energy-systems/)
- [Minh et. al (2016) Human-level control through deep reinforcement learning](http://www.nature.com/articles/nature14236) - [Reinforcement learning course by David Silver (Google DeepMind)](https://www.youtube.com/watch?v=2pWv7GOvuf0)
- [Deep reinforcement learning doesn't work yet](http://www.alexirpan.com/2018/02/14/rl-hard.html)

---

Thanks for reading!
