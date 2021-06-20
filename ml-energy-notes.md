# Core message

## State

- distributed, small scale, intermittent (challenges)
- lots of data, but much still not digitized
- mostly sequence / temporal based
- more things to measure (distributed)
- more uncertainty (variable renewable)
- more challenging control problems - storage, flexible demand
- more data available
- heavy seasonality & trends (non stationary)

## Potential

- cost functions
- data, things to measure
- possibility to simulate

Optimization or prediction problems

Case for ML
- more data in future
- compute cheap

ml-energy - raw data = real world data (sensors) - we have only just started to do

## Challenges
Uncertantity
- interpretability

Digitization

Unlearnable patterns

- not enough data, or no labels
- seasonality & trend
- unlearnable patterns - tasks we can't do
- messy data, data that can't be worked on properly (can't be joined - example of this!)
- non-stationary processes, black swans (predicting the US shale gas revolution)

energy = small data
- renewables plant only 2 years old
- combinigc datasets useful
- Poor state of digitization means working with small data is the primary work of energy data scientists

The primary challenge is access to data.  The energy industry is still in the process of digitization - all my work in the energy has involved setting up the basic data collection infrastructure.  We've seen how important large amounts of data is to machine learning - a lack of historical data can be a show stopper for many energy and machine learning projects.

Forward thinking energy companies know that data can only be collected once.  It's more than just a local historian recording data from the site control system.  The 21st century energy company has everything from sensor level data to accounting data available to the employees and machines that need it, worldwide and in near real time.

The curation of large and interesting datasets is one of the few defensible advantages an energy company can build (another is brand).  These datasets are valuable not only because of how we can use them today, but because of the insights that can be generated tomorrow.

Limited to learn from the past

# SECTION THREE - APPLICATIONS

case studies

- cutomer segmentation
- chatbots
- customer service demand prediction
- churn prediction
- reccomendation emails to customers

- predictive maintenance
- forecasting
- control

When thinking about applying machine learning to an energy problem, the first and most important consideration is the dataset.  In fact, the first step in many machine learning projects is the same - start collecting data.  A dataset for supervised machine learning has two parts - the features (such as images or raw text) and the target (what you want to predict).  A dataset for reinforcement learning is a simulator - something that the learning algorithm can interact with.

Some potential applications of machine learning in energy include (but are not limited too):
- predictive maintenance 
- customer segmentation 
- churn prediction and minimization

Now we will dive into the details of a few applications of machine learning in energy that are happening today.

## Time series forecasting

The economics and environmental impact of energy depends on time of use.  Forecasting has always been an important practice in energy - increased deployment of variable wind and solar makes forecasting more valuable.  [DeepMind have claimed a 20 % improvement in the value of energy using a 36 hour ahead forecast](https://deepmind.com/blog/machine-learning-can-boost-value-wind-energy/).  Better forecasts can increase the value of renewables and reduce the requirement for backup fossil fuels.

![]({{ '/assets/ml_energy/wind.png' }})

Particularly exciting is the ability to forecast wind or solar using satellite images and deep convolutional neural nets - see the work of [Jack Kelly and Dan Travers at Open Climate Fix](https://openclimatefix.github.io/).

## Control

TRADING

Optimal control of complex energy systems is hard.  Reinforcement learning is a framework for decision making that can be applied to a number of energy control problems, availability of reward signals, simulators

Better control of our energy systems will allow us to reduce cost, reduce environmental impact and improve safety.

![]({{"/assets/ml_energy/rl_energy.png"}})

[DeepMind's data centre optimization is the most famous example of energy and machine learning](https://deepmind.com/blog/safety-first-ai-autonomous-data-centre-cooling-and-industrial-control/).  The algorithm makes decisions on a five minute basis using thousands of sensors.  These sensors feed data into deep neural networks, which predict how different combinations of actions will affect the future efficiency of cooling generation.

This sounds like a form of DQN - a reinforcement learning algorithm that predicts future reward for each possible action.

The neural networks perform computations on the cloud, with the suggested action sent back to the data centre before safety verification by the local control system.

![fig2]({{ '/assets/ml_energy/data_centre.png' }})

*Performance of the data centre measured using energy input per ton of cooling (kW/tonC), and improves with more data, from an initial 12% to 30% over nine months.*

---

AI vs Climate Change: Insights from the Cutting Edge (Google I/O'19) - Mustafa Suleyman - https://youtu.be/AyHpt8uxwSo
- diagram at min 11:21 (ed_hawkins)

2 layer convnet to predict app use
- saving in battery life
- trains locally to specalize for the user

3 years to optimize data centers
- 2,500 inputs (min 17:37)
- 20 actions

Wind energy
- min 26:00 - picture of the forecast
