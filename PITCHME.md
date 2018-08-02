### energy_py

#### lessons learnt building a energy reinforcement learning library

#### Adam Green - adam.green@adgefficiency.com

---

---?image=/assets/energy_py_talk/humber.jpg&size=auto 100%

---?image=/assets/energy_py_talk/humber_excel.png&size=auto 100%

---?image=/assets/energy_py_talk/climate.png&size=auto 80%

---

three pieces of info on climate

1. it's bad
2. it's going to be worse for poor countires
3. your personal choices matter

---

ml + energy solution to the climate problem

---

price response flexible demand and the lazy taxi driver

---

energy_py = supporting experimentation

---

---?image=/assets/energy_py_talk/repo.png&size=auto 50%

---
running an experiment

```
#  high level api running experiments from config files

$ cd energy_py/experiments

$ python experiment.py example dqn

$ tensorboard --logdir='./results/example/tensorboard'
```

---

```python
#  low level gym-style api

import energy_py

env = energy_py.make_env(env_id='battery')

agent = energy_py.make_agent(
    agent_id='dqn',
    env=env,
    total_steps=1000000
    )

observation = env.reset()

while not done:
    action = agent.act(observation)
    next_observation, reward, done, info = env.step(action)
    training_info = agent.learn()
    observation = next_observation
```

---

## contributions

DQN

naive agents

energy envs

tools for experiment

---

## style guide

a master and dev branch

single inheritance

Use standard library where possible 

Use tensorflow where possible (processors, schedulers etc)

Full docstrings are optional
Defined if needed, otherwise rely on the infomation about a variable type being visible in the code (ie from being used in a function, having a method called on it etc)

---

## low level functionality

- logging
- saving of agent & env setup
- tensorboard
- saving and analyzing environment historiees
- test suite
- spaces
- wrapping gym envs

In progress

- loading memories
- test and train expts
- early stopping

---

## spaces 

Space design is fundamental to the library 
- code that interacts both with agents and environments

---

## shape dict

use of shape dict, use of named tuple
space.observation_space_shape -> space.observation_space.shape

---

## default dicts

used to create the info dict

easily turned into a dataframe

---

## tools

`python setup.py develop`

dl libraries - keras -> tf

operating systems - windows -> ubuntu -> osx

editors - notepad ++ -> spyder -> atom -> vim

---

## performance

insert latest learning curves

---

lessons

three pieces of info on energy and reinforcement learning
- , importance of a model, using synthetic data for generalization

---

simplicity

---

the env model problem

## context of model

MCTS beating DQN

If you need simulation (because of sample inefficiency) -> you need to have a model

If you get a model for free - what next?

sample inefficiency -> need simulation

simulation is a model!

## backwards induction

goal with backwards induction is to allow an energy_py env to be instantly solvable using BI

BI = model based

Allows measuring the quality of forecasts (i.e. - when the model is wrong)

Just show code for object oriented BI

---

## synthetic data - aka poor mans GANS

Inspiration for this talk is world models (show at start)

The difference in ability to compete across supervised, unsupervised + rl

comapre leading tech giants, startups AND traditional vertical energy utilities (who are below on all now, but will be above eventually)

sample vs distributional model

key idea - we learn behaviour that will generalize

fighting the generalization problem earlier

this is a subtle point - that you want to hve ways to estimate your error accurately
You don't care about the actual accuracy - getting a high test set error is useful feedback to learn from

rl is careless about the test/train problem

in energy we can test it specifically by keeping time series data separate

want to fight the generalization problem head on - take advantage of it

generating exact customer profiles is hard.  generating believeable ones is easier
behaviours that are learnt for one demand profile can be used with another profile
can learn on synthetic, and then test on real
more test data generated continuously (ie if you wait you get a new holdout set)

synthetic data allows a test set to be kept seprate, and allows a an estimation of generalization error

the rl test problem
if you use the input from a supervised model as observation for rl
you should only use the test set press
this limits the data you can train an rl agent with

if having different distributions is valuable (ie makes generalisation better) - this is a great thing! the anti iid

---
## combining with supervised learning
ie use time series forecasting to reduce dimensionality of the observation
because rl tuning process is longer + runs over mutltiple random seeds -> want to keep the supervised separate

---

## next 

model based methods - monte carlo tree search

wrapping other environments - has to be the most efficient use of resources (not repeating work)

modern rl so sample inefficient that you need simualtion
but if you have simulation, then there are other better models such as MCTS

the work in energy is therefore in building useful simulation models - this unlocks both
