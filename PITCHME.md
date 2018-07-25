
## goals

only pictures, code and phrases

## abstract
This talk reviews two years of work on energy_py - [a reinforcement learning for energy systems(https://github.com/ADGEfficiency/energy_py)].  We will look at lessons learned designing the library, experience using the library with Open AI gym and energy_py environments.  Also covered is the use of synthetic data generation in energy_py environments. 

Inspiration for this talk is world models (show at start)

The difference in ability to compete across supervised, unsupervised + rl

comapre leading tech giants, startups AND traditional vertical energy utilities (who are below on all now, but will be above eventually)

```
run experiment during talk

use what readme has
feed stuff back into readme
link to this talk in the readme

```

## agenda

1 - lessons

2 - functionality

3 - next

## energy_py - lesson learnt building a reinforcement learning library

---?image=/assets/energy_py_talk/humber.jpg&size=auto 100%

---?image=/assets/energy_py_talk/humber_excel.png&size=auto 100%

---?image=/assets/energy_py_talk/repo.png&size=auto 80%

---
```python
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

## simplicity
- API (like gym)

## focus on one agent and single version of envs
- two bad implementations don't equal one good one
- low dimensional action spaces (either discrete or managabely discretizable)

discrete representations of env should be ok in energy
dont have combinations of actions (which is what causes the exponential blowup in discrete actions)

---

## focus on one deep learning library
- original idea was to allow support for any library
- better to use one library fully than half of two

---
## iteration over design

allows scaling (v traditional economy of scale)

---
## testing

---
## basic functionality

- logging
- saving of agent & env setup
- tensorboard
- saving and analyzing environment historiees
- test suite
- training and testing experiments
- spaces
- wrapping gym envs
- loading memories
- test and train expts
- early stopping
- memory olympics (ask audience which is quicker?)

Global space from spaces (I think go into detail on spaces)
Space design is fundamental to the library - because it is code that interacts both with agents and environments

Show simple changes like

space.observation_space_shape -> space.observation_space.shape

use of a default dict for the info dict - can add and remove keys as wanted


---
## minor optimizations

memory strucutre (deques versus arrays)

use of shape dict, use of named tuple

---

## removing functionality

previously supported processors and schedulers

now using tensorflow for this (batch norm layer for Bellman target processing)

---

## style guide

a master and dev branch

single inheritance

Use tensorflow where possible (processors, schedulers etc)

Full docstrings are optiomal
Defined if needed, otherwise rely on the infomation about a variable type being visible in the code (ie from being used in a function, having a method called on it etc)

---

## tools

dl libraries - keras -> tf

operating systems - windows -> ubuntu -> osx

editors - notepad ++ -> spyder -> atom -> vim

---

## context of model

MCTS beating DQN

If you need simulation (because of sample inefficiency) -> you need to have a model

If you get a model for free - what next?

sample inefficiency -> need simulation

simulation is a model!

---

## poor mans gans

sample vs distributional model

key idea - we learn behaviour that will generalize

fighting the generalization problem earlier

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

## preseeding with experience of high quality policies

DQN has drawbacks and advantages - try to take advantage of all the advantages

One of these is being off-policy - the ability of DQN to learn from the experience of other policies

---
## combining with supervised learning
ie use time series forecasting to reduce dimensionality of the observation
because rl tuning process is longer + runs over mutltiple random seeds -> want to keep the supervised separate

---

## lessons

python setup.py develop

## ml and energy

advantage in energy - stuff that generates data once often generates it a lot
because data of interest is streaming time series, once you connect to data it keeps on giving
contrast this with a business where you get customer info on signup, and then never again (ie inactive user)

disadvantage in energy - the digitisation challenge
Every project I’ve been involved had digitisation as a key component
This slows down learning, limits datasets, makes datasets heavily non-iid

## backwards induction

goal with backwards induction is to allow an energy_py env to be instantly solvable using BI

BI = model based

Allows measuring the quality of forecasts (i.e. - when the model is wrong)

## next 

wrapping other environments - has to be the most efficient use of resources (not repeating work)


