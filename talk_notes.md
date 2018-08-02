not sure if I should have the  three pieces before i introduce each three or not???

def have summaries of both the climate 3 and the ml 3
 at the end

## goals

only pictures, code and phrases

## abstract

This talk reviews two years of work on energy_py - [a reinforcement learning for energy systems(https://github.com/ADGEfficiency/energy_py)].  We will look at lessons learned designing the library, experience using the library with Open AI gym and energy_py environments.  Also covered is the use of synthetic data generation in energy_py environments. 


```

use what readme has
feed stuff back into readme
link to this talk in the readme

```

## agenda

1 - lessons

2 - functionality

3 - next

--- 

humber slide

---

humber excel

---

climate

---

three pieces of info on climate
- size of the problem, moral problem, personal problem

- https://scripps.ucsd.edu/programs/keelingcurve/
- climate - was 40 oC in Adelaide in April when I was there.  Europes latest heatwave
- value at the intersection (energy + ml)

1. Climate change is our biggest long term problem.  Most climate models are optimistic - reality will be more painful than we expect.  Perverse feedback of running pumps in Miami to remove floodwater.

Fundamentally a discount rate problem.

2. More question of climate - disproportionately on the countries who have done the least.  Climate changes in northern Europe can actually be viewed as positive (if you like longer, hotter summers.  Impact on biodiversity (if you care about animals).

3. Climate is a moral problem - solved by your actions day to day.  Travel and diet.  But be compassionate towards yourself and others.

Also important to acknowledge that nothing lasts forever - our grasping onto our current climate is an attachment to something that is impermanent.

---

Motivation to learn how to use a computer = help solve the climate problem

Role of reinforcement learning in energy = control

advantage in energy - stuff that generates data once often generates it a lot
because data of interest is streaming time series, once you connect to data it keeps on giving
contrast this with a business where you get customer info on signup, and then never again (ie inactive user)

disadvantage in energy - the digitisation challenge
Every project I’ve been involved had digitisation as a key component
This slows down learning, limits datasets, makes datasets heavily non-iid

---

Role of demand side response 

Value in demand side response = avoiding expensive standby plant.  

Analogy of taxi drivers (picture)

- demand side traditional vs price responsive
- big negative is the minimum sizes, also misallocation of total secured flexibility

Both batteries and demand side flex are storage

---

Role of energy_py = supporting experimentation

---

expt snippets and low level snippets

---

contributions

Naive agents as baselines - helpful to confirm the performance of environments (ie we have a baseline agent that we know with perfect forecasts will never lose money, and then in simulation it does that, it suggests the environment is capturing the dynamics)

Naive agents also offer benefit of comparison with learning agnets (two benefits)

--- 

Space design is fundamental to the library 

- code that interacts both with agents and environments

--- 

performance

battery results

recent flex results

---

lessons

1. simplicity
2. the env model problem
3. use of synthetic data

---

simplicity

- API (like gym)

single agent = can develop library to take advantage of it

- ie preseeding of experience because DQN is off policy

## focus on one agent and single version of envs
- two bad implementations don't equal one good one
- low dimensional action spaces (either discrete or managabely discretizable)

discrete representations of env should be ok in energy
dont have combinations of actions (which is what causes the exponential blowup in discrete actions)

## focus on one deep learning library

- original idea was to allow support for any library
- better to use one library fully than half of two

## removing functionality

previously supported processors and schedulers

now using tensorflow for this (batch norm layer for Bellman target processing)

## style guide

a master and dev branch

single inheritance

Use standard library where possible 

Use tensorflow where possible (processors, schedulers etc)

Full docstrings are optional
Defined if needed, otherwise rely on the infomation about a variable type being visible in the code (ie from being used in a function, having a method called on it etc)

three pieces of info on energy and reinforcement learning

1. the environment model problem
2. using synthetic data for generalization
3. combination with supervised learning

---

modern rl so sample inefficient that you need simualtion
but if you have simulation, then there are other better models such as MCTS

the work in energy is therefore in building useful simulation models - this unlocks both

bi

goal with backwards induction is to allow an energy_py env to be instantly solvable using BI

BI = model based

Allows measuring the quality of forecasts (i.e. - when the model is wrong)

Just show code for object oriented BI

---
synthetic data - aka poor mans gans


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
combining with supervised learning

ie use time series forecasting to reduce dimensionality of the observation
because rl tuning process is longer + runs over mutltiple random seeds -> want to keep the supervised separate
