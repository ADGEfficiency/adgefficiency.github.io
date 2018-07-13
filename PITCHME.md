This talk reviews two years of work on energy_py - [a reinforcement learning for energy systems(https://github.com/ADGEfficiency/energy_py)].  We will look at lessons learned designing the library, experience using the library with Open AI gym and energy_py environments.  Also covered is the use of synthetic data generation in energy_py environments. 

```
run experiment during talk

use what readme has
feed stuff back into readme
link to this talk in the readme

```

## energy_py - lesson learnt building a reinforcement learning library

---
![humber]("/assets/energy_py_talk/humber.jpg")

---?image=/assets/energy_py_talk/humber.jpg&size=auto 100%

---?image=/assets/energy_py_talk/humber_excel.png&size=auto 100%

---

## focus on one agent
- two bad implementations don't equal one good one
- low dimensional action spaces (either discrete or managabely discretizable)

discrete representations of env should be ok in energy
dont have combinations of actions (which is what causes the exponential blowup in discrete actions)

---

## focus on one deep learning library
- original idea was to allow support for any library
- better to use one library fully than half of two

---
## minor optimizations

memory strucutre (deques versus arrays)

use of shape dict, use of named tuple

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

## next 

wrapping other environments - has to be the most efficient use of resources (not repeating work)
