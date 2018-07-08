```
run experiment during talk

show flexing of origin customer (real life!)

memory strucutre (deques versus arrays)

use of shape dict, use of named tuple

use what readme has

focus on one agent
discrete representations of env should be ok in energy
dont have combinations of actions (which is what causes the exponential blowup in discrete actions)

the rl test problem
if you use the input from a supervised model as observation for rl
you should only use the test set press
this limits the data you can train an rl agent with

synthetic learning data
generating exact customer profiles is hard.  generating believeable ones is easier
behaviours that are learnt for one demand profile can be used with another profile
can learn on synthetic, and then test on real
more test data generated continuously (ie if you wait you get a new holdout set)

value of being able to generate data

if having different distributions is valuable (ie makes generalisation better) - this is a great thing! the anti iid

combining with supervised learning
ie use time series forecasting to reduce dimensionality of the observation
because rl tuning process is longer + runs over mutltiple random seeds -> want to keep the supervised separate

advantage in energy - stuff that generates data once often generates it a lot
because data of interest is streaming time series, once you connect to data it keeps on giving
contrast this with a business where you get customer info on signup, and then never again (ie inactive user)

disadvantage in energy - the digitisation challenge
Every project I’ve been involved had digitisation as a key component
This slows down learning, limits datasets, makes datasets heavily non-iid

```


## energy_py - lesson learnt building a reinforcement learning library

---
![]({{ "/assets/energy_py_talk/humber.png"}}) 

---
![]({{ "/assets/energy_py_talk/humber_excel.png"}}) 

---

## focus on one agent
- two bad implementations don't equal one good one
- low dimensional action spaces (either discrete or managabely discretizable)

---

## focus on one deep learning library
- original idea was to allow support for any library
- better to use one library fully than half of two

---

## context of model

MCTS beating DQN

If you need simulation (because of sample inefficiency) -> you need to have a model

If you get a model for free - what next?


---

## 

poor mans gans

sample vs distributional model

key idea - we learn behaviour that will generalize

fighting the generalization problem earlier

