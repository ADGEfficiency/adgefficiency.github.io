### energy_py

#### lessons learnt building an energy reinforcement learning library

#### Adam Green - adam.green@adgefficiency.com

---?image=/assets/energy_py_talk/humber.jpg&size=auto 100%

---?image=/assets/energy_py_talk/humber_excel.png&size=auto 100%

three pieces of info on climate

---?image=/assets/energy_py_talk/climate.png&size=auto 80%


ml + energy solution to the climate problem

---

price response flexible demand and the lazy taxi driver

---

energy_py = supporting experimentation 

---?image=/assets/energy_py_talk/repo.png&size=auto 100%

---?code=assets/high_level.py&lang=bash

---?code=assets/low_level.py&lang=bash

---

logging
saving of agent & env setup
tensorboard
saving and analyzing environment historiees
test suite
spaces
wrapping gym envs

---

In progress


---

details

---

```python

action_space = GlobalSpace('action').from_spaces(
    [ContinuousSpace(0, 100), DiscreteSpace(3)],
    ['acceleration', 'gear']
)

action = action_space.sample()

assert action_space.contains(action)

discrete_spaces = action_space.discretize(20)

action = action_space.sample_discrete()

```
---

```python

#  load a state or observation space from a dataset
state_space = GlobalSpace('state').from_dataset('example')

# we can sample an episode from the state
episode = state_space.sample_episode(start=0, end=100)

# sample from the current episode by calling the space
state = state_space(steps=0)
```

---

performance

---

lessons

---

simplicity

---

three pieces of info on energy and reinforcement learning

---

the env model problem

modern RL is sample inefficient -> need simulation

but

if you have a simulator, there are better methods than modern RL

MCTS beating DQN (show the ref!)

---

backwards induction

---

synthetic data - aka poor mans GANS

---

combining with supervised learning

---
next for energy_py

loading memories
test and train expts
early stopping
backwards induction

---


next for ???

model based methods 

building simulations

wrapping other environments
