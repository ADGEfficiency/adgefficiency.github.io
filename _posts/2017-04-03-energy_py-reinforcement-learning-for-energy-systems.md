---
title: 'energy_py - reinforcement learning for energy systems'
date: 2017-04-03
categories:
  - Energy
  - Reinforcement Learning
---
**energy_py is reinforcement learning for energy systems** - [GitHub](https://github.com/ADGEfficiency/energy_py)

Using reinforcement learning agents to control virtual energy environments is a necessary step in using reinforcement learning to optimize real world energy systems.

energy_py supports this goal by providing a **collection of agents, energy environments and tools to run experiments.**

GitHub is full of amazing repos with high quality implementations of agents and environment for classic control, video games and robotics.  energy_py contributes to this by suppling environments that simulate real world energy systems.

## What is reinforcement learning

![Figure 1]({{"/assets/energy_py/sl_usl_rl.png"}})

Reinforcement learning is the branch of machine learning where an agent **learns through action**.  Fundamentally it's about making good decisions.  

It’s quite different from supervised learning. In supervised learning we start with a big dataset. We train a model to replicate the patterns found in this dataset so that we can make predictions.

In reinforcement learning we start out with no data. The agent generates data (sequences of experience) by interacting with the environment. The agent uses it’s experience to learn how to interact with the environment. In reinforcement learning we not only learn patterns from data, we also generate our own data.

This makes reinforcement learning more democratic than supervised learning. Modern supervised learning requires massive amounts of labelled training data.  This requirement allows tech companies to build defensible moats around their unique datasets.

In reinforcement learning all that is needed is an environment (real or virtual) and an agent.  There is also a requirement for hardware (which has become increasingly democratised by variable cost access through the cloud) and the expertise to setup, debug and roll out the agents and environments.

If you are interested in learning more about reinforcement learning, I would recommend these resources
- [Li (2017) Deep Reinforcement Learning: An Overview](https://arxiv.org/pdf/1701.07274.pdf)
- [Sutton & Barto - Reinforcement Learning: An Introduction - 2nd Edition (in progress)](http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf)
- [UCL video lectures by David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

I also teach a two day introduction to reinforcement learning course at Data Science Retreat in Berlin.  You can find the [course materials on GitHub](https://github.com/ADGEfficiency/dsr_rl).  **I am always happy to come give this course for free at interesting companies and universities**.

## Why reinforcement learning in energy?

Optimal operation of energy assets is very challenging. Our current energy transition makes this difficult problem even harder.

The rise of intermittent generation is introducing uncertainty on the generation and demand side.  Delivering value from new technologies such as demand side flexibility and storage require smart decision making.

For a wide range of problems machine learning results are both state of the art and better than human experts. We can get this level of performance using reinforcement learners to control energy system.

Today many operators use rules or abstract models to dispatch assets. A set of rules is not able to guarantee optimal operation in many energy systems.

Optimal operating strategies can be developed from abstract models. Yet abstract models (such as linear programming) are often constrained. These models are limited to approximations of the actual plant.  Reinforcement learners are able to learn directly from their experience of the actual plant. These abstract models also require significant amount of bespoke effort by an engineer to setup and validate.

With reinforcement learning we can use the ability of the same agent to generalize to a number of different environments. This means we can use a single agent to both learn how to control a battery and to dispatch flexible demand. It’s a much more scalable solution than developing site by site heuristics or building an abtract model for each site.

There are challenges to be overcome. The first and most important is safety. Safety is the number one concern in any engineering discipline.

I believe that by reinforcement learning should be first applied on as high a level of the control system as possible. This allows the number of actions to be limited and existing lower level safety & control systems can remain in place. The agent is limited to only making the high level decisions operators make today.

There is also the possibility to design the reward function to incentivize safety. A well-designed reinforcement learner could actually reduce hazards to operators. Operators also benefit from freeing up more time for maintenance.

A final challenge worth addressing is the impact such a learner could have on employment. Machine learning is not a replacement for human operators. A reinforcement learner would not need a reduction in employees to be a good investment.

The value of using a reinforcement learner is to let operations teams do their jobs better. It will allow them to spend more time and improve performance for their remaining responsibilities such as maintaining the plant.  The value created here is a better-maintained plant and a happier workforce – in a plant that is operating with superhuman levels of economic and environmental performance.

Any machine requires downtime – a reinforcement learner is no different. There will still be time periods where the plant will operate in manual or semi-automatic modes with human guidance.

energy_py is one step on a long journey of getting reinforcement learners helping us in the energy industry. The fight against climate change is the greatest that humanity faces. Reinforcement learning will be a key ally in fighting it. You can [checkout the repository on GitHub here](https://github.com/ADGEfficiency/energy_py).

## Design choices

Inspiration for a lot of the design of energy_py comes from [OpenAI gym](https://github.com/openai/gym).

The ultimate goal for energy_py is to have a collection of high quality implementations of different reinforcement
learning agents.  Currently the focus is on implementing a single high quality implementation of DQN and it's
extensions.  A good summary of the DQN extensions is given in [Hessel et. al (2017) Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf).

Creating environments and agents follows a simple API as gym

```python
import energy_py

TOTAL_STEPS = 1000

env = energy_py.make_env(env_id='BatteryEnv',
                         dataset_name=example,
                         episode_length=288,
                         power_rating=2}

agent = energy_py.make_agent(agent_id='DQN',
                             env=env
                             total_steps=TOTAL_STEPS)
```

Observation and action spaces for environments are set using GlobalSpace objects.  To create an action space with one discrete action of three choices and a single continuous value
```python
from energy_py import ContinuousSpace, DiscreteSpace, GlobalSpace

action_space = GlobalSpace([DiscreteSpace(3), ContinuousSpace(1)])
```

Spaces have methods to deal with discretization

```python
actions_list = action_space.discretize(n_discr=20)

random_action = action_space.sample_discrete()
```

Processors can be used to preprocess numpy arrays.  In reinforcement learning different agents will want to normalize or
standardize observations, actions, or targets for neural networks.

The Normalizer transforms a batch to range [0,1], optionally using a historical min & max transforms a batch to range
[0,1], optionally using a historical min & max.  The Standardizer transforms a batch to mean 0, standard deviation of 1
using a historical min & max.
