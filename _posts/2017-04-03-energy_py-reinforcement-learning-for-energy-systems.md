---
title: 'energy_py - reinforcement learning for energy systems'
date: 2017-04-03
categories:
  - Energy
  - Reinforcement Learning
classes: wide
excerpt: An introduction to the open source energy focused reinforcement learning library energy_py.

---

**energy_py is reinforcement learning for energy systems** - [GitHub](https://github.com/ADGEfficiency/energy_py)

energy_py is a tool for running reinforcment learning experiments with virtual energy environments.  This is the first step before using agents with real world environments.

energy_py supports this goal by providing a **collection of agents, energy environments and tools to run experiments.**

GitHub is full of amazing repos with high quality implementations of agents and environment for classic control, video games and robotics.  energy_py contributes to this by suppling environments that simulate real world energy systems.

Simulation is required because

1. modern RL is sample inefficient - simulation is required for learning
2. proving that agents are able to solve energy problems
3. finding the best configuration of agent for specific energy problems

## What is reinforcement learning

![Figure 1]({{"/assets/energy_py/sl_usl_rl.png"}})

Reinforcement learning is the branch of machine learning where an agent **learns through action**.  It's about making good decisions.  

It’s quite different from supervised learning. In supervised learning we start with a big dataset. We train a model to replicate the patterns found in this dataset so that we can make predictions.  

In reinforcement learning we start out with no data. The agent generates data (sequences of experience) by interacting with the environment. The agent uses it’s experience to learn how to interact with the environment. In reinforcement learning we not only learn patterns from data, we also generate our own data.

We also create our own targets.  Data collected in reinforcement learning are sequences of experience - transitions between states, the actions taken and rewards recieved.  The experience tuple `(s,a,r,s')` has no implicit target.  Agents must label experience (for example, a Bellman target) in order to be able to learn from it.

If you are interested in learning more about reinforcement learning, I would recommend these resources

- [Li (2017) Deep Reinforcement Learning: An Overview](https://arxiv.org/pdf/1701.07274.pdf)
- [Sutton & Barto - Reinforcement Learning: An Introduction - 2nd Edition (in progress)](http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf)
- [UCL video lectures by David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

I also teach a two day introduction to reinforcement learning course at Data Science Retreat in Berlin.  You can find the [course materials on GitHub](https://github.com/ADGEfficiency/dsr_rl).  **I am always happy to come give this course for free at interesting companies and universities**.

## Why reinforcement learning in energy?

Optimal operation of energy assets is challenging. Our current energy transition is making this difficult problem even harder.  The rise of intermittent generation is introducing uncertainty on the generation and demand side.  Intelligent operation of clean technologies such as batteries and demand side flexibility are required to smooth out renewables.

Today many operators use rules or abstract models to dispatch assets. A set of rules is not able to guarantee optimal operation in many energy systems.

Optimal operating strategies can be developed from abstract models. Yet abstract models (such as linear programming) are often constrained. These models are limited to approximations of the actual plant.  Reinforcement learners are able to learn directly from their experience of the actual plant. These abstract models also require significant amount of bespoke effort by an engineer to setup and validate.

Neural network powered machine learning has achieved impressive results in computer vision and natural language processing.  Modern reinforcement learning has also achieved landmark results.

Reinforcement learning offers a solution that can learn across a variety of tasks - including control of non-linear systems that require long term planning.

## Challenges

The first and most important is safety. Safety is the number one concern in any engineering discipline.

I believe that by reinforcement learning should be first applied on as high a level of the control system as possible. This allows the number of actions to be limited and existing lower level safety & control systems can remain in place. The agent is limited to only making the high level decisions operators make today.

There is also the possibility to design the reward function to incentivize safety. A well-designed reinforcement learner could actually reduce hazards to operators. Operators also benefit from freeing up more time for maintenance.

Modern reinforcement learning has achieved impressive results - notably DQN and AlphaGo.  Yet modern reinforcment learning is sample inefficient.  This inefficiency means that simulation is required for learning.  Generalization from simulation to real world is a challenge.

energy_py is one step on a long journey of getting reinforcement learners helping us in the energy industry. The fight against climate change is the greatest that humanity faces. Reinforcement learning will be a key ally in fighting it. The project is open source and hosted on [GitHub](https://github.com/ADGEfficiency/energy_py).

## Design choices

Inspiration for the design of energy_py environments comes from [OpenAI gym](https://github.com/openai/gym).

The ultimate goal for energy_py is to have a collection of high quality implementations of different reinforcement
learning agents.  Currently the focus is on implementing a single high quality implementation of DQN and it's
extensions.  A good summary of the DQN extensions is given in [Hessel et. al (2017) Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf).

Creating environments and agents follows a simple API familiar to any users of Open AI gym

```python
import energy_py

TOTAL_STEPS = 1000000

env = energy_py.make_env(
    env_id='BatteryEnv',
    dataset_name=example,
    episode_length=288,
    power_rating=2
)

agent = energy_py.make_agent(
    agent_id='DQN',
    env=env
    total_steps=TOTAL_STEPS
)

while not done:
    action = agent.act(observation)
    next_observation, reward, done, info = env.step(action)
    training_info = agent.learn()
    observation = next_observation
```

Thanks for reading!
