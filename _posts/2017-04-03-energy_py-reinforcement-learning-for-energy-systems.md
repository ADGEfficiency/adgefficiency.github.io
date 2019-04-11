---
title: 'energy-py - reinforcement learning for energy systems'
date: 2017-04-03
categories:
  - Energy
  - Reinforcement Learning
classes: wide
excerpt: An introduction to the open source energy focused reinforcement learning library energy-py.

---

energy-py is a tool for running reinforcement learning experiments with energy environments.  The project is open source and hosted on [GitHub](https://github.com/ADGEfficiency/energy-py).

energy-py provides a **collection of agents, energy environments and tools to run experiments**.

The goal of the project is to demonstrate and experiment with the ability of reinforcement learning to operate simulations of energy systems.  This is the first step in using reinforcement learning to operate real energy systems.

## What is reinforcement learning

![Figure 1]({{ "/assets/energy-py/sl_usl_rl.png" }})

Machine learning can be separated into three broad categories, based on the type of feedback signal available to the learner.  In reinforcement learning an agent learns through action.  It's about making good decisions.  

It’s quite different from supervised learning. In supervised learning we start with (hopefully big) dataset.  We train a model to predict the target associated with each sample.

Reinforcement learning faces two additional challenges.  The first is data generation.  Data is generated through the agent taking actions in the environment.  The agent must take care to explore the state space when it is learning and then exploit it once it has learnt.

The second challenge is target creation.  Data collected in reinforcement learning are sequences of experience - transitions between states, the actions taken and rewards received.  The experience tuple `(s,a,r,s')` has no implicit target.  Agents must label experience (for example, a Bellman target) in order to be able to learn from it.

If you are interested in learning more about reinforcement learning, I would recommend these resources

- [Li (2017) Deep Reinforcement Learning: An Overview](https://arxiv.org/pdf/1701.07274.pdf)
- [Sutton & Barto - Reinforcement Learning: An Introduction - 2nd Edition (in progress)](http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf)
- [UCL video lectures by David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

I also teach a two day introduction to reinforcement learning course at Data Science Retreat in Berlin - you can find the [course materials on GitHub](https://github.com/ADGEfficiency/dsr_rl).

## Why reinforcement learning in energy?

Optimal operation of energy assets is challenging. Our current energy transition is making this difficult problem even harder.  The rise of intermittent generation is introducing uncertainty on the generation and demand side.  

Intelligent operation of clean technologies such as batteries and demand side flexibility are required to smooth out renewables.

Today many operators use rules or abstract models to dispatch assets. A set of rules is not able to guarantee optimal operation in many energy systems.  Optimal operating strategies can be developed from abstract models.   Traditionally these models were linear or required hand engineering.

Neural network powered machine learning has achieved impressive results in computer vision and natural language processing.  Modern reinforcement learning has also achieved landmark results.

Reinforcement learning offers a solution that can learn across a variety of tasks - including control of non-linear systems that require long term planning.  

Energy systems also have a key feature of a well-defined reward signals such as energy cost or carbon emissions.  The lack of a well defined reward signal is a limiting factor in applications such as self driving cars - in energy we can take advantage of reward signals that align with our goals.

## Challenges

The first and most important is safety. Safety is the number one concern in any engineering discipline.  Reinforcement learning should be first applied on as high a level of the control system as possible. This allows the number of actions to be limited and existing lower level safety & control systems can remain in place. The agent is limited to only making the high level decisions operators make today.

Reward functions can also be designed to favour safety.  A well-designed reinforcement learner could actually reduce hazards to operators. Operators also benefit from freeing up more time for maintenance.

Modern reinforcement learning has achieved impressive results - notably DQN and AlphaGo.  Yet modern reinforcement learning is sample inefficient.  This inefficiency means that simulation is required for learning.  Building useful simulation models is another grand challenge for sample inefficient learners.  Generalization from the simulators to real world is also a massive challenge.

## Design 

Inspiration for the design of energy-py environments comes from [OpenAI gym](https://github.com/openai/gym).

The ultimate goal for energy-py is to have a collection of high quality implementations of different reinforcement
learning agents.  Currently the focus is on implementing a single high quality implementation of [DQN and it's extensions](https://arxiv.org/pdf/1710.02298.pdf).

Creating environments and agents follows a simple API familiar to any users of Open AI gym.

```python
import energypy

TOTAL_STEPS = 1000000

env = energypy.make_env(
    env_id='battery',
    dataset_name=example,
    episode_length=288,
    power_rating=2
)

agent = energypy.make_agent(
    agent_id='dqn',
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
