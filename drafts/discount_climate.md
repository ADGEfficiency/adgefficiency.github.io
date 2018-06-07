Climate change is the most challenging problem of our time.  One challenges stems in our psychology - the use of a discount rate in decision making.

Like many fundamental ideas the concept of a discount rate appears in multiple places.  In finance the discount rate increases the value of money the easier it can be earnt.  In reinforcement learning it serves multiple purposes - making the mathematics of the Markov Decision Process (MDP work) as well as influencing agent behaviour.

This post will explain the role of the discount rate in the context of climate, and then expand upon the concept in terms of reinforcement learning.

---

## what is a discount rate
The discount rate reduces the magnitude of future benefits or costs relative to the present.  A discount rate is what allows a smoker to ignore the future risks of lung cancer.  A discount rate is not always a negative thing - like many cognitive biases it exists for a reason.

Human beings make decisions using a discount rate.  In the context of psychology this is known as hyperbolic discounting.   Studies have shown that human beings actually have a temporally variable discount rate.  The discount rate changes depending on the environment (i.e. the presence of potential mates).

## discount rate and the climate
In the context of climate change, the discount rate mans that we we naturally reduce the future damage of climate change compared with the benefits we get from fossil fuels today.

As human beings our decision making mechanisms fundamentally reduce the perceived penalty of climate change.  When we are talking about damage that will occur in 2050 or 2100 our decision making apparatus likely discounts the damage so much that the benefit of using fossil fuels vastly outweighs it.

The reality of our temporal discount rate is just another challenge in the climate change problem.  Technical, political and business model challenges dominate the climate change conversation - the discount rate is a challenge that requires us to overcome our evolution shaped psychology.

## making the maths work
The Markov Decision Process is the mathematical foundation of the reinforcement learning problem.  The agent and environment interact through a cyclic process of the agent taking actions and the environment responding.  This horizon of this process can be a fixed length, a variable but finite length or an infinite length.

The discount rate unites episodic and infinite horizon environments into the same mathematical framework.  To explain why we need to introduce the value function.

The value function estimates the future expected discounted reward.

For an infinite horizon problem, the sum of future rewards approaches infinity.  This would make the value function worthless!

The discount rate allows us to keep the sum of future rewards finite.  By using a discount rate less than one the future rewards become a geometric series, with each reward worth less than the rewards received earlier.

Influencing agent behaviour
The reason we want finite value function is because they help us to act.  A value function quantifies the value of states and actions.  If we know the expected discounted reward for each action, we can select the optimal action by selecting the action with the highest associated value.

The discount rate will change the value function.  A high discount rate (0.9 - 0.99) will increase the value of future rewards, making the agent take actions that have long term benefit.

A low discount rate will increase the value of present rewards, making the agent think short term.  The discount rate has the equal and opposite effect on negative rewards.

Thanks for reading!
