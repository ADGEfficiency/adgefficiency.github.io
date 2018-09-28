---
title: "We won't solve climate change - and why it's ok"
date: 2018-09-28
categories:
  - Energy
excerpt: Our evolution shaped psychology is the reason why.
mathjax: true

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

One of the most curious aspects of the clean-tech space is professionals dedicated to solving the climate problem (myself included) making personal decisions that are well known to be carbon intense (such as flying or eating meat).  What allows this level of inconsistency?

Technical, political and business model challenges dominate the climate change conversation - but our evolution shaped psychology poses a more fundamental problem.

## The discount rate

A discount rate **reduces the magnitude of future benefit or harm relative to the present**.  It encourages making decisions that get reward sooner - even if this leads to equal or greater amounts of harm in the future.  A discount rate is what allows a smoker who knows the risk of lung cancer to keep smoking.

![The discount rate reduces the perceived benefit or harm of actions relative to the short term benefits]({{ "/assets/discount/fig1.png" }})


Like other mental models, the discount rate appears across disciplines.  One is **finance**.  The discount rate decreases the value of future cash flows relative to today. Cash is more valuable today than in the future because I can invest it and generate more cash.

Another is **reinforcement learning**.  The discount rate is one element of the tuple that makes up a Markov Decision Process (MDP) - the mathematical framework for reinforcement learning problems.

*The tuple defining a Markov Decision Process*

$$ (\mathcal{S}, \mathcal{A}, \mathcal{R}, P, R, d_0, \gamma) $$

In an MDP the agent and environment interact through a cyclic process of the agent taking actions and the environment responding with a scalar reward and the next state.  How long this interaction occurs is known as the **horizon**.  The horizon can be a fixed length, a variable but finite length or an infinite length.

In reinforcement learning the goal of the agent is to maximize expected **discounted future reward** over the horizon of the MDP.  The sum of future rewards is known as the return ($$G_t$$) and is also formalized as a value function ($$V_{\pi}(s)$$).  The discount factor serves to exponentially decay the value of future rewards relative to the current time step.

*Both the value function and the return are equal to the expected discounted future reward*

$$ V_{\pi}(s) = \mathbf{E}[G_t | s_t] = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... = \sum_{t=0}^{\infty} \gamma^{t} r_{t} $$

There are multiple reasons why a discount rate is used in reinforcement learning.  The most fundamental is to unite finite and infinite horizon problems into the same mathematical framework.

If we defined return as the undiscounted sum of future rewards, in an infinite horizon problem the return for all actions would be infinite!  This would make the value function useless for making decisions.  By discounting future rewards with a discount factor less than one, we turn the future expected return into a **geometric series** - an infinite series with a finite sum.

## Why the discount rate makes the climate problem intractable

A discount rate is used in **human decision making**.  Psychologists have found that humans have a variable discount rate - men shown pictures of attractive women become less willing to forgo cash in the short term for larger amounts of cash in the future.

Humans struggle to accurately balance the future damage of using fossil fuels versus the benefits we get today.  It's built in to us that even in face of massive amounts of long term damage to the planet, biodiversity and the quality of lives, we will make decisions that give us smaller amounts of immediate benefit.

This explains why even climate warriors make climate unfriendly decisions.  They correctly understand the magnitude of climate change, but discounting means that the benefit of eating meat and flying outweighs the future damage.

## Why it's ok if we don't solve climate change

![]({{ "/assets/discount/fig2.png" }})

Natural selection is the reason we have this bias - and it was likely crucial to us being where we are today.  Like many cognitive biases in the correct context it is useful - natural selection wouldn't have favoured it otherwise.

Natural selection and evolution also provide the reason why the problems caused by the discount rate bias are ok.  The wonderful biodiversity that exists on Earth today isn't unique - in fact the planet has seen five major extinction events.  Each time nature responds with another set of organisms specific to current environment of the planet.  Hopefully nature's next creations will be better able to balance immediate benefits with future damage.

This isn't to say that climate change isn't a tragedy. The cruelty of sudden changes in climate on animals who can have no way of understanding what is happening is heartbreaking.  But I believe that human beings are a part of this - while on the surface we think we have the ability to stop climate change, the whole point of this article is that natural selection has not given us this ability.

## You should keep working on the climate problem

By this point you might be disillusioned.  If our psychology actively favours short term thinking, how can we ever hope to solve a long term problem like climate change?

The discount rate problem may be an opportunity.  In order to make better long term decisions we can do three things

1. increase the undiscounted magnitude of future damage
2. bring forward the perceived timing of damage
3. change our discount rate

It's unclear which of these would be most effective.  We already understand the magnitude of future damage, and we are already seeing the impacts today.  Changing our internal discount rates would require a meta-cognitive revolution on a grand scale.

> You have the right to work, but never to the fruit of work. You should never engage in action for the sake of reward, nor should you long for inaction. Perform work in this world, Arjuna, as a man established within himself - without selfish attachments, and like in success and defeat - The Bhagavad Gita

This doesn't mean that I will stop working on decarbonization.  Even if we end up in the 5 degree world (which is where we are headed), **working in the clean-tech space is an honor**.  The people are great, and working towards a positive mission is a privilege.

But I'm not attached to the outcome.  Whatever the future of the CO<sub>2</sub> concentration on this planet - the show will go on.

Thanks for reading!
