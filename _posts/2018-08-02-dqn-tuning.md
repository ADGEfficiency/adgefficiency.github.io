---
title: 'DQN debugging using Open AI gym CartPole'
date: '2018-08-02'
categories:
  - Machine Learning
  - Reinforcement Learning

---
This article is the second in a series on the new energy_py implementation of DQN.  (The first post documents the debugging process and also a bit of hyperparameter tuning)[https://adgefficiency.com/dqn-debugging/].  This post continues the emotional hyperparameter tuning journey where the first post left off.

These posts follow a problem-hypothesis struture.  Often the speed between seeing cause and effect is quick in computer programming.  In reinforcement learning, the long training time means that it makes more sense to think about the problem before starting another iteration of experiments.

## problem - stability

The major problem at this point was was instability.  The agent was often able to solve the CartPole-v0 environment (Open AI consider this environment solved when an average over the last 100 episodes of 195 is reached).

The issue is with the stability of the policy - the agent often quickly learns to forget and collapses to a poor policy. 

I had a number of hypotheses at this stage about what might be causing the instability
- overestimation bias
- batch size 
- neural network size
- target network update 

I choose to investigate the effects in a somewhat scientific process.  This kind of interactive tuning can be faster than grid searching if the debugger is skilled.  Another benefit is on the tuner - they gain inuition and understanding of how the algorithm is working.

## hypothesis - overestimation bias 

DQN is an agressive algorithm - both in terms of acting (where the argmax will always select the 'best' action, even if the Q value is only slightly higher than another action) and in terms of learning (where the target is formed using a maximum over all possible next actions).

Double Q-Learning (and it's [deep learning variant DDQN](https://arxiv.org/pdf/1509.06461.pdf)) aims to overcome this overestimation bias by selecting actions according to the online network, but quantifying the return for this action using the target network.  In terms of code DDQN only requires a few small changes from DQN.  

```python
if self.double_q:
    online_actions = tf.argmax(self.online_next_obs_q, axis=1)

    unmasked_next_state_max_q = tf.reduce_sum(
        self.target_q_values * tf.one_hot(online_actions,
				          self.num_actions),
        axis=1,
        keepdims=True
)
```

The hypothesis at this stage was that overestimation bias was causing instability (plus I wanted to try out the DDQN code!).  The main hyperparameters for this experiment were

```python
total_steps=400000
discount=0.99
tau=0.001
learning_rate=0.0005
learning_rate_decay=0.1
epsilon_decay_fraction-.05
memory_fraction=0.2
double_q=True
gradient_norm_clip=1.0
batch_size=32
layers=5,5,5
```

The results across three random seeds are shown below in Figure 1.

![fig1]({{ "/assets/dqn_debug_2/fig1.png"}}) 
**Figure 1 - DDQN experiments**

Figure 1 shows that the instability problem still exists.  

## hypothesis - neural network size

In DQN the online network is changed at each step by minimizing the difference between the Bellman target and the current network approximation of `Q(s,a)`.  The hypothesis was that if the batch is too small (previously I was using `batch_size=32`) then the distribution of the data per batch won't be smooth.

Because of the nature of the CartPole environment, this becomes more of an issue as the agent learns, because the episode length increases.  As the episode length increases the relative amount of terminal states will decrease.  I increased the batch size to `256` to see what effect this would have concurrent with the DDQN change.

The original DeepMind paper used a batch size of TODO, but as they were working with images, they are restricted to smaller batches in order to fit the batch on a GPU.  We are using state-actions that are small numpy arrays (and not training on GPU).

There is a relationship between batch size and learning rate - larger batch sizes should allow higher learning rates (because the gradient update is higher quality).

For these simple reinforcement learning problems the idea is that smaller neural networks should be preferred, as they have less change to overfit.

One reason why a small neural network could cause instability is a lot of aliasing i.e. sharing of weights.  This sharing of weights means that changes to one part of the network end up disrupting the understanding encoded in another part of the network.

Up until now I have been using a neural network with three layers, 5 nodes per layer.  My hypothesis was that even though this is not a lot of nodes, because the network has three layers there will be a lot of aliasing (i.e. changes in the input layer affect all the layers below it).  


Updated hyperparameters
```
layers=256,256
batch_size=256
```

![fig2]({{ "/assets/dqn_debug_2/fig2.png"}}) 
**Figure 2 - Larger batch size and neural network**

The batch size and neural network size changes do seem to have improved stability - the agents are able to stay at the maximum reward of 200 for a lot longer.  

The issue is that later in the experiment, the agents forget everything they have learnt and collapse to poor quality policies.  Run 1 is especially disappointing! 

## hypothesis - the target network update

The target network update is an obvious place where instability could arise. There are two common strageties for updating the target network

In the 2013 and 2015 DeepMind Atari work they periodically update the target network weights every `C` steps - in the 2015 paper `C=TODO`. 

Another stragety is to smoothly copy the online network weights to the target network at each step using a parameter `tau=0.001`.  This is the stragety that I've been using so far.

One issue with the `tau=0.001` stragety is that perhaps a neural network has to be taken as a whole.  Because of the massive connectivetivity, change the value of just one weight might lead to drastic changes in other parts of the network.  While the `tau=0.001` stragety seems on it's face to be a smoother and more gradual update, in fact it might cause instability.

Luckily the new energy_py implementation of DQN can eaisly handle both, by setting two parameters

```
#  parameters to update a little bit each step
tau=0.001
update_target_net=1

#  parameters to fully copy weights every 5000 steps 
tau=1.0
update_target_net=5000
```

I setup the same three experiments using the following hyperparameters.  I reduced the size of the neural network as well.

```
batch_size=256
layers=128,128
tau=1.0
update_target_net=5000
```

![fig3]({{ "/assets/dqn_debug_2/fig3.png"}}) 
**Figure 3 - Different target net update strategy, updating the target network fully every 5000 steps.  Note the different scale on the y-axis!** 

![fig35]({{ "/assets/dqn_debug_2/fig3.5.png"}}) 
**Figure 3.1 - The target network weights during training.  Note how the weights stay fixed for a number of steps** 

Figure 3 shows that this target net update stragety slows learning.  This is expected - rather than taking advantage of online network weight updates as they happen, the agent must wait until the target network reflects the agent's new understanding of the world.  Run 2 is especially promising, with the agent able to perform well for around 100k steps.

The main hyperparameter with this style of target network updating is the number of steps between updates.  I ran a set of experiments with a quicker update (`update_target_net=2500`) to see if I could get learning to happen a bit quicker and maintain a bit of stability.  The results for these runs are shown in Figure 4.

![fig4]({{ "/assets/dqn_debug_2/fig4.png"}}) 
**Figure 4 - Updating the target network every 2500 steps seems to worsen performance** 

Looks like that didn't work!  By this point I started to get a bit concerned that I was torturing hyperparameters a bit too much (leading to a form of selection bias where I 'overfit' the implementation of DQN to this specific environment).  So rather than continuning my (somewhat) scientific approach of changing only one thing at a time, I decided to try to combine all the lessons I had learnt and train the best agent I could.  I gave myself two more experiments until I would force myself to stop.

Given that reducing the steps between target network updates seems to reduce performance, I decided to set `update_target_net=10000`.  I also increased the learning rate and decreased the size of the neural network.  Both of these changes were in the context of looking at lot of other impelmentations of DQN for the Cartpole environment.

I increased the learning rate to `0.001` to take advantage of the large batch size (`256`).  A larger batch size should allow higher learning rates, as the gradient updates are higher quality.  I reduced the size of the neural network to `64, 32` based on the feedforward structures I saw in other implementations.

`slower_update_lr_2`

![fig5]({{ "/assets/dqn_debug_2/fig5.png"}}) 
**Figure 5 - Results of the less frequent target net update, larger learning rate, smaller neural network**

## the final run (finally!)

While preprocessing Bellman targets is often given as advice for reinforcement learning, many DQN implementations (including the DeepMind 2015 implementation) don't do any processing of the Bellman target.  For this reason I decided to play around with the batch norm layer a bit.

A reminder of how batch norm is used with the Bellman target in the energy_py implementation of DQN

```
self.bellman = self.reward + self.discount * self.next_state_max_q

bellman_norm = tf.layers.batch_normalization(
    tf.reshape(self.bellman, (-1, 1)),
    center=False,
    training=True,
    trainable=False,
)

error = tf.losses.huber_loss(
    tf.reshape(bellman_norm, (-1,)),
    self.q_selected_actions,
    scope='huber_loss'
)
```

I changed the batch norm arguments to default (i.e. `training=False`).  In the `tf.layers` implementation of batch norm this means that the layer will use historical statistics to normalize the batch.  The effect of this on the Bellman target is quite dramatic.

```
self.bellman = self.reward + self.discount * self.next_state_max_q

bellman_norm = tf.layers.batch_normalization(
    tf.reshape(self.bellman, (-1, 1)),
    center=False,
    training=True,
    trainable=False,
)
```

Figure 6 shows the batch normed Bellman target with `training=True` (i.e. normalize each batch using the batch statistics) for the runs show in Figure 5 above.  Figure 7 shows the Bellman target 

![fig6]({{ "/assets/dqn_debug_2/fig6.png"}}) 
**Figure 6 - The effect of `training=True` on the Bellman target.  As training progresses the Bellman target remains in a range of around 2 to -4**

![fig7]({{ "/assets/dqn_debug_2/fig7.png"}}) 
**Figure 7 - The effect of `training=False` on the Bellman target for Run 2 of the final run.  As training progresses the Bellman target increases with every update of the target network.  Also note how the batch norm operation is not changing the Bellman target!**

So it seems that using the batch norm layer with default parameters is essentially like not using it at all.

For the final run I increased the number of steps from 400k to 1000k.  While this is a massive number of steps, the reality of reinforcement learning is that it is sample inefficient.  While it's possible that torturing hyperparameters can help with the sample inefficieny, another valid fix for sample inefficieny is more samples.  This is equivilant to getting more data in a supervised learning setting.

## IS THIS BECAUSE I HAVE TRAINABLE=FALSE!

![fig8]({{ "/assets/dqn_debug_2/fig8.png"}}) 
**Figure 8 - Results for the final run**

So after a whole heap of CPU time - what can we say about the final performance?  All three runs do solve the environment (this enviroment is considered solved when an an average of 195 is achieved over the last 100 episodes).  

All three runs also show catastrophic forgetting - i.e. after they solve the environment they all managed to forget their good work.  What then is the solution to this?  

## early stopping

In supervised learning it is common to keep track of the validation loss (i.e. the loss on a test set) during training.  Once this validation loss starts to increase, training is stopped (known as early stopping).

This is a potential solution to the stability problem.  Rather than training forever, once good performance has been achieved training should stop.  

If you think about how we learn skills in the real world, we do often use early stopping.  Once we learn handwriting write to a decent level we commonly don't continue to train the skill.  Instead we fall into a repeated pattern of writing in the same way for the rest of our lives.

It's perhaps a bit cruel to continute to force gradient updates on an agent when it has solved an environment.  A small enough learning rate should solve this problem but it's likely that the learning rate would be so small that there is no point doing the update in the first place.

I found one mention of early stopping in the reinforcement learning literature - I show the paragraph in full below

![fig9]({{ "/assets/dqn_debug_2/fig9.png"}}) 
**Figure 9 - https://nihit.github.io/resources/spaceinvaders.pdf**

## takeaways

This process of debugging has been an emotional journey.  It's something I'm very glad I did - Youshia Benjio often talks about how it's crucial to ones development as a machine learning professional to spend the time training and debugging algorithms.  

While you can learn alot from reading, there is no replacement for getting your hands dirty.  Spending the time training a whole bunch of models and hypothesizing what the effects of hyperparameters might be is time well spent.

When I was reviewing other implementations of DQN around GitHub or on blogs, I often saw a successful iteration in the blog post then a number of comments saying that the implementation didn't work.


