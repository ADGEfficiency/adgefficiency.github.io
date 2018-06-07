---
title: 'HHV vs LHV - GCV vs NCV'
date: '2018-07-07'
categories:
  - Machine Learning
  - Reinforcement Learning
---

This post details the debugging process I went through for the new implementation of DQN in energy_py.  This work was performed on the dev branch at [this commit](https://github.com/ADGEfficiency/energy_py/tree/46fd1bf36f744918c962539eb8a84df96102d930).

The work was done using the energy_py wrapper around the Open AI gym CartPole-v0 environment.

The ideas behind documenting this debug process come from the blog post [Lessons Learned Reproducing a Deep Reinforcement Learning Paper](http://amid.fish/reproducing-deep-rl).  The post reccomends keeping a detailled log of the debugging process.  

Another purpose of this post is to show the logic behind a successful debugging, the kinds of silly errors that can eaisly be made and to show how CartPole often performs using DQN. 

This is the third iteration of DQN that I've built - this one was significantly infulenced by the [Open AI baselines implementation of DQN](https://github.com/openai/baselines/tree/master/baselines/deepq).

## the setup

While I think obsession over what tools (i.e. which editor to use) is unhelpful, I do think that anyone who takes their work seriously should take pride in the toolset they use.  For this experiment I used a combination of tmux, vim, an info & debug log and Tensorboard to try get an understanding of whats going on.

I used two tmux windows, one that kept track of the experiment and another with the `energy_py/agents/dqn.py` script open for editing in vim.  The experiment window shows both the `info.log` and `debug.log`.

![]({{ "./assets/debug_dqn/tmux_setup.png"}}) 
**tmux setup with the left pane running the script and showing the info log, the top right pane showing the debug log using `tail -f debug.log` and a Tensorboard server running in the bottom right pane**

![]({{ "./assets/debug_dqn/vim_setup.png"}}) 
**tmux setup with the left pane running the script and showing the info log, the top right pane showing the debug log using `tail -f debug.log` and a Tensorboard server running in the bottom right pane**

## debugging code

For the debug process I wrote a barebones implementation of an experiment under the `if __name__ == '__main__':` block in `energy_py/agents.dqn.py`.  It exposes a lot of the functionality that is all taken care of automatically when using the `experiment()` function in energy_py (`from energy_py import experiment`).

As per common advice 

```python
import random
from energy_py.scripts.experiment import Runner
from energy_py.scripts.utils import make_logger

make_logger({'info_log': 'info.log', 'debug_log': 'debug.log'})
discount = 0.99
total_steps = 400000

seed = 15
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

env = energy_py.make_env('Battery')

with tf.Session() as sess:
agent = DQN(
    sess=sess,
    env=env,
    total_steps=total_steps,
    discount=discount,
    memory_type='deque',
    act_path='./act_tb',
    learn_path='./learn_tb',
    learning_rate=0.001,  
    decay_learning_rate=1.0,
)

runner = Runner(sess,
		{'tb_rl': './tb_rl',
		 'ep_rewards': './rewards.csv'},
		total_steps=total_steps
		)

step = 0
while step < total_steps:

    done = False
    obs = env.reset()
    while not done:
	act = agent.act(obs)
	next_obs, reward, done, info = env.step(act)

	runner.record_step(reward)
	agent.remember(obs, act, reward, next_obs, done)

	agent.learn()

	obs = next_obs
	step += 1

    runner.record_episode()
```

## problem - reward collapsing after exploration 

![fig1]({{ "./assets/debug_dqn/fig1.png"}}) 
**Figure 1 - Collapsing reward after exploration is over**

When using an epsilon greedy exploration policy, early stages of the experiment are mostly randomly selected actions.  For CartPole this ends up being an average reward per episode of between 20 - 30.  For a working implementation the episode returns will stay in this range and start to increase as the agent learns.

What I was seeing was a drop in average reward to around 10 per episode after exploration was over.  This suggests that the argmax over Q(s,a) was selecting the same action each time, resulting in a poor policy that quickly failed the CartPole task.

##  hypothesis - are my weights changing
The idea was that if the online network weights were never changed, then the argmax across the online network might select the same action in every state - leading to the behvaiour we saw.

To do this I added the weights as histograms in TensorBoard by indexing a list of parameters for the online and target networks.  This is a bit hacky - Tensorflow didn't like iterating over this list so I just indexed out the last layer weights and biases for both networks.

```python
self.act_summaries.extend([
    tf.summary.histogram(self.online_params[-1].name, self.online_params[-1]),
    tf.summary.histogram(self.online_params[-2].name, self.online_params[-2]),
    tf.summary.histogram(self.target_params[-1].name, self.target_params[-1]),
    tf.summary.histogram(self.target_params[-2].name, self.target_params[-2]),
		       ])
```

This allows visibility of the weights at each step - and we can see that both the weights and biases are being changed at each step.

![fig2]({{ "./assets/debug_dqn/fig2.png"}}) 
**Figure 2 - Online network weights changing**

##  hypothesis - how am I changing the weights - aka what is the target

In DQN learning is done by minimizing the difference between predictied Q values and Bellman targets.  Creation of the Bellman target is core to the algorithm and a common place for errors.

Reinfocement learning can be though of as a data generation process - interacting with the environment generates sequences of experience tuples of `(s, a, r, s')`.  In order to learn from this data we need to label it - in DQN this labelling process is doing by creating a Bellman target for each sample in a batch.  This then allows supervised learning to be used to fit our predicted `Q(s,a)` to the target. 

From experience with DQN and CartPole I expected to see a inflation in the Q values.  This optimism comes from the argmax operation over `Q(s',a)` in the Bellman target.  When I took a look at the Bellman target I saw something quite different - an increase until a very small value of around 2.0.  Since rewards for CartPole are +1 for each step, this meant that the argmax across `Q(s',a)` was approximately 1.0 as well.

![fig3]({{ "./assets/debug_dqn/fig3.png"}}) 
**Figure 3 - The Bellman target showing a plateau at around 2**

We know can see that the target doesn't seem right - we can check the loss to see if this improperly formed target is being learnt.  Even though DQN uses a target network for the approximation of `max Q(s',a)`, this approximation is still infulenced by the online network via the target net copy operations.

Taking a look at the loss function we can see that the agent is learning to fit this improperly formed Bellman target

![fig4]({{ "./assets/debug_dqn/fig4.png"}}) 
**Figure 4 - A nice looking loss function**

##  hypothesis - the target isn't being created correctly

The Bellman target has two parts - the reward and the discounted max value of the next state `max Q(s',a)`. Getting the reward is as simple as unpacking the batch, meaning the error is most likely in the estimated maximium value of the next state.

One key part of Q-Learning is seting this value to zero for terminal states.  In terminal states the discounted return is simply the reward.  This means that for terminal states, we need to mask out the return from the next state.

I added this part of the Bellman equation to Tensorboard - both the unmasked and masked `Q(s',a)` values.

![fig5]({{ "./assets/debug_dqn/fig5.png"}}) 
**Figure 5 - The unmasked and masked approximations of `max Q(s',a))`**

As exepected none of the unmased values are zero, because they are maximums across all possible actions. But looking at the masked values, it seemed that far too many were zero!  If our batch is sampled well from memory, we would expect the distribution of terminals (and associted zero `Q(s',a)` values) to match the distribution we see in training.  For CartPole with an episode length of around 20, we would expect to see 20 times as many non-zero values as zeros.  From Figure 5 we see the opposite.

Looking at how I was doing the masking the error became clear - I had the masking around the wrong way!  Terminal is a boolean that is `True` when the episode is over, and `False` otherwise.

```python

#  the corrected masking of terminal states Q(s',a) values
next_state_max_q = tf.where(
	self.terminal,
	tf.zeros_like(unmasked_next_state_max_q),
	unmasked_next_state_max_q,
	name='terminal_mask'
	)
```

After making this change, the distribution of masked `Q(s',a)` values looks a lot better

![fig6]({{ "./assets/debug_dqn/fig6.png"}}) 
**Figure 6 - Proper masking out of `Q(s',a)`**

Now after running the experiment we see the increase in Q values that I saw with previous implementations of DQN.  This optimism is a function of the agressive and positively biased maximum value done in creating the Bellman target.  We know this because a pessimistic target (which we had previously with our incorrect `tf.where`) doesn't see this optimism.

![fig7]({{ "./assets/debug_dqn/fig7.png"}}) 
**Figure 7 - Increasingly optimistic Bellman targets and a loss function that now reflects the non-staionary target creation** 

The loss function in Figure 7 is maybe scary for supervised learners - a increasing loss function means that your errors in predicting the target are getting worse.  In the context of reinforcement learning this loss function is a commentary on the non-stationary target being used to train.  Increases in loss function can actually be seen as a good thing, as this means the agent is suprised about how much return it should expect for a particular sample of experience.

## hypothesis - improperly scaled Bellman target

Figure 7 shows that the Bellman target is rather large.  For gradient based methods the optimizer usually expects to see inputs and outputs in the range of -1 to 1 - hence the use of standardization and normalization using training set statistics in supervised learning.

In reinforcment learning we have the problem of not know what are good approximations for the statistics of `Q(s,a)`.  To combat this I used a Tensorflow batch normalization layer to process the Bellman target before it is used in the loss function.

There are three different implementations of batch norm in Tensorflow - tf.nn.batch_normalization, tf.layers.batch_normalization or tf.contrib.layers.batch_norm.

```python
bellman = self.reward + self.discount * next_state_max_q

#  batch norm requires some reshaping with a known rank
#  reshape the input into batch norm, then flatten in loss
#  training=True because we want to normalize each batch

bellman_norm = tf.layers.batch_normalization(
	tf.reshape(bellman, (-1, 1)),
	training=True,
	trainable=False,
	)

error = tf.losses.huber_loss(
	tf.reshape(bellman_norm, (-1,)),
	q_selected_actions,
	scope='huber_loss'
	)
```

One of the hyperparameters in using batch norm is whether to use accumulated statistics across multiple batches (`training=False`) or to process based on statistics from only the current batch (`training=True`).  My intuition is that processing only across the batch removes the difficulty of figuring out how best to remeber and forget statistics, and just focus on favouring some actions within the batch more than others.

TODO PICTURE OF THE BATCH NORMED TARGETS !!!

After making all of these changes the first signs of life appeared

![fig8]({{ "./assets/debug_dqn/fig8.png"}}) 
**Figure 8 - It's alive!** 

## tuning

After seeing some initial signs of life, I went back over some key hyperparameters and manually tuned them.  One of the new features of this energy_py DQN implementation is the ability to decay learning rate.  Historically I used a fixed learning rate between `0.001 - 0.0001`. 

RUN NEW EXPT


## a few learning curves

The point of including these are to show how stochastic the learing is - only difference is the random seed!!
