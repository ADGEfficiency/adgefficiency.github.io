---
title: 'DQN debugging using Open AI gym CartPole'
date: '2018-07-07'
categories:
  - Machine Learning
  - Reinforcement Learning
classes: wide
excerpt: Debugging the new energy_py DQN reinforcement learning agent.

---

This post details the debugging process I went through for the new implementation of DQN in energy_py.  The experiments ran in this post were on the dev branch at [this commit](https://github.com/ADGEfficiency/energy_py/tree/46fd1bf36f744918c962539eb8a84df96102d930).  By the end of this work the energy_py repo has reached over 500 commits!

![]({{ "/assets/debug_dqn/commits.png"}}) 

![]({{ "/assets/debug_dqn/graph.png"}}) 

The work was done using [the energy_py wrapper](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/envs/register.py) around the Open AI gym **CartPole-v0** environment.  CartPole is an environment I am familiar with and use to prove that an agent can learn a well formed reinforcement learning problem.

The idea of documenting this debug process comes from [Lessons Learned Reproducing a Deep Reinforcement Learning Paper](http://amid.fish/reproducing-deep-rl). This post recommends keeping a detailed log of your debugging and also taking the time to form hypotheses about what might be wrong. This is because of the long lead time between cause and effect for reinforcement learning experiments.

This post shows the logic behind a successful debugging, the kinds of silly errors that can easily be made and to show how CartPole often performs using DQN. It then starts the hyperparameter tuning process, which is continued in the second post, [DDQN hyperparameter tuning using Open AI gym CartPole](https://adgefficiency.com/dqn-tuning/).

This is the third iteration of DQN that I've built - this one was significantly influenced by the [Open AI baselines implementation of DQN](https://github.com/openai/baselines/tree/master/baselines/deepq).

## the dqn rebuild

This is the third major iteration of DQN I've built in energy_py.  Each iteration is a complete rewrite.  

> If you are not embarrassed by the first version of your product, you've launched too late - Reid Hoffman

> I know you don’t hit it on the first generation, don’t think you hit it on the second, on the third generation maybe, on the fourth & fifth, that is when we start talking -  Linus Torvalds

I'm quite proud of how far I've come, and of how poor my first implementation looks to me today.

[version 1](https://github.com/ADGEfficiency/energy_py/tree/d21c3832e9116cba00891361e6777b8b896f9b78)
- built in Keras
- no target network
- structuring the neural network with a single output.  this means n passes are required to predict Q(s,a) for n actions, rather than a single pass in a network with n output nodes (one per action).  it does allow the network to connect to the action as an array, so the network can sense the shape of the action 

[version 2](https://github.com/ADGEfficiency/energy_py/commit/774ff3c9cd63b1b1e50359ab606edc7737121c86)
- built in Tensorflow
- target network implemented
- running act or learn requires running multiple session calls, because the algorithm switches between numpy and Tensorflow for operations
- e-greedy policy only

[version 3](https://github.com/ADGEfficiency/energy_py)
- the current master branch (synced over from dev at [this commit](https://github.com/ADGEfficiency/energy_py/tree/f747f0e10741c33cfa81ac7c8b52ebfc4bdca7e4))
- built in Tensorflow, with a single session call per `agent.action()` and `agent.learn()`
- gradient clipping, learning rate decay
- policy is split out to allow either epsilon greedy or a softmax policy to be used

Two more rebuilds to go...

## the setup

While I think obsession over what tools (i.e. which editor to use) is unhelpful, I do think that anyone who takes their work seriously should take pride in the toolset they use.  For this experiment I used a combination of tmux, vim, an info & debug log and tensorboard to try get an understanding of whats going on.

I used two tmux windows, one that kept track of the experiment and another with the `energy_py/agents/dqn.py` script open for editing in vim.  The experiment window shows both the `info.log` and `debug.log`.  The debug log moves too fast to be viewed but is useful for seeing if the agent is working.

![]({{ "/assets/debug_dqn/tmux_setup.png"}}) 
tmux window one setup
- left pane running the script and showing the info log
- the top right pane showing the debug log using `tail -f debug.log` 
- a tensorboard server running in the bottom right pane

![]({{ "/assets/debug_dqn/vim_setup.png"}}) 
tmux window two setup - vim with `agent/dqn.py` open

Switching between tmux windows is as easy at `Ctrl b p`.

## debugging code

For the debug process I wrote a barebones implementation of an experiment under the `if __name__ == '__main__':` block in `energy_py/agents.dqn.py`.  It exposes the functionality that is all taken care of automatically when using the `experiment()` function in energy_py (`from energy_py import experiment`).

Doing this in the same script as the DQN agent means I can easily make changes, and removes dependencies on the rest of the project.

```python
import random

from energy_py.scripts.experiment import Runner
from energy_py.scripts.utils import make_logger

make_logger(
	{'info_log': 'info.log', 
	'debug_log': 'debug.log'}
	)

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

runner = Runner(
	sess,
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

This setup is enough to get logging setup with two log files and tensorboard running.  Three tensorboard writers are used - one for `agent.act()`, one for `agent.learn()` and one for `runner.record_episode()`.  I setup these log files in the local directory.  To view the tensorboard log files I start a server in the same directory that the `dqn.py` script lives in.

```bash
$ cd energy_py/energy_py/agents

$ tensorboard --logdir='.'
```

## problem - reward collapsing after exploration 

![fig1]({{ "/assets/debug_dqn/fig1.png"}}) 

**Figure 1 - Collapsing reward after exploration is over**

When using an epsilon greedy exploration policy, early stages of the experiment are mostly randomly selected actions.  For CartPole this ends up being an average reward per episode of between 20 - 30.  For a working implementation the episode returns will stay in this range and start to increase as the agent learns.

What I was seeing was a drop in average reward to around 10 per episode after exploration was over.  This suggests that the argmax over `Q(s,a)` was selecting the same action each time, resulting in a poor policy that quickly failed the CartPole task.  This policy is worse than a random policy!

##  hypothesis - are my weights changing
The idea was that if the online network weights were never changed, then the argmax across the online network might select the same action in every state - leading to the behavior we saw.

To do this I added the weights as histograms in tensorBoard by indexing a list of parameters for the online and target networks.  This is hacky - tensorflow didn't like iterating over this list so I just indexed out the last layer weights and biases for both networks.

```python
self.act_summaries.extend([
    tf.summary.histogram(self.online_params[-1].name, 
    		         self.online_params[-1]),
    tf.summary.histogram(self.online_params[-2].name, 
    			 self.online_params[-2]),

    tf.summary.histogram(self.target_params[-1].name, 
    		 	 self.target_params[-1]),
    tf.summary.histogram(self.target_params[-2].name, 
    			 self.target_params[-2]),
		       ])
```

This allows visibility of the weights at each step. Figure 2 below shows that both the weights and biases are being changed.

![fig2]({{ "/assets/debug_dqn/fig2.png"}}) 

**Figure 2 - Online network weights changing**

##  hypothesis - how am I changing the weights - aka what is the target

In DQN learning is done by minimizing the difference between predicted Q values and Bellman targets.  Creation of the Bellman target is core to the algorithm and a common place for errors.

Reinforcement learning can be though of as a data generation process - interacting with the environment generates sequences of experience tuples of `(s, a, r, s')`.  In order to learn from this data we need to label it - in DQN this labelling process is doing by creating a Bellman target for each sample in a batch.  This then allows supervised learning to be used to fit our predicted `Q(s,a)` to the target. 

From experience with DQN and CartPole I expected to see a inflation in the Q values.  This optimism comes from the max operation over `Q(s',a)` in the Bellman target.  When I took a look at the Bellman target I saw something quite different - an increase until a small value of around 2.0.  Since rewards for CartPole are +1 for each step, this meant that the max across `Q(s',a)` was approximately 1.0 as well.

![fig3]({{ "/assets/debug_dqn/fig3.png"}}) 

**Figure 3 - The Bellman target showing a plateau at around 2**

We now can see that the target doesn't seem right - we can check the loss to see if this improperly formed target is being learnt.  Even though DQN uses a target network for the approximation of `max Q(s',a)`, this approximation is still influenced by the online network via the target net copy operations.

Taking a look at the loss function (Figure 4) we can see that the agent is learning to fit this improperly formed Bellman target.

![fig4]({{ "/assets/debug_dqn/fig4.png"}}) 

**Figure 4 - A nice looking but wrong loss function**

##  hypothesis - the target isn't being created correctly

The Bellman target has two parts - the reward and the discounted max value of the next state `max Q(s',a)`. Getting the reward is as simple as unpacking the batch, meaning the error is most likely in the estimated maximum value of the next state.

One key part of Q-Learning is setting this value to zero for terminal states.  In terminal states the discounted return is simply the reward.  This means that for terminal states, we need to mask out the return from the next state.

I added this part of the Bellman equation to tensorboard - both the unmasked and masked `Q(s',a)` values.

![fig5]({{ "/assets/debug_dqn/fig5.png"}}) 

**Figure 5 - The unmasked and masked approximations of `max Q(s',a))`**

As expected none of the unmasked values are zero, because they are maximums across all possible actions. But looking at the masked values, it seemed that far too many were zero!  If our batch is sampled well from memory, we would expect the distribution of terminals (and associated zero `Q(s',a)` values) to match the distribution we see in training.  For CartPole with an episode length of around 20, we would expect to see 20 times as many non-zero values as zeros.  From Figure 5 we see the opposite.

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

After making this change, the distribution of masked `Q(s',a)` values looks better.  

![fig6]({{ "/assets/debug_dqn/fig6.png"}}) 

**Figure 6 - Proper masking out of `Q(s',a)`**

As part of the DQN rebuild I added a [suite of tests](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/tests/test_dqn.py) to test the new agent.  Tests include the variable sharing and copy operations, along with checking the Bellman target creation for both DQN and DDQN.

Unfortunately I had repeated the same error with `tf.where` in my test for the Bellman target creation!  I actually wrote a note pointing out the test mirrored the tensorflow code exactly... maybe my subconscious saw something I didn't.

Now after running the experiment we see the increase in Q values that I saw with previous implementations of DQN.  This optimism is a function of the aggressive and positively biased maximum value done in creating the Bellman target.  We know this because a pessimistic target (which we had previously with our incorrect `tf.where`) doesn't see this optimism.

![fig7]({{ "/assets/debug_dqn/fig7.png"}}) 

**Figure 7 - Increasingly optimistic Bellman targets and a loss function that now reflects the non-staionary target creation** 

The loss function in Figure 7 is maybe scary for supervised learners - a increasing loss function means that your errors in predicting the target are getting worse.  In the context of reinforcement learning this loss function is a commentary on the non-stationary target being used to train.  Increases in loss function can actually be seen as a good thing, as this means the agent is surprised about how much return it should expect for a particular sample of experience.

This surprise can be used to improve the agents understanding of actions that are good or bad.

## hypothesis - improperly scaled Bellman target

Figure 7 shows that the Bellman target is rather large.  For gradient based methods the optimizer usually expects to see inputs and outputs in the range of -1 to 1 - hence the use of standardization and normalization using training set statistics in supervised learning.

In reinforcement learning we have the problem of not know what are good approximations for the statistics of `Q(s,a)`.  To combat this I used a tensorflow batch normalization layer to process the Bellman target before it is used in the loss function.

I manually wrote Processor objects to do normalization and standardization in energy_py previously.  Using tensorflow to to the processing will allow me to keep the processing within the tensorflow graph, and mean less code for me to maintain.

There are three different implementations of batch norm in tensorflow - tf.nn.batch_normalization, tf.layers.batch_normalization or tf.contrib.layers.batch_norm.  I chose the implementation from the layers module.

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

One of the hyperparameters in using batch norm is whether to use accumulated statistics across multiple batches (`training=False`) or to process based on statistics from only the current batch (`training=True`).  My intuition is that processing only across the batch removes the difficulty of figuring out how best to remember and forget statistics, and just focus on favouring some actions within the batch more than others.

![fig8]({{ "/assets/debug_dqn/fig8.png"}}) 

**Figure 8 - The Bellman target before and after batch normalization**

After making these changes the first signs of life appeared

![fig9]({{ "/assets/debug_dqn/fig9.png"}}) 

**Figure 9 - It's alive!** 

## tuning

After seeing some initial signs of life, I am pretty happy that there are no errors in the algorithm.  I know from experience that a random agent (i.e. `epsilon=1.0` achieves a mean episode return of 20 - 30.  Seeing these jumps is evidence that learning is possible - but it is not stable.  Learning stability is a key challenge - the early DeepMind work on DQN was focused on learning stability and generalization, not absolute performance.

Getting good performance from a reinforcement learning requires a good set of hyperparameters and comparisons of **runs over multiple different random seeds**.

The hyperparameter tuning process followed a similar hypothesis - problem structure.

## problem - unstable learning

![fig10]({{ "/assets/debug_dqn/fig10.png"}}) 

**Figure 10 - Examples of learning and forgetting.**
- note the different y-axis on the second plot.  The second plot shows the collapse to a policy that is worse than random 
- the third agent actually solves the environment (Open AI consider the env solved for an average reward of 195 over 100 consecutive trails) but then forgets

## hypothesis - learning rate is too high OR state space not being explored well

At this point I had two main thoughts about what might cause an unstable policy
- a large learning rate means that the optimizer is forced to change policy weights, even when the policy is performing well 
- the exploration period being too short means that only late in life does the agent see certain parts of the state space, making the policy unstable in these regions of the state space

I previous had the `epsilon_decay_fraction` hyperparameter set to `0.3` - this meant that the entire epsilon decay is done in the first 30% of the total steps for this experiment.  I changed this to `0.5` - giving the agent more of a chance to see the state space early in life.  This could be investigated further by looking at how the distribution of the observations (either during acting or training) were changing. I decided not to do this. 

The second set of changes were with the learning rate.  Historically with DQN I had used a fixed learning rate of `0.0001`.  This was a chance to play with the decay. 

When I saw Vlad Mnih speak at the 2017 Deep RL bootcamp, he mentioned that larger neural networks can be more stable because they alias less.  By alias I mean less weights are shared.  One option here would be to introduce a larger neural network, but this comes with the risk of overfitting.

Another change I made at this point was to set `centre=False` to the target batch normalization.  John Schulman notes in the talk **The Nuts and Bolts of Deep RL Research** ([video](https://www.youtube.com/watch?v=8EcdaCk9KaQ) and [slides](https://github.com/ADGEfficiency/dsr_rl/blob/master/literature/reinforcement_learning/2016_schulman_nuts-and-bolts.pdf) that removing the mean from the target might affect the agent's will to live.

```python
bellman_norm = tf.layers.batch_normalization(
	tf.reshape(self.bellman, (-1, 1)),
	center=False,
	training=True,
	trainable=False,
)
```
![fig11]({{ "/assets/debug_dqn/fig11.png"}}) 

**Figure 11 - Learning rate decay**

![fig12]({{ "/assets/debug_dqn/fig12.png"}}) 

**Figure 12 - Learning curves across three random seeds**

And here is the setup these final three agents

```python
agent = DQN(
    sess=sess,
    env=energy_py.make_env('CartPole'),
    total_steps=400000,
    discount=0.9,
    memory_type='deque',
    act_path='./act_tb',
    learn_path='./learn_tb',
    learning_rate=0.0001,  
    decay_learning_rate=0.05,
    epsilon_decay_fraction=0.5,
)
```

## concluding thoughts

best practices followed
- using simple env that I'm familiar with
- running comparisons across multiple random seeds
- keeping a detailed log of your thoughts

error fixes
- incorrect masking of `Q(s',a)`
- test code repeating error of incorrect masking

hyperparameters
- batch normalization setup to only scale and not remove the mean
- exploration changed from `0.3` to `0.5`
- learning rate reduced and decayed

Thanks for reading!
