





This article is the second in a series on debugging a new implementation of DQN I built in energy_py.

## problem - stability

The major problem that remained at the end of my first debugging session was instability.  The agent was able to solve the CartPole-v0 environment (Open AI consider this environment solved when an average over the last 100 episodes of 195 is reached).

The issue is with the stability of the policy - the agent often quickly learns to forget it's good work and collapses to the worst policy possible - a single action selected for any observation.

I had a number of hypotheses at this stage about what might be causing the instability
- overestimation bias
- batch size 
- neural network size
- target network update 

## hypothesis - overestimation bias and batch size

DQN is an agressive algorithm - both in terms of acting (where the argmax will always select the 'best' action, even if the Q value is only slightly higher than another action) and in terms of learning (where the target is formed using a maximum over all possible futures).

Double Q-Learning (and it's deep learning variant DDQN) aims to overcome this overestimation bias by TODO.  In terms of code DDQN only requires a few small changes from DQN
- sharing weights between TODO
- changing the target creation TODO

The second hypothesis relates to batch size.  In DQN the online network is changed at each step by minimizing the difference between the Bellman target and the current network approximation of `Q(s,a)`.  The hypothesis was that if the batch is too small (previously I was using `batch_size=32`) then the distribution of the data per batch won't be smooth.

Because of the nature of the CartPole environment, this becomes more of an issue as the agent learns, because the episode length increases.  As the episode length increases the relative amount of terminal states will decrease.  I increased the batch size to `512` to see what effect this would have concurrent with the DDQN change.

I setup three runs over three different random seeds at this [commit in energy_py](https://github.com/ADGEfficiency/energy_py/tree/8facd61485dced0d76c756eff714eb4053583915).  These experiments were setup using `.ini.` config files

```ini
[DEFAULT]
agent_id=dqn
total_steps=400000
discount=0.99
tau=0.001
batch_size=32
layers=25,25,25
learning_rate=0.0005
learning_rate_decay=0.1
epsilon_decay_fraction=0.5
memory_fraction=0.20
memory_type=deque
double_q=False

[DDQN1_512]
double_q=True
batch_size=512
seed=42

[DDQN2_512]
double_q=True
batch_size=512
seed=15

[DDQN3_512]
double_q=True
batch_size=512
seed=2
```

The results of the three runs are shown below.  I concluded that both of these changes helped, but that the instability problem has clearly not been solved.

## hypothesis - neural network size

For these simple reinforcement learning problems the idea is that smaller neural networks should be preferred, as they have less change to overfit.

One reason why a small neural network could cause instability is a lot of aliasing i.e. sharing of weights.  This sharing of weights means that changes to one part of the network end up disrupting the understanding encoded in another part of the network.

Up until now I have been using a neural network with three layers, 25 nodes per layer.  My hypothesis was that even though this is not a lot of nodes, because the network has three layers there will be a lot of aliasing (i.e. changes in the input layer affect all the layers below it).  

I ran the same set of experiments as above with a shallower and wider structure of `layers=(512)`.

```
sess,<tensorflow.python.client.session.Session object at 0x120169f98>
learning_rate_decay,0.1
epsilon_decay_fraction,0.5
memory_type,deque
total_steps,400000
discount,0.99
memory_fraction,0.20
env,<TimeLimit<CartPoleEnv<CartPole-v0>>>
batch_size,512
env_repr,<TimeLimit<CartPoleEnv<CartPole-v0>>>
agent_id,dqn
act_path,/Users/adam/git/energy_py/energy_py/experiments/results/DDQN_debug/tensorboard/wide_net_1/act
tau,0.001
learning_rate,0.0005
double_q,True
learn_path,/Users/adam/git/energy_py/energy_py/experiments/results/DDQN_debug/tensorboard/wide_net_1/learn
seed,42
layers,256
```

## hypothesis - the target network update

The target network update is an obvious place where instability could arise. There are two common strageties for updating the target network

In the 2013 and 2015 DeepMind Atari work they periodically update the target network weights every `C` steps - in the 2015 paper `C=TODO`. 

Another stragety is to smoothly copy the online network weights to the target network at each step using a parameter `tau=0.001`.  This is the stragety that I've been using so far.
