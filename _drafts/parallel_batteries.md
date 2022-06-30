---
title: 'Fast & Efficient Parallel Rollouts of Battery Storage Simulation - energypy experiments - the finale'
date: 2022-06-12
categories:
  - Machine Learning
  - Energy
classes: wide
excerpt: An introduction to how the UK recovers electricity grid balancing costs.

---

In my spare time I work on [energy-py](https://github.com/ADGEfficiency/energy-py).

I'm currently (June 2022) in the process of rewriting it in PyTorch.  I had a [bad experience upgrading from Tensorflow 2.5 to 2.6+](https://github.com/ADGEfficiency/energy-py/pull/60/commits/689f0e4a7d0612bc0ea35f32e173fd700490fc87).


## Parallel Batteries in Numpy

One feature I recently added was parallel rollouts of battery storage simulation in numpy.

The implementation is in [energypy.envs.battery.Battery]() - as it's likely incredibly difficult to read my code (it is for me), I will outline the key ideas here.

Let's start with some assumptions:

- battery 

We are working with a battery model

## Single battery

```python
power = 2
capacity = 4

action = 0.5
```

## Further work

Parallelizing on GPU with TF/Pytorch


## Pytorch Conversion

---


Dependencies

    energypy
    energypy-linear,
    nem-data.

Running the experiment

All done through the top level Makefile.



## Methods

    perfect foresight
    lag prices only
    how useful is pretraining?
    how useful is attention?
    more time series features




