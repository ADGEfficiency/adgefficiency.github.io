---
title: 'The Four Most Important Deep Learning Layer Architectures'
date: 2020-10-26
classes: wide2
toc: true
toc_sticky: true
categories:
  - Machine Learning
excerpt:  What they are, when to use them.

---


Deep learning is complex - a few of the things a machine learning engineer must master:

- **backpropagation**
- **optimizers** such as SGD or Adam
- **activation functions** such as the relu or sigmoid
- **loss functions** such as cross entropy or the Huber loss
- **hyperparameters** (such as learning rate or batch size)
- **layer architectures** 

This post is about layers.  We will look at the four most important layer architectures:

- fully connected
- convolution
- attention
- recurrent

After this article you will:

- understand when to use each layer
- what the important hyperparameters are for each layer
- how easy they are to train

All code examples were run using `tensorflow==2.2.0`, using the Keras Functional API.


# Fully connected layer

The fully connected layer (also known as a dense layer)

No structure

Use when your data has no structure - common in reinforcement learning when learning from a flat environment observation.

A fully connected layer is defined by:
- the number of nodes
- an activation function


```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten

np.random.seed(42)
tf.random.set_seed(42)

#  dataset of 4 samples, 32x32 with 3 channels
x = np.random.rand(4, 32, 32, 3)

inp = Input(shape=x.shape[1:])
hidden = Dense(8, activation='relu')(inp)
flat = Flatten()(hidden)
out = Dense(2)(flat)
mdl = Model(inputs=inp, outputs=out)
mdl(x)

"""
<tf.Tensor: shape=(4, 2), dtype=float32, numpy=
array([[ 0.23494382, -0.40392348],
       [ 0.10658629, -0.31808627],
       [ 0.42371386, -0.46299127],
       [ 0.34416917, -0.11493915]], dtype=float32)>
"""
```

For hidden layers, the most common choice of activation function is the rectified-linear unit (the ReLu).

For the output layer, the correct activation function depends on what the network is predicting:

- regression, target can be positive or negative -> linear (no activation)
- regression, target can be positive only -> ReLu
- classification -> Softmax
- control action, bound between -1 & 1 -> Tanh


A neural network built of fully connected layers can be thought of as a blank canvas

An attractive property of fully connected layers is that (given enough width, depth & correct weights) they can approximate any function.  This is known as the Universal Approximation Theorem.

**Just because it can approximate any function, this does not mean we can learn any function**.  Actually finding the correct weights, using the datasets we have available is the challenge.

It's a common lesson in machine learning - a bit of bias is usually better than no bias.

In practice, using layers that are less general purpose is much better.  **This is known as inductive bias** - hardcoding architecture into a network improves learning.  Perhaps the best example of useful inductive bias is the convolutional layer.


## Convolution

Conv = sliding a filter along a signal

Spatial structure

![]({{ '/assets/ml_energy/conv.png' }})

*Deep convolutional neural network used in the [2015 DeepMind Atari work](https://github.com/ADGEfficiency/dsr_rl/blob/master/literature/reinforcement_learning/2015_Minh_Atari_Nature.pdf)*

The convolutional layer is inspired by our own visual cortex, and is what powers modern computer vision.  They allow machines to see.  They can be used to classify the contents of the image, recognize faces and create captions for images.

World Models post!

2d conv but data is three D (one dim is convolved over)


```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

np.random.seed(42)
tf.random.set_seed(42)

#  dataset of 4 images, 32x32 with 3 channels
x = np.random.rand(4, 32, 32, 3)

inp = Input(shape=x.shape[1:])
c1 = Conv2D
hidden = Dense(8, activation='relu')(inp)
flat = Flatten()(hidden)
out = Dense(2)(flat)
mdl = Model(inputs=inp, outputs=out)
mdl(x)

"""
<tf.Tensor: shape=(4, 2), dtype=float32, numpy=
array([[ 0.23494382, -0.40392348],
       [ 0.10658629, -0.31808627],
       [ 0.42371386, -0.46299127],
       [ 0.34416917, -0.11493915]], dtype=float32)>
"""

## LSTM

Temporal

![]({{ '/assets/ml_energy/recurr.png' }})

*An unrolled recurrent neural network - [colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)*

A recurrent layer processes input and generates output as sequences, and powers modern natural language processing.  Recurrent networks allow machines to understand the temporal structure in data, such as words in a sentence.

Slow to train


```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM, Flatten

np.random.seed(42)
tf.random.set_seed(42)

#  dataset of 4 samples, 3 timesteps, 32 features
x = np.random.rand(4, 3, 32)

inp = Input(shape=x.shape[1:])
lstm = LSTM(8)(inp)
out = Dense(2)(lstm)
mdl = Model(inputs=inp, outputs=out)
mdl(x)

"""
<tf.Tensor: shape=(4, 2), dtype=float32, numpy=
array([[-0.06428523,  0.3131591 ],
       [-0.04120642,  0.3528567 ],
       [-0.04273851,  0.37192333],
       [ 0.03797218,  0.33612275]], dtype=float32)>
"""
```

You'll notice we only got one output for each of our four samples - where are the other two timesteps?  To get these, we need to use `return_sequences=True`:

```python
tf.random.set_seed(42)
inp = Input(shape=x.shape[1:])
#  tell the LSTM to return the output for each timestep
lstm = LSTM(8, return_sequences=True)(inp)
out = Dense(2)(lstm)
mdl = Model(inputs=inp, outputs=out)
mdl(x)

"""
<tf.Tensor: shape=(4, 3, 2), dtype=float32, numpy=
array([[[-0.08234972,  0.12292314],
        [-0.05217044,  0.19100665],
        [-0.06428523,  0.3131591 ]],

       [[ 0.0381453 ,  0.26402596],
        [ 0.04725918,  0.34620702],
        [-0.04120642,  0.3528567 ]],

       [[-0.21114576,  0.08922277],
        [-0.02972354,  0.24037611],
        [-0.04273851,  0.37192333]],

       [[-0.06888272, -0.01702049],
        [ 0.0117887 ,  0.10608622],
        [ 0.03797218,  0.33612275]]], dtype=float32)>
"""
```

## Attention

Temporal

Softmax forces tradeoffs

- more weight in one place means less in another





## Summary

Table w
- inductive bias / data structure
- paralleizablitiy
- 
