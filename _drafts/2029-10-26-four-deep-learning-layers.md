---
title: 'Four Important Deep Learning Layer Architectures'
date: 2020-10-26
classes: wide2
toc: true
toc_sticky: true
categories:
  - Machine Learning
excerpt:  What they are, when to use them.

---

Checklist
- when it was introduced
- famous examples of work




Deep learning is complex - a few of the things a machine learning engineer must master:

- **layer architectures**, such as convolution or attention
- **activation functions**, including the ReLu or Sigmoid
- **loss functions**, such as cross entropy or the Huber loss
- **backpropagation**, to assign error to weight updates
- **optimizers**, such as SGD or Adam
- **hyperparameters**, including learning rate or batch size

This post is about four of the most common & important layer architectures:

- [the fully connected layer](#fully-connected-layer)
- [2D convolution layer](#2D-convolution-layer)
- [the LSTM layer](#lstm-layer)
- [the attention layer](#attention-layer)

After this article you will:

- how each layer works
- understand when to use each layer
- what the important hyperparameters are for each layer
- how to use the layer in Keras 

All code examples are built using `tensorflow==2.2.0`, using the Keras Functional API.


# Fully Connected Layer

The fully connected layer (also known as a dense or feed-forward layer) is the simplest of our four layers - the node in each layer takes input from each node in the previous layer, and gives output to each node in the next layer.  Each node is fully connected to the nodes before it.

<center>
<img align="center" src="/assets/four-dl-arch/dense.gif">
</center>

<p align="center">The fully connected layer</p>

The strength of the connection between nodes in different layers are controlled by weights - the shape of these weights depending on the number of nodes layers on either side.  Each node has an additional parameter known as a bias, which can be used to shift the output of the node independently of it's input.

A fully connected layer is defined by a number of nodes (also known as units), each with an activation function.  While you could have a layer with different activation functions on different nodes, most of the time each node in a layer has the same activation function.

For hidden layers, the most common choice of activation function is the rectified-linear unit (the ReLu). For the output layer, the correct activation function depends on what the network is predicting:

- regression, target can be positive or negative -> linear (no activation)
- regression, target can be positive only -> ReLu
- classification -> Softmax
- control action, bound between -1 & 1 -> Tanh

For example, a network with a single fully connected layer is used

<center><img align="center" src="/assets/mistakes-data-sci/trpo.png"></center>

<p align="center">A fully connected layer being used to power state of the art reinforcement learning</p>

Fully connected layers will also be found as the last layer on convolutional neural networks performing classification, with one node per class.

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

A fully connected layer is the most general deep learning architecture - it imposes no constraints on connectivity except by depth. Use when your data has no structure - common in reinforcement learning when learning from a flat environment observation.

A neural network built of fully connected layers can be thought of as a blank canvas

An attractive property of fully connected layers is that (given enough width, depth & correct weights) they can approximate any function.  This is known as the Universal Approximation Theorem.

**Just because it can approximate any function, this does not mean we can learn any function**.  Actually finding the correct weights, using the datasets we have available is the challenge.

It's a common lesson in machine learning - a bit of bias is usually better than no bias.

In practice, using layers that are less general purpose is much better.  **This is known as inductive bias** - hardcoding architecture into a network improves learning.  Perhaps the best example of useful inductive bias is the convolutional layer.


# 2D Convolution Layer

The convolutional neural network has powered deep learnings greatest achievements.  It's foundational in computer vision.

Spatial structure

<center><img align="center" src="/assets/four-dl-arch/conv.gif"></center>

<p align="center">A filter producing a filter map by convolving over an image</p>

Layers are connected by filters.  These filters detect certain features, such as lines or edges.  The filters are learnt - they are equilivant to the weights of a fully connected layer.  

To understand how these filters work, let's work with a small image and two filters.

The basic operation in a convolutional neural network is to use these filters to detect patterns in the image, by performing element-wise multiplication and summing the result:

```python
import numpy as np

img = np.eye(3)
"""
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
"""

f1 = np.zeros((3, 3))
f1[1, :] = 1
"""
array([[0., 0., 0.],
       [1., 1., 1.],
       [0., 0., 0.]])
"""
np.sum(img * f1)
# 1.0

f2 = np.eye(3)
"""
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
"""
np.sum(img * f2)
# 3.0
```

We can see that `f1` produces a weaker result than `f2`.  That's because `f2` is a feature detector for a diagonal line - which exactly describes our image.

If we were to work with a different image, our line detector `f1` will produce a stronger response:

```python
img = np.zeros((3, 3))
img[1, :2] = 1
"""
array([[0., 0., 0.],
       [1., 1., 0.],
       [0., 0., 0.]])
"""

np.sum(img * f2)
#  2.0
```

For larger images (which are often `32x32` or larger), this same basic operation is performed, with the filter being passed over the entire image.  The output of this operation acts as feature detection, for the filters that the network has learnt, producing a feature map:


Reusing filters over the entire image allows features to be detected in any part of the image - a property known as translation independence.  This property is ideal for classification - you want to detect a cat no matter where it occurs in the image.

The operation of max-pooling (which reduces the dimensionality of the feature map) increases the strength of translation independence.

hyperparams
- num filter
- filter size
- strides paddingc etc

2D conv - image is 3D, filter is 2D

https://en.wikipedia.org/wiki/Convolutional_neural_network

Conv = sliding a filter along a signal


![]({{ '/assets/ml_energy/conv.png' }})

*Deep convolutional neural network used in the [2015 DeepMind Atari work](https://github.com/ADGEfficiency/dsr_rl/blob/master/literature/reinforcement_learning/2015_Minh_Atari_Nature.pdf)*

The convolutional layer is inspired by our own visual cortex, and is what powers modern computer vision.  They allow machines to see.  They can be used to classify the contents of the image, recognize faces and create captions for images.

World Models post!

Below is an example of a convolutional neural network built using the Keras Functional API.  

flatten, softmax

The attentive reader will note that our images have three dimensions `32, 32, 3`, corresponding to height, width and color channels.  This makes our image a three-dimensional volume

Important hyperparameters for convolutional layers are

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
conv = Conv2D(filters=8, kernel_size=(3, 3), activation='relu')(inp)
flat = Flatten()(conv)
feature_map = Dense(8, activation='relu')(flat)
out = Dense(2, activation='softmax')(flat)
mdl = Model(inputs=inp, outputs=out)

mdl(x)
"""
<tf.Tensor: shape=(4, 2), dtype=float32, numpy=
array([[-0.39803684, -0.08939186],
       [-0.48165476, -0.28876644],
       [-0.32680377, -0.24380796],
       [-0.45394567, -0.28233868]], dtype=float32)>
"""
```

# LSTM Layer

A recurrent layer processes input and generates output as sequences, and powers modern natural language processing.  Recurrent networks allow machines to understand the temporal structure in data, such as words in a sentence.

<center><img align="center" src="/assets/four-dl-arch/recurr.gif"></center>

The third of our deep learning layer architectures is the LSTM, or Long Short-Term Memory layer.  The LSTM is a recurrent neural network, which processes data in a sequence.  A recurrent neural network also generates

A recurrent neural network is different from a normal neural network because it:
- processes data in a sequence
- uses a hidden state $h$ to pass infomation between timesteps

Hidden state = memory

A normal neural network recieves a single input tensor $x$ and generates a single output tensor $y$.  A recurrent neural network differs in the following ways:
- the input tensor $x$ is processed in timesteps
- the network output $y$ is generated at each timestep
- the network recieves a second input $h$, the hidden state of the network at the previous timestep
- the network outputs this hidden state $h$ at each timestep

A recurrent net = inductive bias for sequences

Slow to train

### Entering the timestep dimension

A key concept for understanding recurrent neural networks is getting used to the idea of a timestep dimension.  Imagine we have input data $x$, that is a sequence of integers `[0, 0] -> [2, 20] -> [4, 40]` etc.

If we were using a fully connected layer, we would present this data to the network as a flat array:

```python
import numpy as np

x = np.zeros(10).astype(int)
x[0::2] = np.arange(0, 10, 2)
x[1::2] = np.arange(0, 100, 20)
x = x.reshape(1, -1)

print(x)
# array([[ 0,  0,  2, 20,  4, 40,  6, 60,  8, 80]])

print(x.shape)
# (1, 10)
```

Although the sequence is obvious to us, it's not obvious to a fully connected layer.  The sequential structure would need to be learnt by the network.

We can restructure our data $x$ to explicitly model this sequential structure, by adding a timestep dimension.  The values in our data do not change - only the shape changes:

```python
import numpy as np

x = np.vstack([np.arange(0, 10, 2), np.arange(0, 100, 20)]).T
x = x.reshape(1, 5, 2)

print(x)
"""
array([[[ 0,  0],
        [ 2, 20],
        [ 4, 40],
        [ 6, 60],
        [ 8, 80]]])
"""

print(x.shape)
# (1, 5, 2)
```

Our data $x$ is now structured with three dimensions - `(batch, timesteps, features)`.

### The LSTM

The LSTM is a specific implementation of a recurrent neural network.  The LSTM addresses a challenge that recurrent neural networks struggled with - the ability to think long term.  In a recurrent neural network all infomation passed to the next time step has to fit in a single channel, the hidden state $h$.

The LSTM addresses this by using two hidden states, known as the hidden state $h$ and the cell state $c$.  Having two channels allows the LSTM to remember on both a long and short term.

Internally the LSTM makes use of three gates to control the flow of infomation:
- a forget gate to determine what infomation to delete
- an input gate to determine what to remember
- an output gate to determine what to predict

A good analogy for these gates is that they allow the LSTM to work like a database - matching the `GET`, `POST` & `DELETE` of a REST API, or the read-update-delete operations of a CRUD application.

*For a deeper look at LSTM's, I cannot reccomend the blog post [Understanding LSTM Networks - colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) highly enough.*

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

It's also common to want to access the hidden states of the LSTM - this can be done using the argument `return_state=True`.

seq2seq example


# Attention Layer

Our final deep learning layer architecture is our most recent - attention.  First introduced in 2015, attention has revolutionzied natural language processing.  Attention powers the Transformer - a neural network architecture that forms the backbone of the GPT series of language models.

The first use of attention (known as Bahdanau or additive attention) addressed on the of the limitations of the seq2seq (sequence to sequence) model that was previously the state of the art in machine translation.  As explained in the LSTM section, the basic process in a seq2seq model is:
- encode the input sequence into a fixed length context vector
- decode the context vector into the output sequence

The issue is with all of the infomation from the encoder passing through the fixed length context vector.

An alignment score is used with the previous decoder hidden state $s_{t-1}$ and the new encoder hidden state $h_{t}$.  The alignment is a softmax produced over a score function - the score function being a single layer fully connected neural network:


The score function is learnt.


A second type of attention used in the Transformer is known as dot-product attention.

The dot-product acts like a similarity





Softmax forces tradeoffs

- more weight in one place means less in another

https://datascience.stackexchange.com/questions/45475/variable-input-output-length-for-transformer


Below we demonstrate how to setup an dot-product attention based neural network, doing machine translation.  Note

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Attention, Dense, Embedding, GlobalAveragePooling1D

np.random.seed(42)
tf.random.set_seed(42)

#  data has already been tokenized
#  i.e. 'cat' -> 1, 'dog' -> 2 etc
data = np.random.randint(0, 64, 4 * 5).reshape(4, -1)
tokens = Input(shape=(None,), dtype='int32')

#  here we assume a vocab size of 64 for both languages
#  the output dim of 8 is a hyperparameter - the size of the embedding
embedder = Embedding(input_dim=64, output_dim=8)
embedding = embedder(tokens)
attention = Attention()([embedding, embedding])
pool = GlobalAveragePooling1D()(attention)
out = Dense(3, activation='softmax')(pool)
mdl = Model(inputs=tokens, outputs=out)
mdl(data)
```

The cool thing about this is that we can use a sequence of a different length - our attention layer can handle variable length sequences, and we get rid of the sequence dimension using global average pooling.

```python
data = np.random.randint(0, 64, 4 * 3).reshape(4, -1)
mdl(data)
```


## Summary

Table w
- inductive bias / data structure
- paralleizablitiy
- 
