---
title: 'Animated Guide to Four Deep Learning Layers'
date: 2020-10-26
classes: wide2
toc: true
toc_sticky: true
categories:
  - Machine Learning
excerpt:  What they are, when to use them.

---

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

After this article you will understand:

- how each layer works
- what are the important hyperparameters
- when to use each layer
- the intitution behind each layer
- the inductive bias of each layer

If you haven't heard of inductive bias before, you can think of it as hardcoded in structure that helps a network to learn.

All code examples are built using `tensorflow==2.2.0`, using the Keras Functional API.


# Fully Connected Layer

The fully connected layer (also known as a dense or feed-forward layer) is the first of our four layers.  The fully connected layer is the most general and least specalized deep learning layer.  By using a fully connected layer, you are imposing a minimal amount of structure on what the network can learn.

A neural network built of fully connected layers can be thought of as a blank canvas

## How does the fully connected layer work?

At the heart of the fully connected layer is the artificial neuron - the history of which goes all the way back to McCulloch & Pitt's Threshold Logic Unit in 1943.  The artificial neuron is a weighted linear combination of inputs, squeezed through a non-linear activation function.

NEURON PICTURE (in book)

The artificial neuron was inspired by the biological neurons in our brians.  The actual mechanics of an artificial neuron are far remvoed from the complexity of a biological neuron.

A fully connected layer is composed of many neurons (also known as nodes or units). The fully connected gets it's name from how each layer is connected to the layers before & after it.  Each node:

- recieves input from all nodes in the previous layer
- sends output to each node in the next layer

The intitution behind all these connections is one of putting no constraints on how infomation can flow through the network.  By letting all nodes take infomation from each node in the previous layer, and send infomation to each node in the next, the network can (in theory) approximate any function (more on this below).

<center>
<img align="center" src="/assets/four-dl-arch/dense.gif">
</center>

<p align="center">The fully connected layer</p>

The strength of the connection between nodes in different layers are controlled by weights - the shape of these weights depending on the number of nodes layers on either side.  Each node has an additional parameter known as a bias, which can be used to shift the output of the node independently of it's input.

## What hyperparameters are important for a fully connected layer?

The two hyperparameters you'll often set in a fully connected layer are:

- the number of units 
- the activation function

A fully connected layer is defined by a number of nodes (also known as units), each with an activation function.  While you could have a layer with different activation functions on different nodes, most of the time each node in a layer has the same activation function.

ACT FUNCTION PIC

For hidden layers, the most common choice of activation function is the rectified-linear unit (the ReLu). For the output layer, the correct activation function depends on what the network is predicting:

- regression, target can be positive or negative -> linear (no activation)
- regression, target can be positive only -> ReLu
- classification -> Softmax
- control action, bound between -1 & 1 -> Tanh

## Using fully connected layers with the Keras Functional API

Below is an example of how to use a fully connected layer with the Keras functional API.  We are actually using input data that is shaped like an image, to show the flexibility of the fully connected layer. 

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten

#  the least random of all random seeds
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


## When should I use a fully connected layer?

A fully connected layer is the most general deep learning architecture - it imposes no constraints on connectivity except by depth. Use when your data has no structure that you can take advantage of - if your data is a flat array (common in tabular data problems).

Fully connected layers are common in reinforcement learning when learning from a flat environment observation. For example, a network with a single fully connected layer is used in the Trust Region Policy Optimization (TRPO) paper from 2015: 

<center><img align="center" src="/assets/mistakes-data-sci/trpo.png"></center>

<p align="center">A fully connected layer being used to power the reinforcement learning algorithm TRPO</p>

Most neural networks will have fully connected layers somewhere.  It's common to have the penultimate & final layer as fully connected on convolutional neural networks performing classification.  The number of units in the fully connected output layer will be equal to the number of classes, with a softmax activation function used to create a distribution over classes.


## What about Universal Approximation?

An attractive property of fully connected layers is that (given enough width, depth & correct weights) they can approximate any function.  This is known as the Universal Approximation Theorem.

**Just because it can approximate any function, this does not mean we can learn any function**.  Actually finding the correct weights, using the datasets we have available is the challenge.

It's a common lesson in machine learning - a bit of bias is usually better than no bias.

In practice, using layers that are less general purpose is much better.  **This is known as inductive bias - hardcoding architecture into a network improves learning**.  Perhaps the best example of useful inductive bias is the convolutional layer, which is where we head next.


# 2D Convolution Layer

If you had to pick one architecture as the most important of modern deep learning, convolution would have to be it.  AlexNet (a convolutional neural network that won the 2012 ImageNet competition) is seen by many as the start of modern deep learning.

When introduced - https://en.wikipedia.org/wiki/Convolutional_neural_network#History

The convolutional neural network is the workhorse of deep learning - it has been used with text, audio, video and images.  For computer vision, convolution is king.  They can be used to classify the contents of the image, recognize faces and create captions for images.

The 2D convolutional layer is inspired by our own visual cortex.  Work by Hubel & Wiesel in the 1950's showed that neurons in the visual cortexes of mammals respond to small regions on vision.

The intitution behind convolution is looking for small, spatial patterns in a larger space.  The convolution layer has inductive bias for space - such as length, width or depth.


## How does a 2D convoultion layer work?

Convolution is a mathematical operation.  A good mental model for convolution is the process of sliding a filter over a signal, at each point checking to see how well the filter matches the signal.  This checking process is pattern recognition.

For 2D convolution, we are using the following components:
- a 3D image, with shape (height, width, color channels)
- a 2D filter, with shape (height, width)

A convolutional layer is defined by it's filters.  These filters are learnt - they are equilivant to the weights of a fully connected layer.

Filters in the first layers of a convolutional neural network detect simple features such as lines or edges.  Deeper in the network, filters can detect more complex features that help the network perform it's task (such as classification).

To further understand how these filters work, let's work with a small image and two filters.  The basic operation in a convolutional neural network is to use these filters to detect patterns in the image, by performing element-wise multiplication and summing the result:

REPLACE ALL OF THIS WITH A ANIMATION
```python
import numpy as np

img = np.eye(3)
"""
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
"""

#  first filter
f1 = np.zeros((3, 3))
f1[1, :] = 1
"""
array([[0., 0., 0.],
       [1., 1., 1.],
       [0., 0., 0.]])
"""
#  result of using the first filter on our image
np.sum(img * f1)
# 1.0

#  second filter
f2 = np.eye(3)
"""
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
"""
#  result of using the second filter on our image
np.sum(img * f2)
# 3.0
```

We can see that `f1` produces a weaker result than `f2` (`1.0` versus `3.0`).  That's because `f2` is a feature detector for a diagonal line - which exactly describes our image.

If we were to work with a different image, our line detector `f1` will produce a stronger response:

```python
img = np.zeros((3, 3))
img[1, :2] = 1
"""
array([[0., 0., 0.],
       [1., 1., 0.],
       [0., 0., 0.]])
"""

#  result of using the second filter on our second image
np.sum(img * f2)
#  2.0
```

Reusing filters over the entire image allows features to be detected in any part of the image - a property known as translation independence.  This property is ideal for classification - you want to detect a cat no matter where it occurs in the image.

The number of filters in each layer is a hyperparameter - it's roughly the same as the number of nodes in a fully connected layer.

<center><img align="center" src="/assets/four-dl-arch/conv.gif"></center>

<p align="center">A filter producing a filter map by convolving over an image</p>

For larger images (which are often `32x32` or larger), this same basic operation is performed, with the filter being passed over the entire image.  The output of this operation acts as feature detection, for the filters that the network has learnt, producing a 2D feature map.

The feature maps produced by each filter are concatenated, resulting in a 3D volume (the length of the third dimension being the number of filters). The next layer then performs convolution over this new volume, using a new set of learned filters.

TODO IMAGE OF MULITPLE FEATURE MAPS BEING COCATED - EXAMPLE IN BOOK

## 2D convolutional neural network built using the Keras Functional API

Below is an example of how to use a 2D convolution layer with the Keras functional API:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

np.random.seed(42)
tf.random.set_seed(42)

#  dataset of 4 images, 32x32 with 3 color channels
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


## What hyperparameters are important for a convolutional layer?

The important hyperparameters in a convolutional layer are:

- number of filter
- filter size
- activation function
- strides
- padding
- dilation rate

The number of filters determines how many patterns each layer can learn.  It's common to have the number of filters increasing with the depth of the network.

Filter size is commonly set to `(3, 3)`, with a ReLu as the activation function.

Strides can be used to skip steps in the convolution, resulting is smaller feature maps.  Padding can be used to allow pixels on the edge of the image to act as if they are in the middle of an image.  Dilations allow the filters to operate over a larger area of the image, while still producing feature maps of the same size.


## When should I use a convolutional layer?

A cov net inductive bias for space - any data that has a spatial structure.

Convolution works when your data has a spatial structure - for example, images have spatial structure in height & width.  You can also get this structure using techniques such as Fourier Transforms, and perform convolution in the frequency domain.

If you are working with images, convolution is king.

An example

![]({{ '/assets/ml_energy/conv.png' }})

<p align="center">Deep convolutional neural network used in the [2015 DeepMind Atari work](https://github.com/ADGEfficiency/dsr_rl/blob/master/literature/reinforcement_learning/2015_Minh_Atari_Nature.pdf)</p>

So what other kinds of structure can data have, other than spatial?  Many types of data also have a sequential structure - motivating our next two layer architectures.


# LSTM Layer

The third of our deep learning layer architectures is the LSTM.  The LSTM (Long Short-Term Memory) is a recurrent architecture - it processes input and generates output as sequences.

Recurrent networks allow machines to understand the temporal structure in data, such as words in a sentence.  A recurrent neural network has a memory - it is able to pass infomation forward to the next time step, in the form of a hidden state.

A normal neural network recieves a single input tensor $x$ and generates a single output tensor $y$. A recurrent neural network differs from a non-recurrent neural network in two ways:

1. data (both input & output) is processed as a sequence of timestups
2. a learnt hidden state $h$ is used to pass infomation between timesteps

<center><img align="center" src="/assets/four-dl-arch/recurr.gif"></center>

<p align="center">A recurrent neural network</p>

A key concept for understanding recurrent neural networks is getting used to the idea of a timestep dimension.

Understanding how neural networks can process data in a sequence requires one key insight - how to shape data with a timestep dimension.


### Entering the timestep dimension

Imagine we have input data $x$, that is a sequence of integers `[0, 0] -> [2, 20] -> [4, 40]`.  If we were using a fully connected layer, we would present this data to the network as a flat array:

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

Although the sequence is obvious to us, it's not obvious to a fully connected layer.  **All a fully connected layer would see is a list of numbers - the sequential structure would need to be learnt by the network**.

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

Our data $x$ is now structured with three dimensions - `(batch, timesteps, features)`.  A recurrent neural network processes the features one timestep at a time.

Now that we understand how to structure data to be used with a recurrent neural network, we can look at details of how the LSTM layer works.


## How does an LSTM layer work?

*For a deeper look at LSTM's, I cannot reccomend the blog post [Understanding LSTM Networks - colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) highly enough.*

The LSTM was first introduced in 1997, and has formed the backbone of many important sequence based deep learning models, such as machine translation.

The LSTM is a specific implementation of a recurrent neural network.  The LSTM addresses a challenge that recurrent neural networks struggled with - the ability to think long term.  In a recurrent neural network all infomation passed to the next time step has to fit in a single channel, the hidden state $h$.

The LSTM addresses this by using two hidden states, known as the hidden state $h$ and the cell state $c$.  Having two channels allows the LSTM to remember on both a long and short term.

Internally the LSTM makes use of three gates to control the flow of infomation:
- a forget gate to determine what infomation to delete
- an input gate to determine what to remember
- an output gate to determine what to predict

One important architecture is known as seq2seq, used for machine translation, which we discuss below.

- encode the input sequence into a fixed length context vector
- decode the context vector into the output sequence

<center><img align="center" src="/assets/four-dl-arch/seq2seq.gif"></center>

## What is the intitution behind an LSTM?

A good mental model for an LSTM is a database.  **The three gates allow the LSTM to work like a database** - matching the `GET`, `POST` & `DELETE` of a REST API, or the read-update-delete operations of a CRUD application.

The forget gate acts like a `DELETE`, allowing the LSTM to remove infomation that isn't useful.  The input gate acts like a `POST`, where the LSTM can choose infomation to remember.  The output gate acts like a `GET`, where the LSTM chooses what to send back to a user request for infomation.


## What is the inductive bias for the LSTM?

The LSTM has two forms of inductive bias - one for processing data as a sequence, and the other for storing a memory.


## Using LSTM layers with the Keras Functional API

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

It's also common to want to access the hidden states of the LSTM - this can be done using the argument `return_state=True`.  We now get back three tensors - the output of the network, the LSTM hidden state and the LSTM cell state.  Note the shape of the hidden states is equal to the number of units in the LSTM:

```python
tf.random.set_seed(42)
inp = Input(shape=x.shape[1:])
lstm, hstate, cstate = LSTM(8, return_sequences=False, return_state=True)(inp)
out = Dense(2)(lstm)
mdl = Model(inputs=inp, outputs=[out, hstate, cstate])
out, hstate, cstate = mdl(x)

print(hstate.shape)
# (4, 8)

print(cstate.shape)
# (4, 8)
```

If you wanted to access the hidden states at each timestep, then you can use `return_sequences=True` and `return_state=True`.


## What hyperparameters are important for an LSTM layer?

For an LSTM layer, the main hyperparameter is the number of units.  The number of units will determine the size of the hidden state.

While not a hyperparameter, it can be useful to include gradient clipping when working with LSTMs.


## When should I use an LSTM layer?

When working with sequence data, an LSTM (or it's close cousin the GRU) is a common choice.

One major downside of the LSTM is that they are slow to train.  This is because processing the sequence can't be easily parallezized.

One useful feature of the LSTM is the learnt hidden state.  This can be used by other models as a compressed representation - such as in the World Models paper.


# Attention Layer

Like the LSTM, our final deep learning layer is also designed to process sequences.  It's also our youngest.

First used with LSTM, transfromer = no recurrence

**First introduced in 2015, attention has revolutionzied natural language processing**.  Attention powers the Transformer - a neural network architecture that forms the backbone of Open AI's GPT series of language models.






## What is the intitution behind attention?

Intitution = choosing what part of sequence to take infomation from

The intuition behind attention is simple - **some parts of a sequence are more important that others**. Take the example of machine translation, to translate the German sentence `Ich bin eine Maschine` into the English `I am a machine`.

When predicitng the last word in the translation `machine`, all of our attention should be placed on the last word of the source sentence `Maschine`.  There is no point looking at earlier words in the source sequence.

If we take a more complex example of translating the German `Ich habe ein bisschen Deutsch gelernt` into the Engilsh `I have learnt a little German`.

When predicting the third token of our Engilsh sentence (`learnt`), attention should be placed on the last token of the German sentence (`gelernt`).

TODO drawing here


## What is the inductive bias of attention?

Attention is an indutive bias for prioritizing infomation flow.  A fully connected layer can allow infomation to flow between all nodes in subesquent layers, and could in theory learn a similar pattern that an attention layer does.

The use of a softmax in attention layers force the layer to prioritize.  The softmax forces the network to make tradeoffs about infomation flow - more weight in one place means less in another.  There is no such restriction in a fully connected layer, where increasing one weight need not affect another.


## How does an attention layer work?

The attention layer recieves three inputs - a query, keys and values.

The attention layer can be thought of as three mechanisms in sequence:

- alignment (or similarity) of a query and keys
- a softmax to convert the alignment into a probaility distrubtion
- selecting keys based on


### Query, key, value

In the same way that understanding the time-step dimension is a key step in understanding recurrent neural networks, understanding what the query, key & value mean is foundational in attention.

<center><img align="center" src="/assets/four-dl-arch/attention.gif"></center>

A good analogy is with the Python dictionary.  Let's start with a simple example:

```python
query = 'dog'

#  keys = 'cat', 'dog', values = 1, 2
database = {'cat': 1, 'dog': 2}

database[query]
#  2
```

Exact match

How does this relate to attention?  In attention, our query, keys and values are all vectors:

```python
query = [0, 0.9]
#  keys = [0, 0], [0, 1] values = [0], [1]
database = {[0, 0]: [0], [0, 1]: [1]}
```

Now we don't have an exact match for our query - instead of using an exact match, we instead can calculate a similarity (i.e. alignment) between our query and keys, and return the closest value:

```python
database.similarity(query)
#  [1]
```


A finally technicality - often the keys and values are the same.  This simply means that you compare your query with the keys (as normal), and then use those keys after computing the alignment.

To summarize:

- query = what we are looking for
- key = what we compare the query with
- value = what we place attention over

## Attention mechanisms

We will briefly look at tow - Additive Attention and Dot-Product Attention.


### Additive Attention

The first use of attention (known as Bahdanau or Additive Attention) addressed on the of the limitations of the seq2seq (sequence to sequence) model that was previously the state of the art in machine translation.  

As explained in the LSTM section, the basic process in a seq2seq model is to encode the source sentence into a fixed length context vector.  The issue is with all of the infomation from the encoder passing through the fixed length context vector.

In Bahdanau et. al 2015, Additive Attention is used to learn an alignment between all the encoder hidden states and the decoder hidden states.  As the sequence is processed, the output of this alignment is used in the decoder to predict the next token.


## Dot-Product Attention

A second type of attention is Dot-Product Attention. 

used in the Transformer is known as 

Instead of using addition, the dot-product attention layer uses matrix multilpication to measure similarity between the query and the keys.

The dot-product acts like a similarity - comparison with cosine:

Implementation of  a transfromer head TODO


## Using Dot-Product attention layer with the Keras Functional API

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

## What hyperparameters are important in an Attention layer?

Type of attention mechanism

Scaling

Size


## When should I use an attention layer?

Sequence based with fast training

Use the alignment scores for interpretability


## Summary

Table w
- inductive bias / data structure
- paralleizablitiy
- 
