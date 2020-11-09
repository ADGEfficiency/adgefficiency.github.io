---
title: 'An Animated Guide to Deep LearningLayers '
date: 2020-10-26
classes: wide2
toc: true
toc_sticky: true
categories:
  - Machine Learning
excerpt:  What they are, when to use them.

---

**Deep learning is complex**. Some of the many things a machine learning engineer must master include:

- **layer architectures**, such as convolution or attention
- **activation functions**, including the ReLu or Sigmoid
- **loss functions**, such as cross entropy or the Huber loss
- **backpropagation**, to assign error to weight updates
- **optimizers**, such as SGD or Adam
- **hyperparameters**, including learning rate or batch size

**This post is about layer architectures** - together we will look at four of the most common & important deep learning layers that are commonly used for deep learning on images, text and audio:

- the [fully connected layer](#fully-connected-layer)
- the [2D convolutional layer](#2D-convolution-layer)
- the [LSTM layer](#lstm-layer)
- the [attention layer](#attention-layer)

After this article you will understand:

- **how each layer works**
- the **intuition** & **inductive bias** of each layer
- what the **important hyperparameters** of each layer are
- **when to use** each layer

All code examples are built using `tensorflow==2.2.0` using the Keras Functional API.


# Fully Connected Layer

**The fully connected layer** (also known as a dense, feed-forward or perceptron layer) **is the most general deep learning layer**.

The fully connected layer imposes the least amount of structure of any layer architecture.  It is the least specialized of our four layers, and will be found in almost all neural networks.


## How does the fully connected layer work?

At the heart of the fully connected layer is the artificial neuron - the history of which goes all the way back to 1943 with McCulloch & Pitt's *Threshold Logic Unit*.  

The artificial neuron was inspired by the biological neurons in our brains.  The actual mechanics of an artificial neuron are far removed from the complexity of a biological neuron.  Yet even with this simplification, artificial neurons are capable of learning wonderful things.

The artificial neuron composed of three steps:

1. a weighted linear combination of inputs
2. a sum across all weighted inputs
3. an activation function

<center><img align="center" src="/assets/four-dl-arch/neuron.gif"></center>

<p align="center"><i>A single neuron with a ReLu activation function</i></p>

The strength of the connection between nodes in different layers are controlled by weights - the shape of these weights depending on the number of nodes layers on either side.  Each node has an additional parameter known as a bias, which can be used to shift the output of the node independently of it's input.

After applying the weight and bias, all of the inputs into the neuron are summed together to a single number.  This is then passed through an activation function. The most important activation functions are:

- a linear activation function -> unchanged output
- a ReLu -> 0 if the input is negative, otherwise input is unchanged
- a Sigmoid squashes the input to the range 0, 1
- a Tanh squashes the input to the range 0, 1

The output of the activation function is then sent as input to all neurons in the next layer.  This is where the fully connceted layer gets it's name from - each layer is fully connected to the layers before & after it.

A fully connected layer is composed of many of these neurons (also known as nodes or units), with each node:

- receiving input from all nodes in the previous layer
- sending output to each node in the next layer

For the first layer, the node gets it's input from the data being fed into the network (each data point is connected to each node).  For last layer, the output is the prediction of the network.

<center><img align="center" src="/assets/four-dl-arch/dense.gif"></center>

<p align="center"><i>The fully connected layer</i></p>


## What is the intuition & inductive bias of a fully connected layer?

**The intuition behind all the connections in a fully connected layer is to put no constraints on how information can flow through the network**.

The fully connected layer imposes no structure and makes no assumption about the data or task the network will perform.  **A neural network built of fully connected layers can be thought of as a blank canvas**.  The intuition behind a fully connected layer is to impose no structure and let the network figure everything out.

**This lack of structure is what gives neural networks (of sufficient depth & width) the ability to approximate any function**. This is known as the Universal Approximation Theorem.

The ability to approximate any function at first sounds attractive.  Why do we need any other architecture if a fully connected layer can learn anything?

**The key insight is that being able to learn in theory does not mean we can learn in practice**.  Actually finding the correct weights, using the data we have available is impractical.

This motivates the use of the more specialized layer architectures we will look at later.  By hard coding in assumptions about the structure of the data & task, we can learn functions in practice that we couldn't other wise.  **This is known as inductive bias - making assumptions that make a network less general but more useful**.

It's a common lesson in machine learning - a bit of bias is usually better than no bias (especially if you trade it for variance).


## When should I use a fully connected layer?

A fully connected layer is the most general deep learning architecture - it imposes no constraints on connectivity except by depth. Use when your data has no structure that you can take advantage of - if your data is a flat array (common in tabular data problems).

Fully connected layers are common in reinforcement learning when learning from a flat environment observation. For example, a network with a single fully connected layer is used in the Trust Region Policy Optimization (TRPO) paper from 2015: 

<center><img align="center" width="50%" src="/assets/mistakes-data-sci/trpo.png"></center>

<p align="center"><i>A fully connected layer being used to power the reinforcement learning algorithm TRPO</i></p>

Most neural networks will have fully connected layers somewhere.  It's common to have the penultimate & final layer as fully connected on convolutional neural networks performing classification.  The number of units in the fully connected output layer will be equal to the number of classes, with a softmax activation function used to create a distribution over classes.

## What hyperparameters are important for a fully connected layer?

The two hyperparameters you'll often set in a fully connected layer are:

- the number of units 
- the activation function

A fully connected layer is defined by a number of nodes (also known as units), each with an activation function.  While you could have a layer with different activation functions on different nodes, most of the time each node in a layer has the same activation function.
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



# 2D Convolution Layer

**If you had to pick one architecture as the most important in deep learning, it's hard to look past convolution**.  

AlexNet (a convolutional neural network that won the 2012 ImageNet competition) is seen by many as the start of modern deep learning.

Another landmark use of convolution was Le-Net-5 in 1998, a 7 layer convolutional neural network used by Yann LeCun to classify handwritten digits.  This work eventually resulted in the MNIST dataset.

The convolutional neural network is the workhorse of deep learning - it can be used with text, audio, video and images.  **For computer vision, convolution is king**.  Convolutional neural networks can be used to classify the contents of the image, recognize faces and create captions for images.


## What is the intuition behind convolution?

The 2D convolutional layer is inspired by our own visual cortex.  Work by Hubel & Wiesel in the 1950's showed that individual neurons in the visual cortexes of mammals are activated by small regions of vision.  

The history of using convolution in artificial neural networks goes back decades to the neocognitron, an architecture introduced by Kunihiko Fukushima in 1980, inspired by the work done of Hubel & Wiesel.

Convolution itself is a mathematical operation, commonly used in signal processing.  **A good mental model for convolution is the process of sliding a filter over a signal, at each point checking to see how well the filter matches the signal**.  

This checking process is pattern recognition, and is the intitution behind convolution - looking for small, spatial patterns anywhere in a larger space.


## What is the inductive bias of a 2D convolutional layer?

The convolution layer has inductive bias for space - such as length, width or depth.


## How does a 2D convoultion layer work?

Above we defind the intution of convolution being looking for patterns in a larger space.

In a 2D convolutional layer, the patterns are filters, and the larger space is an image.  

For 2D convolution, we are using the following components:
- a 3D image, with shape (height, width, color channels)
- a 2D filter, with shape (height, width)

A convolutional layer is defined by it's filters.  These filters are learnt - they are equilivant to the weights of a fully connected layer.

Filters in the first layers of a convolutional neural network detect simple features such as lines or edges.  Deeper in the network, filters can detect more complex features that help the network perform it's task (such as classification).

To further understand how these filters work, let's work with a small image and two filters.  The basic operation in a convolutional neural network is to use these filters to detect patterns in the image, by performing element-wise multiplication and summing the result:

<center><img align="center" width="50%" src="/assets/four-dl-arch/filters.gif"></center>

<p align="center"><i>Applying different filters to a small image</i></p>

Reusing filters over the entire image allows features to be detected in any part of the image - a property known as translation invariance.  This property is ideal for classification - you want to detect a cat no matter where it occurs in the image.

Another important feature of convolution in many applications is translation invariance - that a feature can be detected in any position of the image.

The number of filters in each layer is a hyperparameter - it's roughly the same as the number of nodes in a fully connected layer.

<center><img align="center" src="/assets/four-dl-arch/conv.gif"></center>

<p align="center">A filter producing a filter map by convolving over an image</p>

For larger images (which are often `32x32` or larger), this same basic operation is performed, with the filter being passed over the entire image.  The output of this operation acts as feature detection, for the filters that the network has learnt, producing a 2D feature map.

The feature maps produced by each filter are concatenated, resulting in a 3D volume (the length of the third dimension being the number of filters). The next layer then performs convolution over this new volume, using a new set of learned filters.

<center><img align="center" width="50%" src="/assets/four-dl-arch/map.gif"></center>

<p align="center"><i>Applying different filters to a small image</i></p>


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

**Since it's introduction in 2015, attention has revolutionzied natural language processing**.

First used in combination with the LSTM based seq2seq model, attention is also to power the Transformer - a neural network architecture that forms the backbone of Open AI's GPT series of language models.

The Transformer is a sequence model without recurrence (it doesn't use an LSTM), allowing it to be efficiently trained (avoiding backpropagation through time).


## What is the intitution behind attention?

The intuition behind attention is simple - **some parts of a sequence are more important that others**. Take the example of machine translation, to translate the German sentence `Ich bin eine Maschine` into the English `I am a machine`.

When predicitng the last word in the translation `machine`, all of our attention should be placed on the last word of the source sentence `Maschine`.  There is no point looking at earlier words in the source sequence.

If we take a more complex example of translating the German `Ich habe ein bisschen Deutsch gelernt` into the Engilsh `I have learnt a little German`.

When predicting the third token of our Engilsh sentence (`learnt`), attention should be placed on the last token of the German sentence (`gelernt`).

TODO drawing here

Intitution = choosing what part of sequence to take infomation from

Alignment / similarity


## What is the inductive bias of attention?

**Attention is indutive bias for prioritizing infomation flow**.  A fully connected layer can allow infomation to flow between all nodes in subesquent layers, and could in theory learn a similar pattern that an attention layer does.  Remember that in theory doesn't not mean in practice.

The use of a softmax in an attention layer forces the layer to prioritize.  **The softmax forces the network to make tradeoffs about infomation flow - more weight in one place means less in another**.  There is no such restriction in a fully connected layer, where increasing one weight does not affect another.


## How does an attention layer work?

The attention layer can be thought of as three mechanisms in sequence:

- alignment (or similarity) of a query and keys
- a softmax to convert the alignment into a probaility distrubtion
- selecting keys based on the alignment

Different attention layers (such as Additive Attention or Dot-Product Attention) use different mechanisms in the alignment step.  The softmax & key selection steps are common to all attention layers.

<center><img align="center" src="/assets/four-dl-arch/attention.gif"></center>

The attention layer receives three inputs - a query, keys and values:

- query = what we are looking for
- key = what we compare the query with
- value = what we place attention over

Note that often the keys are set equal to the values.  This simply means that the quantity we are doing the similarity comparison with is also the quantity we will place attention over.


### Query, key, value

In the same way that understanding the time-step dimension is a key step in understanding recurrent neural networks, understanding what the query, key & value mean is foundational in attention.

A good analogy that uses the same terminology is with the Python dictionary.  Let's start with a simple example:

```python
query = 'dog'
#  keys = 'cat', 'dog', values = 1, 2
database = {'cat': 1, 'dog': 2}
database[query]
#  2
```

In the above example, we find an exact match for our query `'dog'`.

In the neural network attention, we are not working with strings - we are working with vectors.  Our query, keys and values are all vectors:

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


## Attention mechanisms

Above we pointed out that an attention layer involves three steps:
1. alignment based on similarity
2. softmax to create attention weights
3. choosing values based on attention

The second & third steps are common to all attention layers - the differences all occur in the first step - how the alignment is done.

We will briefly look at t - Additive Attention and Dot-Product Attention.


### Additive Attention

This first use of attention (known as Bahdanau or Additive Attention) addressed on the of the limitations of the seq2seq (sequence to sequence) model that was previously the state of the art in machine translation.  

As explained in the LSTM section, the basic process in a seq2seq model is to encode the source sentence into a fixed length context vector.  The issue is with all of the infomation from the encoder passing through the fixed length context vector.

In Bahdanau et. al 2015, Additive Attention is used to learn an alignment between all the encoder hidden states and the decoder hidden states.  As the sequence is processed, the output of this alignment is used in the decoder to predict the next token.


## Dot-Product Attention

A second type of attention is Dot-Product Attention. This is the alignment mechanism used in the Transformer.

Instead of using addition, the dot-product attention layer uses matrix multilpication to measure similarity between the query and the keys.

The dot-product acts like a similarity - comparison with cosine:


## Implementing a Single Attention Head with the Keras Functional API

<center><img align="center" width="40%" src="/assets/four-dl-arch/head.png"></center>

<p align="center">The multi-head attention layer used in the Transformer</p>

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

qry = np.random.rand(4, 16, 32).reshape(4, -1, 32).astype('float32')
key = np.random.rand(4, 32).reshape(4, 1, 32).astype('float32')
values = np.random.rand(4, 32).reshape(4, 1, 32).astype('float32')

q_in = Input(shape=(None, 32))
k_in = Input(shape=(1, 32))
v_in = Input(shape=(1, 32))

capacity = 4
q = Dense(4, activation='linear')(q_in)
k = Dense(4, activation='linear')(k_in)
v = Dense(4, activation='linear')(v_in)

score = tf.matmul(q, k, transpose_b=True)
attention = tf.nn.softmax(score, axis=-1)
output = tf.matmul(attention, v)

mdl = Model(inputs=[q_in, k_in, v_in], outputs=[score, attention, output])
sc, attn, out = mdl([qry, key, values])
print(f'query shape {qry.shape}')
print(f'score shape {sc.shape}')
print(f'attention shape {attn.shape}')
print(f'output shape {out.shape}')
"""
query shape (4, 16, 32)
score shape (4, 16, 1)
attention shape (4, 16, 1)
output shape (4, 16, 4)
"""
```

This architecture also works with a different length query:

```python
qry = np.random.rand(4, 8, 32).reshape(4, -1, 32).astype('float32')
sc, attn, out = mdl([qry, key, values])
print(f'query shape {qry.shape}')
print(f'score shape {sc.shape}')
print(f'attention shape {attn.shape}')
print(f'output shape {out.shape}')
"""
query shape (4, 8, 32)
score shape (4, 8, 1)
attention shape (4, 8, 1)
output shape (4, 8, 4)
"""
```


## What hyperparameters are important in an Attention layer?

When using attention heads as shown above, hyperparameters to consider are:

- size of the linear layer used to transform the query, values & keys
- the type of attention mechanism
- how to scale the alignment before the softmax


## When should I use an attention layer?

Sequence based with fast training

Use the alignment scores for interpretability


[Transformer model for language understanding](https://www.tensorflow.org/tutorials/text/transformer)

[Attention? Attention! - Lilian Wang](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)


## Summary

Table w
- inductive bias / data structure
- paralleizablitiy
- 
