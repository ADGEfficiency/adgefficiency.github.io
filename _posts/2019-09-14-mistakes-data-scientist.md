---
title: 'Mistakes data scientists make'
date: 2019-09-14
categories:
  - Data Science, Python, Machine Learning
excerpt: Badges of honour for the accomplished data scientist.

---

> An expert is a person who has made all the mistakes that can be made in a very narrow field - Niels Bohr - Nobel Prize in Physics 1922

> If you don't make mistakes, you’re not working on hard enough problems. And that’s a big mistake – Frank Wilczek - Nobel Prize in Physics 2004

The antifragile system uses error for improvement.  **Using mistakes for progress is a fascinating paradox**.  Examples of antifragile systems include business, evolutionary learning, biological evolution, training neural networks and also learning data science.

I've made many mistakes while learning, working and teaching data science. I see those same mistakes made by my students at [Data Science Retreat](https://datascienceretreat.com).

Patterns exist in the mistakes made learning data science - hopefully sharing them will help you to only make them once.  But do make them - the mistake is useful for progress.

## Not plotting the target

Prediction is a skill that separates the data scientist from the data analyst.  The data analyst analyzes the past - the data scientist **predicts the future**.

Data scientists use supervised machine learning as a tool to predict a target from features.  

This target can be either a number (regression) or a type (classification), and is a prediction that can be used to optimize business decisions.

Understanding the **distribution of your target** is a key step in data exploration, and will inform many decisions the data scientist makes in the future about what techniques to use.

For regression, a histogram will show if the target distribution is multimodal and highlight outliers:

```python
import numpy as np
import pandas as pd

data = np.concatenate([
    np.random.normal(5, 1, 10000),
    np.random.normal(-5, 1, 10000),
    np.array([-20, 20] * 1000)
])

pd.DataFrame(data).plot(kind='hist', legend=None, bins=100)
```

The histogram shows the four distributions generated this dataset - two normal and two uniform.

<center><img src="/assets/mistakes-data-sci/reg.png"></center>

In classification you want to know how balanced your classes are - we can see this using a bar chart:

```python
data = ['awake'] * 1000 + ['asleep'] * 500 + ['dreaming'] * 50

pd.Series(data).value_counts().plot(kind='bar')
```

<center><img src="/assets/mistakes-data-sci/class.png"></center>

Both of these plots are simple, yet many new data scientists are not in the habit of plotting them.

## Not thinking in terms of dimensionality

Data scientists learn to see the world through the lens of **dimensionality**.  Dimensionality is structure - this structure provides hope that the world can be understood.

**Dimensionality reduction** is a key skill of the data scientist.  Machine learning is excellent at it - dimensionality reduction is so often done with machine learning that it is a useful working definition for the entire field.

Examples of dimensionality reduction include:
- pixels in an image -> cat or dog (binary classification)
- pixels in an image -> image caption text
- customer data -> lifetime value estimation (regression)

**Business decisions are made in low dimensional spaces** - this is the value of dimensionality reduction.

A list of 50 numbers about a customer becomes an estimate of lifetime value, which can be used to take an action.  Prediction becomes control.

Reducing dimensionality is desirable, and increasing dimensionality is undesirable.  The difficulty of working in high dimensional spaces is the curse of dimensionality.

### The curse of dimensionality

To understand the curse of dimensionality we need to reason about the space (aka volume) that data occupies.

We can imagine a dense dataset - a large number of diverse samples within a small volume.  We can also imagine a sparse dataset - a small number of samples in a large volume.

What happens to the density of a dataset as we add dimensions?  It becomes less dense, because the data is now more spread out.  The problem is that adding a dimension makes understanding the size of the space exponentially larger.  

Why?  Because this new dimension needs to be understood in relation to every other combination of all the existing dimensions.

The **curse of dimensionality** is this exponential increase in the size of the space that occurs as we add dimensions:

```python
import itertools

def calc_num_combinations(data):
    return len(list(itertools.permutations(data, len(data))))

def test_calc_num_combinations():
    test_data = (
        ((0, ), 1), ((0, 1), 2), ((0, 1, 2), 6)
    )
    for data, length in test_data:
        assert length == calc_num_combinations(data)

test_calc_num_combinations()

curse = []
for l in range(11):
    curse.append((l, calc_num_combinations(range(l))))

print(curse)

[(0, 1),
 (1, 1),
 (2, 2),
 (3, 6),
 (4, 24),
 (5, 120),
 (6, 720),
 (7, 5040),
 (8, 40320),
 (9, 362880),
 (10, 3628800)]
```

To be useful, a model needs to understand the space - the larger the size of the space the longer it can take to learn it.  This is why adding features with no information (uncorrelated noise or perfect correlation) is extra painful.  The model needs to understand the relationship of this new column with all other combinations of all other columns.

Getting a theoretical understanding of dimensionality is step one. Next is noticing where it appears in the daily practice of data science. An experienced data scientist will be take actions to avoid it.  Prediction becomes control.

### Too many hyperparameters

An expensive mistake of the new data scientist is **excessive grid searching**.  Now that we understand the curse of dimensionality, we now that adding a single new value in our grid means a massive increase in the number of models you need to train.

A related mistake is **narrow grid searches** - a logarithmic scale will be more informative than a small linear range:

```python
#  this search isn't wide enough
useless_search = sklearn.model_selection.GridSearchCV(
    sklearn.ensemble.RandomForestRegressor(n_estimators=10), param_grid={'n_estimators': [10, 15, 20]
)

#  this search is more informative
useful_search = sklearn.model_selection.GridSearchCV(
    sklearn.ensemble.RandomForestRegressor(n_estimators=10), param_grid={'n_estimators': [10, 100, 1000]
)
```

### Too many features

I had this misunderstanding when working for the first six months as a data scientist.  Seeing the results in computer vision, where deep neural networks are state of the art, I thought that any neural network were ideal for any high dimensionality data.

The first mistake was not realizing that convolution provides useful inductive bias, specific to the structure found in images.  The second mistake was not appreciating the exponential cost of adding columns (commonly done when one-hot encoding).  Adding columns is expensive!

### Too many metrics

This is especially common for new data scientists working on regression problems - reporting a range of metrics (such as mean absolute error, mean absolute percentage error, root mean squared error etc) rather than a single metric.

Combine this with reporting a test & train error (or even one test & train per cross validation fold), the number of metrics becomes too many to glance at and make decisions with.  Pick one metric that best aligns with your business goal and stick with it - reduce the dimensionality of your metrics so you can take actions with them.

### Too many models

One of the pleasures of being a data scientist is the availability of high quality implementations of models in scikit-learn.  This can however be a problem when new data scientists repeatedly train a suite of models (say linear, svms and random forests) without a deliberate reason why these models should be looked at in parallel.  At most this should occur a few times, after which a single model is chosen and optimized.

Quite often I see a new data scientist train a linear model, an SVM and a random forest.  An experienced data scientist will just train the random forest (or XGBoost), and focus on using the feature importances to either engineer or drop features.

Why is a random forest a good first model?  A few reasons:
- they can be used for either regression or classification
- no scaling of target or features required
- no one-hot encoding
- training can be parallelized across CPU cores
- they work well on tabular data (win many Kaggle competitions)
- feature importances are interpretable

## Not seeing the interaction between prediction and control

The interaction between prediction and control drives reinforcement learning.  The better you are able to predict the world, the better you can control it.

This is also a working definition of a data scientist - **making predictions that lead to action**.

## Not understanding bias & variance

Error is the sum of bias, variance and noise.   **Noise is unmanageable** - no model can use it.

```python
error = bias + variance + noise
```

**Bias is a lack of signal** - the model misses seeing relationships that can be use to predict the target.  This is underfitting.  Bias can be reduced by increasing model capacity (either through more layers / trees, a different architecture or more features).

**Variance is the model fitting to noise** - patterns in the training data that will not appear in the data at test time.  This is overfitting.  Variance can be reduced by adding training data.

It is possible to have a model that is high bias & high variance - to simultaneously over and underfit.  Most of the time you will train a model with some bias & high variance, and your options will usually result in a tradeoff between bias & variance.

Increasing model capacity will reduce bias, but can increase variance (if that capacity is used to fit to noise).  Decreasing model capacity (such as through regularization) will reduce variance but can increase bias.

Another option is more data - this should reduce variance, but will have no effect on bias.  More data can even make bias worse - it gives your model the chance to give highly precise, wrong answers (see [Statistical Thinking for Data Science - SciPy 2015 - Chris Fonnesbeck](https:/www.youtube.com/watch?v=TGGGDpb04Yc)).

## Not thinking about where error comes from

Three common sources of error are in a dataset are:
- sampling error - arises from using statistics of a subset of a larger population
- sampling bias - samples having different probabilities than others
- measurement error - difference between measurement & true value

Is is the first step to understand these three sources of error while in statistics class - harder when you are working with a real dataset.

IID is another lens through which to look at sources of error.  IID stands for **independent & identically distributed**.  Non-independent sampling causes sampling bias, non-identical distributions is an example of sampling error.

## An obsession with the width & depth of fully connected neural nets

We saw above in *Not understanding bias & variance* that model capacity & architecture reduce model bias.  Yet when it comes to fully connected (aka feedforward aka dense) neural nets, the architecture doesn't matter so much.  As long as you give the model enough capacity and sensible hyperparameters, it should be able to learn.

A dense network has very little inductive bias - meaning that playing around with architecture isn't likely to help!  Let your gradients work with the capacity you give them.

Case in point - [Schulman et. al (2015) Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), which uses a simple feedforward neural network as a policy on locomotion tasks.  The locomotion tasks use a flat input vector, unlike the pixel based tasks, where a convolutional neural network is used.

<center><img src="/assets/mistakes-data-sci/trpo.png"></center>


The correct mindset with a fully connected neural network is a depth of two or three, with a width that is similar to your input (features) and output (target).  If this isn't enough capacity (i.e. your error metrics are indicating bias), consider adding a layer or some width.

The **wide & deep** architecture mixes wide memorization feature interactions with deep unseen, learned feature combinations.  I've never used one, but experimenting with this architecture seems like a better bet than adding a dense layer.

<center><img src="/assets/mistakes-data-sci/wide-deep.png"></center>

<center>Cheng et. al (2016) Wide & Deep Learning for Recommender Systems - https://arxiv.org/abs/1606.07792</center>

## Not paying attention to PEP 8

Code is run by computers but it is read, written and rewritten by humans.  PEP8 violations scream junior programmer.

```python
#  bad
Var=1

#  good
var = 1

# bad
def adder ( x =10 ,y= 5):
    return  x+y

# good
def adder(x=10, y=5):
    return x + y
```

The best way to pay attention to PEP8 is via automatic syntax highlighting - a good text editor will have this as a plugin.

## Not dropping the target

If you ever get a model with an impossibly low training error, it is likely that your target is a feature. 

```python
#  bad
data.drop('target', axis=1)

#  good
data.drop('target', axis=1, inplace=True)
data = data.drop('target', axis=1)
```

## Learning rate being too high

If there is one hyperparameter worthy of searching over when training neural networks it is learning rate (second is batch size).  Setting this too high will make training of neural networks unstable - LSTM's especially.

Batch size is less intuitive - a smaller batch will mean high variance gradients, but some of the value of using batches is having that variance to break out of local minima.

## Not scaling the target or features

Notice this error by seeing a high loss - anything higher than 2-3 is a sign that your target hasn't been scaled.  By scaling, I mean either standardization:

```python
standardized = (data - np.mean(data)) / np.std(data)
```

Or normalization:

```python
normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
```

The same requirement for scale applies to features as well (but not for random forests!).

## Not working with a sample of the data during development

A no-brainer when writing code that processes data - and an expensive mistake in developer time if you run the data processing using your full dataset each time you are fixing a bug.

You can work on a sample of your data roughly using an integer index:

```python
data = data[:1000]
```

Or more cleanly with a command line argument:

```python
parser.add_argument('--debug', default='local', nargs='?')
args = parser.parse_args()

#  paranoia!
debug = bool(int(args.debug))

if debug:
    data = data[:1000]
```

## Bonus - not using $HOME to store data

This one isn't a mistake - but it is a pattern that has made my life dramatically simpler.  

One issue you have when developing Python packages is that you don't know where the user will clone your repo, or where they scripts that execute the code will live.  This makes saving and loading data in a portable way difficult!

One solution to this is to create a folder in the user's `$HOME` directory, and use it to store data:

```python
import os

home = os.environ['HOME']
path = os.path.join(home, 'adg'))
os.makedirs(path, exist_ok=True)
np.save(path, data)
```

This means your work is portable - both to your colleagues and to remote machines.

---


Thanks for reading!
