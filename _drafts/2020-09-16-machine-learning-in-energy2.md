---
title: 'Machine Learning in Energy Two'
date: 2020-10-21
classes: wide2
toc: true
toc_sticky: true
categories:
  - Energy
  - Machine Learning
excerpt:  A guide to machine learning for energy professionals.
redirect_from: 
  - /machine-learning-in-energy-part-one/
  - /machine-learning-in-energy-part-two/

---

*Update history* 
- *2017-04-30 - original two part posts*
- *2019-04-25 - conoslidation of part one & two*
- *2020-10-21 - rewrite*


# Introduction

The energy industry is used to technological change.  Experienced engineers have seen transitions from coal to oil to gas - all within a single lifetime. We are now in the middle of another transition towards renewables.

In the last ten years, artificial intelligence & machine learning have revolutionized computing.  Is this revolution important for energy industry professionals?  What about machine learning - is this a revolution for energy?


## How to use this post

This post is aimed at two audiences:
- energy professionals wanting to understand how machine learning and energy fit together
- data scientists & machine learning engineers wanting get an energy industry perspective

This post is organized into three main sections - best read together but designed to be readable separately:
- section one - the concepts of AI, machine learning and deep learning
- section two - how machine learning intersects with energy
- section three - case studies of machine learning applications in energy

Understanding how these technologies impact the work of energy professionals and the industry at large is not simple.  That's what this post is about - helping energy industry professionals to understand:

- What are artificial intelligence, machine learning and deep learning?  How are they different?
- Is the hype justified?
- What potential does machine learning have to be used to solve energy industry problems?
- What will prevent a machine learning project being a success?
- How it is being used today to solve energy industry problems?

A secondary audience for this article are machine learning professionals who want to understand what impact their work can have in the energy industry.

The article is for suitable for both a technical and non-technical reader.


# SECTION ONE

> Machine learning is the biggest advance in how we can do engineering since the scientific method - Steve Juvertson

The last ten years has seen a revolution in artificial intelligence, powered by machine learning.  Machine learning has blown past the previous state of the art across a wide range of difficult problems.

![fig2]({{ '/assets/ml_energy/fig1.png' }})

*Recent progress in computer vision on the ImageNet benchmark - [The Malicious Use of Artificial Intelligence: Forecasting, Prevention, and Mitigation](https://arxiv.org/pdf/1802.07228.pdf)*

Currently, machine learning forms a core part of the breakthrough performance in computer vision and language processing.  The state of the art on tasks object recognition, image classification, speech recognition and language translation are all powered by a specific kind of machine learning - deep learning.

![fig1]({{ '/assets/ml_energy/gans.png' }})

*All the images to the right are machine generated - [Karras et. al (2019) A Style-Based Generator Architecture for Generative Adversarial Networks (NVIDIA)](https://arxiv.org/pdf/1812.04948.pdf)*

This section will clarify the relationship between these three concepts that are often misused & confused - artificial intelligence (AI), machine learning & deep learning:

- artificial intelligence = creating intelligent machines that can solve problems
- machine learning = a branch of AI, where machines learn from data
- deep learning = branch of machine learning, using neural networks

![fig2]({{ '/assets/ml_energy/ai-ml-dl.png' }})


# What is AI?

The pursuit of artificial intelligence (AI) is older than modern computing. Making intelligent machines that can solve our problems is a dream of many researchers.

This section will answer:

- where does intelligence stop and automation begin?
- is an elevator an AI?

The term *artificial* is clear - machines, computer programs.  Intelligence however is more complex.


## What is intelligence?

We find two concepts useful to understand intelligence - competence & comprehension:

- competence = the ability to perform a task well
- comprehension = the ability to understand

Let's take the task of operating an elevator.  Historically this task was done using human operators, who had both competence (to operate the elevator safely & efficiently) and comprehension (they understood what an elevator was, it's passengers etc).

Now elevators are operated by machines - rule based, automated systems.  These systems are competent, but lack comprehension.  

When and how comprehension arises is well beyond the scope of this article.  The main point here is that rule based systems (commonly grouped as automation) display competence - and if you value competence (the ability to do a task well), then many machines we live and work with today can be considered intelligent.

Many energy industry professionals will be familiar with the programmable logic controller (PLC) - a simple device responsible for controlling over 95% of industrial systems on the planet.  Is a PLC AI?

We've seen above that given a broader definition of intelligence, AI has been here for a while, in the form of rule based automation.  But rule based automation is not the only approach to AI - there are multiple approaches to creating intelligent machines. 

In order to understand the relationship between AI and machine learning, we will split AI into two broad approaches - rule based and learning based.


## Rule based AI 

As we have seen above, if we define competence (the ability to do a task well) as intelligence, then much of what we today call automation can also be called AI.

There are also other approaches to building AI based on rules, such as knowledge bases or expert systems. 

Because these systems are based on rules, human programmers need to develop these rules themselves.  This can lead to these systems being expensive (in terms of programmer time), and can limit the performance to human levels.

Because these systems are human designed, they can be understood by their programmers.


## Learning based AI

Another approach to creating intelligent machines is to let them learn for themselves.  In the case of machine learning, learning occurs from data.

Learning has a number of potential advantages over rule based systems:

- less effort required by programmer to design rules

Learning systems can also have some disadvantages

- programmer effort is replaced by computing costs
- reliance on data to learn from


## Case study - DeepBlue vs. AlphaGo

The crown jewel of modern reinforcement learning is the 2016 AlphaGo victory over Lee Sedol.  Go was the last great challenge for AI in board games - creating a superhuman Go computer was thought to be a decade away.  AlphaGo used deep neural networks to to map from the high dimensional board state to an optimal next move.

![]({{ '/assets/ml_energy/alphago.jpg' }})

*Lee Sedol in his eventual 4-1 loss to AlphaGo*

AlphaGo stands in contrast to Deep Blue, the computer that solved chess with a 1996 victory over Garay Kasparov.  All of Deep Blue's intelligence originated from a team of programmers and chess Grandmasters, who handcrafted moves into the machine.  Contrast this to the supervision that AlphaGo used - learning by playing itself at the game of Go.

DeepMind (the London lab behind AlphaGo) then introduced AlphaZero, which beat the previous version 100-0.  This version never relied on a dataset of human expert moves, and learnt entirely from adversarial self-play.


# What is machine learning?

Machine learning is the branch of artificial intelligence where machines learn from data.

This section will cover

Because machine learning is a subset of artificial intelligence, they should not be used interchangeably.  One reason machine learning is often confused with AI is how well modern machine learning is performing.  The other reason is that AI is better at generating hype - you'll likely raise more money for your startup if you are 'an innovative AI platform'.


## What is data?

Data is central to machine learning - it's one of the three things you need to get it to work:

<center>
  <code>machine learning = data + compute + algorithm</code>
</center>
<p></p>

Common forms of data in machine learning include:

- tabular
- images
- text
- audio
- time series

As this post is focused on energy, we ourselves will focus on two forms of data commonly collected in the energy industry.

It is impacting every industry - this ability stems from the capability of neural networks to learn from the same raw high dimensional data that we use and learn from, such as images or text.


![]({{ '/assets/ml_energy/tabular.png' }})


## Tabular data

Tabular data will be familiar to anyone who has spent time in Excel - data is organized into rows and columns.  Each row represents a sample of data, each column represents a certain feature or characteristic about each row.

A key feature of tabular data is that the order doesn't matter.


## Time series data

Time series data looks very similar to tabular data - the key difference is that now there


## Decision data

Order matters, action changes the data you see next.


## Case study - customer data

Let's now use an example to explain the difference between these three common types of data collected by energy companies - collecting an energy retail company (called EnergyCo) collecting information about their customers.

An example of tabular data EnergyCo would collect would be fixed infomation, such as where the customer lives or when they signed up:

| Name | Address         | Contract Start |
|------|-----------------|----------------|
| Bill | 123 Fake Street | 2016-01-01     |
| Jane | 321 Real Lane   | 2016-01-15     |

An example of time series data would be the monthly consumption of these customers - the data has a temporal & sequential structure:

| Month   | Name | Consumption |
|---------|------|-------------|
| 2016-01 | Bill | 116.8       |
| 2016-02 | Bill | 204.3       |
| 2016-01 | Jane | 50.4        |
| 2016-02 | Jane | 311.0       |

An example of decision data would be the data collected by EnergyCo's customer support team:

| Timestamp  | Customer Name | Reason for call  | Action                  |
|------------|---------------|------------------|-------------------------|
| 2016-01-03 | Bill          | Invoice too high | Reissue invoice         |
| 2016-02-15 | Jane          | Invoice late     | Update customer address |


## What tasks can machine learning do?

### Three branches of machine learning

To learn these patterns, machine learning makes use of three distinct learning signals, which separates machine learning into three branches.
There are three main tasks commonly done by machine learning:

- predicting
- generating data
- making decisions

can
- rl, vae, bayesian inference, image augmentation, clustering (smote), hidden markov models

Some may say augmentation != generation - but the aug is learnt from data same way gradients are

can't
- supervised learning,

## Prediction or control

both
- linear programs
- reinforcement learning

## Supervised learning

The first is **supervised learning**, where the machine used labelled training data to learn how to predict the labels of unseen data.  Examples include time series forecasting, computer vision and language translation.  Supervised learning is the reason why Facebook can tell which of your friends is in your photo, or why Google can translate text from on a photo on your smart phone.


## Unsupervised learning

The second is **unsupervised learning**, where the machine is able to generate new data without the supervision of labels.  Examples include artistic style transfer and generating realistic faces.

Generative Adversarial Networks (GANs) learn to generate realistic images using two competing neural networks.  One network generates images (the generator) and a second network has to decide if the image is real or fake.  This kind of adversarial learning is can be effective.


## Reinforcement learning

Adversarial learning can also be used in our final branch of machine learning - **reinforcement learning**.  In reinforcement learning the machine learns to select actions with the supervision of a scalar reward signal.  Reinforcement learning is applicable to a wide range of decision making problems with a reward signal, such as cost or carbon emissions.


## How does machine learning work?

The way machine learning works is by trial & error

![fig1]({{ '/assets/ml_energy/nn_black_box.png' }})



## Two perspectives on machine learning

- dimensionality reduction
- function approximation

Let's return to our equation for machine learning - the three things you need to get machine learning to work:

<center>
  <code>machine learning = data + compute + algorithm</code>
</center>
<p></p>

Machine learning is a deep topic - below we offer two useful perspectives to help you understand the complexity without having to dive deep into math or code.


## Perspective one - programming without explicit programming

Earlier when we discussed AI, we made a distinction between rule based AI and learning AI.

This is our first useful perspective on machine learning - **programming without explicit programming**.

With machine learning, supervision comes from data.  Lear

Learning from experience


## Perspective two - pattern recognition

A second useful perspective on machine learning is of pattern recognition.

A key challenge in pattern recognition is separating signal from the noise.  

This is known as generalization.

Examples of failing to generalize include:


Examples of successful generalization include:



These three 




# What is deep learning?

In this section you will understand:

- why deep learning works on data with structure


**Deep learning is foundation of the hype in modern machine learning.**  Deep learning is a family of machine learning models based on artificial neural networks - inspired by our own biological neural network (our brain).

Deep learning uses complex, many layered neural networks to learn patterns from large datasets.  

As with AI, we are going to split machine learning roughly into two branches - classical machine learning and deep learning.  To best understand this split, we need a third perspective on what machine learning is - function approximation.


https://youtu.be/0oyCUWLL_fU?t=584 (fast.ai)

- what deep learning is good at
- 


### Perspective three - function approximation


### How does function approximation relate to supervised, unsupervised & reinforcement learning?



## Classical machine learning


## Deep learning

Each of these three branches is undergoing a tremendous period of performance, research activity and hype.  Fundamental to all of this improvement is deep learning - the use of multiple layer neural networks as complex function approximators.

These artificial neural networks are inspired by the biological neural networks in our own brains.  The artificial neural networks used by machines are much simpler - yet they are powering the performance of modern machine learning.

A neural network is like any other function - it takes some input and produces an output.  Reducing a high dimensional sample of data to a lower dimension is the fundamental process in machine learning.  Examples include predicting solar power generation from satellite images, or dispatching a battery from grid data.


Neural networks are general purpose.  If neural networks were only applicable in computer vision, this would still be a huge deal.  Yet neural networks are pushing the boundaries in multiple directions.  This generalizability of neural networks has allowed them to be state of the art across a wide range of problems, and also allows machine learning to be applied to a wide range of industries.

The atomic unit of a neural network is the perceptron - a simple model that combines input from other perceptrons, squeezes it through a non-linear function (such as a sigmoid or rectifier) and sends output to child perceptrons.  The mathematics of this are beyond the scope of this article - the key takeaway is that stacking these perceptrons together into many layers allows neural networks to learn complex functions from large amounts of data.

There is more to machine learning than just deep neural networks - algorithms like logistic regression and random forests are suitable for many business problems.      The problem with classical machine learning is that it doesn't benefit from massive amounts of data.  Because the capacity of a neural network can be increased by adding depth, neural networks are able to break through the limits of classical machine learning models.

![fig3]({{ '/assets/ml_energy/fig2.png' }})

*Deep neural networks are able to learn from massive amounts of data  - [adapted from 'AI is the New Electricity' (Andrew Ng)](https://www.youtube.com/watch?v=21EiKfQYZXc)*

Layers are selected to take advantage of the structure in raw data.  Three common layers are fully connected, convolution and recurrent.

The ability to see and understand language not only drives performance, it also allows machine learning to generalize.  Vision and language understanding are low level skills used in essentially every domain of human life.  Mastering these low level skills means that machines can be useful in a range of industries.  Energy is no different.


# Is the hype justified?

Google, Amazon and Facebook all have world class AI labs and much of their business has been transformed by machine learning. The potential of machine learning is more latent in industries that are less digitized (such as healthcare, energy or education).


## What about general AI?

some researchers even think it's all we will need to solve the general intelligence problem.  What exactly is needed is unclear, but we are many breakthroughs away from providing general intelligence.  Narrow superhuman machine intelligence is already here.


## Achievements


## Challenges

energy = small data
- renewables plant only 2 years old
- combinigc datasets useful
- Poor state of digitization means working with small data is the primary work of energy data scientists


So far machine learning has provided narrow artificial intelligence (AI).  The power of these systems are often superhuman, and more than enough to justify the hype around machine learning.  Typically the task involves perception, using high dimensional data (i.e. images).  This is the main contribution of machine learning - being able to create business value from raw, high dimensional data.

This narrow intelligence stands in contrast to the goal of many AI researchers - general AI, where a machine can perform a single machine can variety of tasks.  While it is almost certain that machine learning will form part of a general artificial intelligence, much more is needed to provide an intelligent machine that can perform a variety of tasks.


There are a wide range of challenges in machine learning.  Examining them all is outside the scope of this article - issues such as interpretability, worker displacement and misuse of powerful narrow AI are significant issues and the focus of much research.  There is also much work to be done extending the powerful, narrow machine intelligence we have today into a general artificial intelligence.  Instead we will focus on challenges specific to using machine learning on energy problems.

The primary challenge is access to data.  The energy industry is still in the process of digitization - all my work in the energy has involved setting up the basic data collection infrastructure.  We've seen how important large amounts of data is to machine learning - a lack of historical data can be a show stopper for many energy and machine learning projects.

Forward thinking energy companies know that data can only be collected once.  It's more than just a local historian recording data from the site control system.  The 21st century energy company has everything from sensor level data to accounting data available to the employees and machines that need it, worldwide and in near real time.

The curation of large and interesting datasets is one of the few defensible advantages an energy company can build (another is brand).  These datasets are valuable not only because of how we can use them today, but because of the insights that can be generated tomorrow.


# Machine learning & energy

## Trends in energy

More things to measure (distributed), more measurements

## Trends in machine learning

## Why

- cost functions

## Why not

- unlearnable patterns - tasks we can't do
- no data
- non-stationary / unpredictable


# Applications

- cutomer segmentation
- chatbots
- customer service demand prediction
- churn prediction
- reccomendation emails to customers

- predictive maintenance
- forecasting
- control

When thinking about applying machine learning to an energy problem, the first and most important consideration is the dataset.  In fact, the first step in many machine learning projects is the same - start collecting data.  A dataset for supervised machine learning has two parts - the features (such as images or raw text) and the target (what you want to predict).  A dataset for reinforcement learning is a simulator - something that the learning algorithm can interact with.

Some potential applications of machine learning in energy include (but are not limited too):
- predictive maintenance 
- customer segmentation 
- churn prediction and minimization

Now we will dive into the details of a few applications of machine learning in energy that are happening today.

## Time series forecasting

The economics and environmental impact of energy depends on time of use.  Forecasting has always been an important practice in energy - increased deployment of variable wind and solar makes forecasting more valuable.  [DeepMind have claimed a 20 % improvement in the value of energy using a 36 hour ahead forecast](https://deepmind.com/blog/machine-learning-can-boost-value-wind-energy/).  Better forecasts can increase the value of renewables and reduce the requirement for backup fossil fuels.

![]({{ '/assets/ml_energy/wind.png' }})

Particularly exciting is the ability to forecast wind or solar using satellite images and deep convolutional neural nets - see the work of [Jack Kelly and Dan Travers at Open Climate Fix](https://openclimatefix.github.io/).

## Control

Optimal control of complex energy systems is hard.  Reinforcement learning is a framework for decision making that can be applied to a number of energy control problems, availability of reward signals, simulators

Better control of our energy systems will allow us to reduce cost, reduce environmental impact and improve safety.

![]({{"/assets/ml_energy/rl_energy.png"}})

[DeepMind's data centre optimization is the most famous example of energy and machine learning](https://deepmind.com/blog/safety-first-ai-autonomous-data-centre-cooling-and-industrial-control/).  The algorithm makes decisions on a five minute basis using thousands of sensors.  These sensors feed data into deep neural networks, which predict how different combinations of actions will affect the future efficiency of cooling generation.

This sounds like a form of DQN - a reinforcement learning algorithm that predicts future reward for each possible action.

The neural networks perform computations on the cloud, with the suggested action sent back to the data centre before safety verification by the local control system.

![fig2]({{ '/assets/ml_energy/data_centre.png' }})

*Performance of the data centre measured using energy input per ton of cooling (kW/tonC), and improves with more data, from an initial 12% to 30% over nine months.*

# Summary and Takeaways

- data challenge in energy
- neural nets can generalize across multiple problems -> ML can generalize
- why now

We've just had a whirlwind introduction to machine learning.  Key takeaways are:
- machine learning is the subset of AI that is working
- machine learning has three broad branches - supervised, unsupervised and reinforcement learning
- deep learning is powering modern machine learning
- convolution for vision, recurrent for sequences
- performance is driven by the availability of data, cloud compute and algorithms


# Further reading


# References
