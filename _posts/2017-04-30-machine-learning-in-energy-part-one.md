---
title: 'Machine learning in energy - part one'
date: 2017-04-30
categories:
  - Energy
  - Machine Learning
excerpt:  What is machine learning anyway? 

---

This is a two part series.  This first post introduces what machine learning is, why it's a fundamental paradigm shift, what's driving performance and what some of the challenges are. [The second post covers energy applications of machine learning](http://adgefficiency.com/machine-learning-in-energy-part-two/).

## what is machine learning 

Modern artifical intelligence is powered by **deep learning**.  Deep learning is why everyone is so excited about artificial intelligence.   There is reason for the hype - deep learning is now state of the art in fundamental problems such as computer vision and natural language understanding.  Deep learning is part of a larger group of algorithms known as **machine learning**.

> ... the business plans of the next 10,000 startups are easy to forecast: *Take X and add AI* - Kevin Kelly

![fig1]({{ '/assets/ml_energy/fig1.png' }})

**Figure 1 – Recent progress in compuer vision on the ImageNet benchmark [The Malicious Use of Artificial Intelligence: Forecasting, Prevention, and Mitigation](https://arxiv.org/pdf/1802.07228.pdf)**

## artifical intelligence vs machine learning vs deep learning

it's important to appreciate the difference between artificial intelligence, machine learning and deep learning. 

> Machine learning is the biggest advance in how we can do engineering since the scientific method - Steve Juvertson

Machine learning is a branch of artificial intelligence there are other approaches.  Machine learning is currently the part of artificial intelligence that is working the best.

The expressive mathematical power of machine learning algorithms - in particular deep learning - means that they generalize across multiple problems, such as computer vision or decision making.

There are approaches other than deep learning within machine learning - logistic regression, random forests and support vector machines are other machine learning algorithms that deliver business value today.  

Deep learning excels on problems where the structure of the network (i.e. convolution or recurrent) can take advantage of structure in data (i.e. images or text).  They are also able to make use (in fact require) massive amounts of data.  The problem with classicial machine learning is that it doesn't benefit much from massive amounts of data - see Figure 2.

![fig2]({{ '/assets/ml_energy/fig2.png' }})

**Figure 2 – amount of data versus machine learning algorithm performance.  [Adapted from AI is the new electricity - Andrew Ng](https://www.youtube.com/watch?v=21EiKfQYZXc)**

## learning what humans can't 

Traditionally humans would write code to tell a machine what to do.  Machine learning models figure out solutions for themselves by learning patterns in data.  **Learning allows machines to exceed human performance**.  

Deep learning has blown past the previous state of the art in fundamental computer science problems such as computer vision and natural language understanding.  It's also already better than any human at complex decision making problems such as Go.   Previously computers were limited by the intelligence of the programmer - now they are limited by the infomation available in data.

## Deep Blue vs. AlphaGo

The learning paradigm shift can be illuminated by comparing two landmark achievements in artifical intelligence - IBM's Deep Blue and Alphabet/Google/DeepMind's AlphaGo.

In 1996 IBM's Deep Blue defeated World Chess Champion Gary Kasparov. IBMs Deep Blue derived it's playing strength from brute force computing power and deterministic rules. All of Deep Blue's intelligence originated from a team of programmers and chess Grandmasters, who handcrafted moves into the machine.  This system without learning was able to beat Garay Kasparov.

Go however presents a more difficult decision making problem than chess.  The dimesionality of the state and action space make brute force approaches impossible and position evaulation challenging.  Go was the last great challenge in the classic games for artificial intelligence.  Consensus was that we were at least 10 years away from solving Go.  Then came AlphaGo.

![]({{ '/assets/ml_energy/alphago.jpg' }})

**Lee Sedol playing against AlphaGo in an eventual 4-1 loss**
In 2016 Alphabet's AlphaGo defeated Go legend Lee Sedol 4-1.  In contrast to Deep Blue, most of AlphaGo's intelligence originated from self play.  The original AlphaGo first learnt from human expert moves before learning from self play.  

A later iteration of the algorithm known as AlphaGo Zero learnt *tabula rasa* (I had to look it up too).  AlphaGo Zero learnt only from the experience of playing the game versus earlier versions of itself.  AlphaGo Zero is vastly superior to the previous verion - leaving human performance in the dust.

**Machine learning allowed AlphaGo to learn on it's own, and become better than any human will ever be at the game of Go**.  AlphaGo learnt to play go by being fed the board positions in their raw form.  This ability to learn from the raw structure of the world is fundamental to modern machine learning.

## seeing the structure of our world

Deep learning excels when it can take advantage of structure in the raw, senory level data.  This is important.  Using the same raw input as our eyes get, machines can see patterns in high dimensional space that we can't see.  Recurrent neural networks take advantage of the temporal structure in data - meaning machines can experience time.  This allows machines to break barriers in natural language.  They can understand the temporal strucutre in language.

Previously computers were given abstract visions of the world.  Features were hand engineerd by us to make problems eaiser for algorithms. Now deep learning generates it's own features.  This ability comes from the structure of the neural networks.  It also means humans don't have to do this job.

**This 'getting out of the way of the machine' is a major paradigm shift**.  Images and text have structure that can be used by neural networks of the correct structure (convolution for vision, recurrent for temporal problems).  Another key trend in deep learning is adverserial learning.  Both reinforcement learning and unsupervised learning (specifically GANS) use adverserial self play to drive learning.

The ability to see and understand language not only drives performance, it also allows machine learning to generalize.  Vision and language understanding are low level skills used in essentially every domain of human life.  Mastering these low level skills means that machines can be useful in a whole bunch of different contexts.

It's hard to think of a vertical industry that machine learning won't transform.  The limiting factor is digitization.  Internet technology is already digitized, whereas many medical records or energy systems are not.

## modern deep learning

Since deep learning is what everyone is excited about, lets dig into it deeper (ha).  Deep learning uses artifical neural networks with multiple layers.  How many are needed before a network is deep depends on who you talk too.  Simple networks such as a three layer fully connected network can be used to solve a variety of problems.  This is in contrast to massive convolutional or recurrent networks that can eaisly have 20 or more layers.

The artifical neural network is inspired by the biological neural networks in our brain.  Weights connect multiple layers of artificial neurons together, and gradient descent slowly changes these weights so that the output of the network matches patterns that we want to learn.  The network learns patterns in the data, the pattern of processing a high dimensional input into a more useful low dimensional input (such as pixels into a caption).

Convolutional neural networks are inspired by our own visual cortex.  They allow machines to 'see' - taking the image as a raw input and being able to see what's in the image.  They can be used to classify the contents of the image, recongize faces or even to create captions for images.

![]({{ '/assets/ml_energy/conv.png' }})

**[deep convolutional neural network used in the 2015 DeepMind Atari work](https://github.com/ADGEfficiency/dsr_rl/blob/master/literature/reinforcement_learning/2015_Minh_Atari_Nature.pdf)**

Recurrent neural networks get access to the temporal structure of data.  The inputs and outputs of the network are sent in sequences.  The network can learn whether to remember or forget earlier parts of the sequence.  Language has this structure (it matters what order words are in).  This makes recurrent neural networks a revolution in natural language processing and understanding. 

![]({{ '/assets/ml_energy/recurr.png' }})

**[an unrolled recurrent neural network](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)**

Convolutional and recurrent networks have the following similar features
- the network structure is designed to take advantage of the structure of the data
- performance increases as more data is fed in
- state of the art 
- use raw sensory level data as input

This last point is crucial.  Letting machines experience the structure of the world is fundamental to the promise of modern machine learning.  We see the raw structure of the world through sight - convolution allows machines to do the same.  We know that the world has a temporal structure - recurrent neural networks allow machines to understand this too.

## why now - three major trends

## one - data

It's hard to overestimate the importance of data to modern machine learning.  Today we produce more data than ever before.  The world is full of devices that generate data through sensing, communication and computation.  This titanic volume of data is fundamental to driving modern machine learning.  

**Training powerful neural networks requires a lot of data**.  Data is most useful if it is labelled.  For example, an image of a dog outside a house that is labelled with `dog` and `house` is more valuable than the same photo with no label.  This is because the labels allow us to penalize the algorithm for labelling it with `cat`.  This is *supervised learning*.

For many large technology companies such as Alphabet or Facebook their labelled data has become a major source of the value of their businesses. A lot of this value comes from the insights that machines can learn (or will learn in the future) from such large data sets.  Data has become one of the most valuable commodities - and it also provides a defensible position.  It's unlikely any company can match the big tech companies data quantity or quantity.

Sitting alongside supervised learning are *unsupervised learning* and *reinforcement learning*.  These three areas of machine learning differ in the feedback that is available to the learner.  Currently most of the business value of machine learning is driven by supervised learning.  Unsupervised and reinforcement learning are more challenging due to a lower quality learning signal - but also hold great promise precisely because of the potential to learn with less supervision.

- supervised learning = learns from a label for every sample of data
- unsupervised learning = learns from 'hidden' patterns in the data
- reinforcement learning = learns to maximize a scalar reward signal

## two - hardware

Two hardware trends are driving modern machine learning.  The first is the use of **graphics processing units (GPUs)** and the second is the increased **availability** of computing power.

In the early 2000's computer scientists innovated the use of graphics cards designed for gamers for machine learning.  GPUs were optimized to perform matrix multiplication - exactly the operation used in neural networks.  A GPU allows a massive speedup in machine learning training time by allowing vectorized and parallel mathematics.

This speed up is important. **Most of our understanding of machine learning is empirical**. This knowledge is built up faster by reducing the training time of machine learning models.

The second trend is the availability of computing power on the cloud.  The cloud gives access to computation and data storage at scale on a variable cost basis.  Platforms such as Amazon Web Services or Google Cloud allow on-demand access to a large amount of GPU-enabled computing power with cheap data storage alongside it.  Researchers can eaisly run multiple experiments for a number of different hyperparameters, or train models distributed over multiple machines.

This access to computing power works both vertically within large technology companies and for smaller companies.  Access allows more companies to build machine learning products. It enables the balance sheet benefit of shifting a capital expense (building data centres) into an operating expense.

## three - algorithms & tools

Key components of neural networks such as backpropagation have been around since the 1980's.  Backpropagation is the technique of assigning error to neural network weights.  The true potential of the theory needed data and computation to be unleashed.

Underpinning many of the advances are combinations of simpler techniques such as using rectified linear units, the ADAM optimizer, dropout, batch normalization and residual connections.

In unsupervised learning generative adverserial networks (GANs) can be used to generate realistic pictures.  The adverserial theme is also powering AlphaGo AlphaZero, reinforcement learning agents that use self-play to blow past human performance.

Another key trend in machine learning algorithms is the availability of open source tools and literature. Companies such as Alphabet or Facebook make many of their machine learning tools all open source and available.  Essentially all machine learning literature is published on open platforms such as arXiv.

While these technology companies share their tools and knowledge, they don't share their data. This is because data is the crucial element in producing value from machine learning. World-class tools and computing power are not enough to deliver value from machine learning - you need data to make the magic happen.

## challenges

## data

Being able to utilize massive amounts of data is the reason neural networks can learn so well.  Alternative algorithms such as random forests or support vector machines are limited by the amount of data they can learn from.  Neural networks are able to learn from massive amounts of data.

But this reliance on data is neural network's achilles heel.  You can't train large neural networks with small amounts of data.

Human beings are able to learn from small amounts of training data - burning yourself once on the oven is enough to learn not to touch it again. Many machine learning algorithms are not able to learn in this way.  This is known as **sample inefficiency** - the requirement for large amounts of data to get superhuman performance.  It's especially a problem in reinforcement learning, where the learning signal is weak. 

The dependence on data also creates a defensible moat for large technology companies.  The unique and massive datasets owned by Google and Facebook allow unique and powerful machine learning driven businesses.  This combined with the 'virtuous cycle of AI' (AI drives better products -> more data -> better products) means that machine learning will likely continue the dominance of a few large players.

## interpretability

Neural networks don't lend themselves to explanation. The high dimensionality of the input and parameter space means that it's hard to pin down cause to effect. Industries such as finance are legally prevented from using uninterpretable models - a lot of work is occuring in the financial sspace to make machine learning interpretable.  This is a key advantage of random forests - you can estimate feature importance in an interpretable way.

Related to this is our current lack of a theoretical understanding. Many academics are uncomfortable with machine learning. We can empirically test if machine learning is working, but we don't really know why it is working.  Modern machine learning is an empirical science - we don't really know why one algorithm works and another doesn't, we can only run experiments and test one versus the other. 

Training a neural network is like raising a child.  It's hard to understand exactly what you have made, and it's hard to predict what you are going to get.  A neural network evolves it's understanding in a way that doesn't lend itself to linear input/output interpretation that humans use to understand the world.  

We can learn about the process of training neural networks, in the same way that we can learn about different ways to raise children.  But like raising a child, we can't purposefully design what the network will end up looking like.

## misuse of narrow AI by humans 

The major risk and challenge in artificial intelligence is not from a superhuman AI that turns the world into paperclips.  It's from groups of humans using the superhuman ability of machine learning in specific, narrow areas in immoral ways.  GANs can now be used to generate fake video - the final destination of 'fake news'.  Advanced computer vision techniques can be used to improve the tracking citizens throughout cities by their governments.

The reason these risks are higher than a superhuman AI is that narrow AI progress will always be ahead of more general AI.  We already have the ability to generate fake video using GANs.  François Chollet (lead developer of Keras) goes into this in detail - see [What Worries Me About AI](https://medium.com/@francois.chollet/what-worries-me-about-ai-ed9df072b704), which expands upon this misue of narrow AI concept in the context of social media.

## worker displacement

Worker displacement is a challenge as old as the Industrial revolution.  Political innovation (such as the universal basic income) are needed to fight the inequality that already exists at a ridiculous and immoral level.

**It is possible to deploy automation and machine learning while increasing the quality of life for all of society.** The move towards a machine intelligent world will be a positive one if we **share the value created**.

The issue with machine driven value is that it supercharges the already broken self fufilling capitalist prophecy - the rich getting richer.  

## digitization of the energy industry

The progress of machine learning differs across industries, roughly proportional to how digitized the industry is.  The rate at which modern machine learning can be utilized to extract business value depends on how data is collected, the quality of data and if it's available.  So far the technology industries are far ahead of healthcare, education and energy. 

Digtization is the necessary first step before any machine learning can be done.  It requires both equipment to measure temperatures, flow rates and pressures and equipment to store that data.  

The energy system is poor at managing data.  Often data is not captured, meters are left uncalibrated or data is only stored on site for a fixed length of time.  Throughout my career I've spent significant effort getting systems in place so that data can be collected. 

Ideally everything from sensor level data to prices are accessible to employees & machines, worldwide in near real time.

It's not about having a local site plant control system and historian setup. The 21st-century energy company should have all data available in the cloud in real time. This will allow machine learning models deployed to the cloud to help improve the performance of our energy system. It's easier to deploy a virtual machine in the cloud than to install & maintain a dedicated system on site.

Data is one of the most strategic assets a company can own. It's valuable not only because of the insights it can generate today, but also the value that will be created in the future. **Data is an investment that will pay off.**

[Read the second part on machine learning applications in energy](http://adgefficiency.com/machine-learning-in-energy-part-two/).

Thanks for reading!

## Sources and further reading

* [Demis Hassabis (CEO DeepMind) &#8211; Artificial Intelligence and the Future](https://www.youtube.com/watch?v=i3lEG6aRGm8)
* <a style="font-size: 1rem;" href="https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer)">Deep Blue &#8211; Wikipedia</a>
* <a style="font-size: 1rem;" href="https://en.wikipedia.org/wiki/AlphaGo">Alpha Go &#8211; Wikipedia</a>
* [Oh & Jung (2004) GPU implementation of neural networks](http://www.sciencedirect.com/science/article/pii/S0031320304000524)
* [The financial world wants to open AI's boxes &#8211; MIT Technology Review](https://www.technologyreview.com/s/604122/the-financial-world-wants-to-open-ais-black-boxes/)
* [Artificial Intelligence is the New Electricity - Andrew Ng](https://www.youtube.com/watch?v=zWQOJ001PDs)
* [Nuts and Bolts of Applying Deep Learning - Andrew Ng](https://www.youtube.com/watch?v=F1ka6a13S9I)
* [Paperclip maximizer](https://wiki.lesswrong.com/wiki/Paperclip_maximizer)
* [What worries me about AI – François Chollet](https://medium.com/@francois.chollet/what-worries-me-about-ai-ed9df072b704)
* [Software 2.0 - Andrej Karpathy (Tesla & OpenAI) ](https://medium.com/@karpathy/software-2-0-a64152b37c35)
