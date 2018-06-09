---
title: 'machine learning in energy - part one'
date: 2017-04-30
categories:
  - Energy
  - Machine Learning

---

This is the first of a two part series - [read the second part on energy machine learning applications](http://adgefficiency.com/machine-learning-in-energy-part-two/).

## what is machine learning 

Modern artifical intelligence is powered by **deep learning** - multiple layer artifical neural networks.  These artifical networks are inspired by the biological neural networks in our own brain.  Deep neural networks are why everyone is so excited about the near future in artificial intelligence. 

![fig1]({{ '/assets/ml_energy/fig1.png' }})

**Figure 1 – Recent progress in compuer vision on the ImageNet benchmark**

**[The Malicious Use of Artificial Intelligence: Forecasting, Prevention, and Mitigation](https://arxiv.org/pdf/1802.07228.pdf)**

Neural networks are exciting because they **learn**.  They aren't the only algorithms that can learn - along with logistic regression, random forests and support vector machines they form a family of techniques known as **machine learning**.

### *artifical intelligence > machine learning > deep learning*

Artificial intelligence encompasses more than just deep learning.  It's possible to be a successful machine learning practioner and not know much about other areas of artifical intelligence.  

Most of the time when companies advertize they are using artifical intelligence, they really mean machine learning.  Both are correct, but machine learning is more correct.

Machine learning doesn't always mean deep learning.  Deep learning excels on image and text problems - more classical models such as random forests perform better on other kinds of datasets.  

## learning what humans can't 

Machine learning is a family of algorithms that **learn patterns in data**.  Traditionally humans would write code to tell a machine what to do.  Machine learning models figure out solutions for themselves.  Data is the teacher.

> Machine learning is the biggest advance in how we can do engineering since the scientific method - Steve Juvertson

**Learning allows machines to exceed human performance**.  Deep learning has blown past the previous state of the art in fundamental computer science problems such as computer vision and natural language understanding.

Giving machines the ability to see and understand language allows machine learning to generalize into different domains.  The same vision models can be used for collision detection in autonomous driving or for spotting cancer in x-rays.  The same language models can be used to both listen to and respond in conversations on any topic.

This ability to see and learn can be used in almost any industry.  It's hard to think of a vertical industry that machine learning won't transform.  The limiting factor is usually digitization.  Internet technology is already digitized, whereas many medical records or energy systems are not.

## Deep Blue vs. AlphaGo

This paradigm shift can be demonstrated by comparing two landmark achievements in artifical intelligence - IBM's Deep Blue and Alphabet/Google/DeepMind's AlphaGo.

In 1996 IBM's Deep Blue defeated World Chess Champion Gary Kasparov. IBMs Deep Blue derived it's playing strength from brute force computing power. All of Deep Blue's intelligence originated from a team of programmers and chess Grandmasters, who handcrafted moves into the machine.  

In 2016 Alphabet's AlphaGo defeated Go legend Lee Sedol 4-1. AlphaGo was not given any information about Go stragety from its programmers. Alpha Go used reinforcement learning to learn from its own actions and experience. **Machine learning allowed AlphaGo to learn on it's own, and become better than any human will ever be at the game of Go**.

![]({{ '/assets/ml_energy/alphago.jpg' }})

**Lee Sedol playing against AlphaGo in an eventual 4-1 loss**

AlphaGo learnt to play go by being fed the board positions in their raw form.  This ability to learn from the raw structure of the world is fundamental to modern machine learning.

## seeing the structure of our world

The success of modern machine learning in solving computer vision and natural language understanding comes from giving machines the ability to experience the structure of the world.

We take this for granted.  We can see the physical strucutre of the world through sight.  We experience time - understanding the relationship between the past, present and future.  Modern machine learning allows machines to understand to this structure using the same inputs that we do.

Previously computers were given abstract visions of the world.  Performance required programmers to create these abstractions (also known as feature engineering) to reduce dimensionality and help the machine understand what parts of the data were important.  This dimensionality reduction is not ideal for two reasons - it both removes infomation and requires us to engineer this view of the world.

Modern machine learning is able to feed data in a raw form into models, so they can see these patterns for themselves.  Often they can see high dimensional patterns we would never be able to comprehend.  This 'getting out of the way of the machine' is a major paradigm shift in computer and data science.

## modern deep learning

Deep learning refers to neural networks with multiple layers.  How many are needed before a network is deep depends on who you talk too.  Simple networks such as a three layer fully connected network can be used to solve a variety of problems.  This is in contrast to massive convolutional or recurrent networks that can have 20 or more layers.

The artifical neural network is inspired by the biological neural networks in our brain.  Weights connect multiple layers of artificial neurons together, and gradient descent slowly changes these weights so that the output of the network matches patterns that we want to learn.

Convolutional neural networks are inspired by our own visual cortex.  They allow machines to 'see' - taking the image as a raw input and being able to see what's in the image.  They can be used to classify the contents of the image, recongize faces or even to create captions for images.

![]({{ '/assets/ml_energy/conv.png' }})

**[deep convolutional neural network used in the 2015 DeepMind Atari work](https://github.com/ADGEfficiency/dsr_rl/blob/master/literature/reinforcement_learning/2015_Minh_Atari_Nature.pdf)**

Recurrent neural networks process sequences.  For each part of the sequence the network chooses whether to remember or forget.  As the sequence is processed the network can remember what is important to improve the prediction.

Recurrent neural networks model the temporal structure of data.  Language has this structure (it matters what order words are in).  This makes recurrent neural networks foundational in natural language processing and understanding. 

![]({{ '/assets/ml_energy/recurr.png' }})

**[an unrolled recurrent neural network](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)**

Convolutional and recurrent networks have the following similar features
- the network structure is designed to take advantage of the structure of the data
- performance increases as more data is fed in
- state of the art 
- use raw sensory level data as input

This last point is crucial.  

Letting machines experience the structure of the world is fundamental to the promise of modern machine learning.  We see the raw structure of the world through sight - convolution allows machines to do the same.  We know that the world has a temporal structure - recurrent neural networks allow machines to understand this too.  **Seeing the world as it is allows machines to understand it**.

## why now - three major trends

## one - data

It's hard to overestimate the importance of data to modern machine learning.  The internet and smartphone cameras means we are generating more data than ever.  This volume of data is fundamental to driving modern machine learning.  

**Training powerful neural networks requires lots of data**.  If you are trying to classify images, a convolutional neural network requires a large number of example images to understand what different classes look like.

Data is most useful if it is labelled.  For example, an image of a dog outside a house that is labelled with `dog` and `house` is more valuable than the same photo with no label.  This is because the labels allow us to penalize the algorithm for labelling it with `cat`.  This is *supervised learning*.

For many large technology companies such as Alphabet or Facebook their labelled data has become a major source of the value of their businesses. A lot of this value comes from the insights that machines can learn (or will learn in the future) from such large data sets.

Sitting alongside supervised learning are *unsupervised learning* and *reinforcement learning*.  These three areas of machine learning differ in the feedback that is available to the learner
- supervised learning = learns from a label for every sample of data
- unsupervised learning = learns from 'hidden' patterns in the data
- reinforcement learning = learns to maximize a scalar reward signal

Currently most of the business value of machine learning is driven by supervised learning.  Unsupervised and reinforcement learning are more challenging due to a lower quality learning signal - but also hold great promise precisely because of the potential to learn with less supervision.

## two - hardware

Two hardware trends are driving modern machine learning.  The first is the use of **graphics processing units (GPUs)** and the second is the increased **availability** of computing power.

In the early 2000's computer scientists innovated the use of graphics cards designed for gamers for machine learning. GPUs gave massive descreases in training times - reducing them from months to days. Rendering graphics involves matrix multiplication, which GPUs are optimized for. Matrix multiplication is also the operation that is used in training neural networks.

This speed up is important. **Most of our understanding of machine learning is empirical**. This knowledge is built up faster by reducing the training time of machine learning models.

The second trend is the availability of computing power - also known as cloud computing.  The cloud gives access to computation and data storage at scale on a variable cost basis.  Platforms such as Amazon Web Services or Google Cloud allow on-demand access to a large amount of GPU-enabled computing power with cheap data storage alongside it.

Researchers can eaisly run multiple experiments for a number of different hyperparameters, or train models distributed over multiple machines.

This access to computing power works both vertically within large technology companies and for smaller companies.  Access allows more companies to build machine learning products. It enables the balance sheet benefit of shifting a capital expense (building data centres) into an operating expense.

## three - algorithms & tools

Key components of neural networks such as backpropagation have been around since the 1980's.  Backpropagation is the technique of assigning error to neural network weights.  The true potential of the theory needed data and computation to be unleashed.

Underpinning many of the advances are combinations of simpler techniques such as using rectified linear units, the ADAM optimizer, dropout, batch normalization and residual connections.

In unsupervised learning generative adverserial networks (GANs) can be used to generate realistic pictures.  The adverserial theme is also powering AlphaGo AlphaZero, reinforcement learning agents that use self-play to blow past human performance.

Another key trend in machine learning algorithms is the availability of open source tools and literature. Companies such as Alphabet or Facebook make many of their machine learning tools all open source and available.  Almost all of the machine learning literature is published on open platforms such as arXiv.

While these technology companies share their tools and knowledge, they don't share their data. This is because data is the crucial element in producing value from machine learning. World-class tools and computing power are not enough to deliver value from machine learning - you need data to make the magic happen.

## challenges

## data

Being able to utilize massive amounts of data is the reason neural networks can learn so well.  Alternative algorithms such as random forests or support vector machines are limited by the amount of data they can learn from.  Neural networks are able to learn from massive amounts of data.

But this reliance on data is neural network's achilles heel.  You can't train large neural networks with small amounts of data.

Human beings are able to learn from small amounts of training data - burning yourself once on the oven is enough to learn not to touch it again. Many machine learning algorithms are not able to learn in this way.  This is known as **sample inefficiency** - the requirement for large amounts of data to get superhuman performance.

## interpretability

Neural networks don't lend themselves to explanation. The high dimensionality of the input and parameter space means that it's hard to pin down cause to effect. Industries such as finance are legally prevented from using uninterpretable models.  

Related to this is our current lack of a theoretical understanding. Many academics are uncomfortable with machine learning. We can empirically test if machine learning is working, but we don't really know why it is working.  Modern machine learning is an empirical science - we don't really know why one algorithm works and another doesn't, we can only run experiments and test one versus the other. 

Training a neural network is like raising a child.  It's hard to understand exactly what you have made, and it's hard to predict what you are going to get.  A neural network evolves it's understanding in a way that doesn't lend itself to linear input/output interpretation that humans use to understand the world.  

We can learn about the process of training neural networks, in the same way that we can learn about different ways to raise children.  But like raising a child, we can't purposefully design what the network will end up looking like.

## misuse of narrow AI by humans 

The major risk and challenge in artificial intelligence is not from a superhuman AI that turns the world into paperclips.  It's from groups of humans using the superhuman ability of machine learning in specific, narrow areas in immoral ways.  GANs can now be used to generate fake video - the final destination of 'fake news'.  Advanced computer vision techniques can be used to improve the tracking citizens throughout cities by their governments.

The reason these risks are higher than a superhuman AI is that narrow AI progress will always be ahead of more general AI.  We already have the ability to generate fake video using GANs.  I can reccomend the blog post [What Worries Me About AI by François Chollet](https://medium.com/@francois.chollet/what-worries-me-about-ai-ed9df072b704) that expands upon this misue of narrow AI concept in the context of social media.

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

[Read the second part](http://adgefficiency.com/machine-learning-in-energy-part-two/) on machine learning applications in energy.

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
