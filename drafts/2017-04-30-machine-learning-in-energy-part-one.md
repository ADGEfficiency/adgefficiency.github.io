---
title: 'machine learning in energy - part one'
date: '2017-04-30'
categories:
  - Energy
  - Machine Learning
classes: wide

---

Artifical intelligence is finding it's way into everything.  Notice how good your autocorrect is lately?  The impact of artificial intelligence is felt almost daily by anyone with a cell phone.  Every vertical industry will be impacted by artifical intelligence as these industries digitize.  So what's been going on in the last few years that is causing all the hype?

The promise and potential of artificial intelligence is not new.  Early pioneers of computer science laid the foundations for modern computing as well the foundations of artifical intelligence. 

This is the first of a two part series.  This first post introduces what modern machine learning is, whats driving recent performance and highlights challenges facing the field today.  [The second post covers specific applications of machine learning in energy](http://adgefficiency.com/machine-learning-in-energy-part-two/).

## what is deep learning, machine learning and artificial intelligence?

The progress in modern artifical intelligence comes from **deep learning** - multiple layer artifical neural networks, whose structure is inspired by the biological neural networks in our own brain.   

What's exciting about neural networks is that they **learn**.  Being able to learn allows a machine to learn what we can't.  The ability to learn fundamental tasks means that almost every vertical industry (healthcare, education, energy etc) will be affected. 

Neural networks aren't the only algorithms that can learn - together with logistic regression, random forests and support vector machines they form a family of techniques known as **machine learning**.

**picture showing AI -> ML -> DL**

This heirarchy is important.  Artificial intelligence encompasses a lot more than just deep learning.  It's possible to be a machine learning practioner and not know much about other artifical intelligence techniques.  Most of the time when companies advertize they are using artifical intelligence, they really mean machine learning.

## learning what we can't 

Machine learning is a family of algorithms that **learn patterns in data**.  In contrast to traditional computer science where a human would tell a machine what to do, machine learning models figure out solutions for themselves.

This is a major paradigm shift.  Steve Juvertson (a venture capitalist who finished an eelectrical engineering degree at Stanford in 2 years!) thinks that "... machine learning is the biggest advance in how we can do engineering since the scientific method...".  

The ability to learn allows machine learning models to **exceed human performance in a variety of domains**.  Deep learning has blown past the previous state of the art in fundamental computer science problems such as computer vision and natural language understanding.

![fig1]({{ '/assets/ml_energy/fig1.png' }})
Figure 1 – Recent progress in compuer vision on the ImageNet benchmark [The Malicious Use of Artificial Intelligence: Forecasting, Prevention, and Mitigation](https://arxiv.org/pdf/1802.07228.pdf)

This ability of machine learning to generalize across different fundamental problems also allows machine learning models to **generalize across different industries**.  It's hard to think of a vertical industry that machine learning won't transform.

The paradigm shift of machine learning can be demonstrated by comparing two landmark achievements in artifical intelligence - IBM's Deep Blue and Alphabet/Google/DeepMind's AlphaGo.

In 1996 IBM's Deep Blue defeated World Chess Champion Gary Kasparov. IBMs Deep Blue derived it's playing strength from brute force computing power. All of Deep Blue's intelligence originated from a team of programmers and chess Grandmasters, who handcrafted moves into the machine.  

In 2016 Alphabet's AlphaGo defeated Go legend Lee Sedol 4-1. AlphaGo was not given any information about Go stragety from its programmers. Alpha Go used reinforcement learning to learn from its own actions and experience. **Machine learning allowed AlphaGo to learn on it's own, and become better than any human will ever be at the game of Go**.

One of the features of AlphaGo is how the game of Go was presented to AlphaGo.  The model was fed the positions of the stones on the board in it's raw form. This ability to learn from the raw structure of the world is fundamental to modern machine learning.


### seeing the structure of our world

The success of modern machine learning in solving fundamental computer science problems such as computer vision and natural language understanding comes from giving machines the ability to experience the structure of the world as we do.

We take for granted being able to experience the structure of the world.  We can see the physical strucutre of the world through sight.  We experience time - understanding the relationship between the past, present and future.  Modern machine learning allows machines to understand to this structure in a raw form.

Previously computers were given abstract visions of the world.  Performance required programmers to create these abstractions (also known as feature engineering) to reduce dimensionality and help the machine understand what parts of the data were important.  This dimensionality reduction is not ideal for two reasons - it both removes infomation and requires a lot of work for us to engineer this view of the world.

Modern machine learning is able to feed data in a raw form into models, so they can see these patterns for themselves.  Often they can see high dimensional patterns we would never be able to comprehend.  This 'getting out of the way of the machine' is a major paradigm shift in computer and data science.

The major advances in machine learning stem from **neural networks**.  The neural network is inspired by the biological neural networks in our brain.  Weights are used to connect multiple layers of artificial neurons together, and gradient descent slowly changes these weights so that the output of the network matches patterns that we want to learn.

Convolutional neural networks are inspired by our own visual cortex.  They allow machines to 'see' - taking the image as a raw input and being able to see what's in the image.  They can be used to classify the contents of the image, recongize faces or even to create captions for images.

Recurrent neural networks are the foundation of natural language processing and understanding. By processing the input sequentially and being able to remember (or forget) previous parts of the sequence, machines can learn the context of a word in a sentence.  The sequential nature of recurrent neural networks means they can experience the temporal dimension of time series problems.

Letting machines experience the structure of the world is fundamental to the promise of modern machine learning.  We see the raw structure of the world through sight - convolution allows machines to do the same.  We know that the world has a temporal structure - recurrent neural networks allow machines to understand this too.  **Seeing the world as it is allows machines to understand it**.

## Why now

Three major trends are driving modern machine learning. 

### one - data

**It's hard to overestimate the importance of data to modern machine learning.** 

The internet and smartphone cameras means we are generating more data than ever.  This volume of data is fundamental to driving modern machine learning.  

Training neural networks requires lots of examples.  If you are trying to classify images, a convolutional neural network requires a large number of example images in order to understand what different classes look like.

Data is most useful if it is labelled.  For example, an image of a dog outside a house that is labelled with `dog` and `house` is more valuable than the same photo with no label.  This is because the labels allow us to penalize the algorithm for labelling it with `cat`.  This kind of training is known as *supervised learning*.

For many large technology companies such as Alphabet or Facebook their labelled data has become a major source of the value of their businesses. A lot of this value comes from the insights that machines can learn (or will learn in the future) from such large data sets.

Sitting alongside supervised learning are *unsupervised learning* and *reinforcement learning*.  These three areas of machine learning differ in the kind of feedback that is available to the learner
- supervised learning = learns from a label for every sample of data
- unsupervised learning = learns from 'hidden' patterns in the data
- reinforcement learning = learns to maximize a scalar reward signal

Currently most of the business value of machine learning is driven by supervised learning.  Unsupervised and reinforcement learning are more challenging due to a lower quality learning signal - but also hold great promise precisely because of the potential to learn without supervision.

### two - hardware

Two hardware trends are driving modern machine learning.  The first is the use of **graphics processing units (GPUs)** and the second is the increased **availability** of computing power.

In the early 2000's computer scientists innovated the use of graphics cards designed for gamers for machine learning. GPUs gave massive descreases in training times - reducing them from months to days. Rendering graphics involves a lot of matrix multiplication, which GPUs are optimized for. Matrix multiplication is also the operation that is used in training neural networks.

This speed up is important. **Most of our understanding of machine learning is empirical** . This knowledge is built up a lot faster by reducing the training time of machine learning models.

The second trend is the availability of computing power - also known as cloud computing.  The cloud gives access to computation and data storage at scale on a variable cost basis.  Platforms such as Amazon Web Services or Google Cloud allow on-demand access to a large amount of GPU-enabled computing power with cheap data storage alongside it.

Researchers can eaisly run multiple experiments for a number of different hyperparameters, or train models distributed over multiple machines.

This access to computing power works both vertically within large technology companies and for smaller companies.  Access allows more companies to build machine learning products. It enables the balance sheet benefit of shifting a capital expense (building data centres) into an operating expense.

### three - algorithms & tools

Neural networks form the basis of many state of the art machine learning applications. Neural networks with multiple layers of non-linear processing units (known as deep learning) form the backbone of the most impressive applications of machine learning today. These artificial neural networks are inspired by the biological neural networks inside our brains.

Key components such as backpropagation have been around since the 1980's.  Backpropagation is the technique of assigning error to neural network weights.  The true potential of the theory needed data and computation to be unleashed.

GANS, dropout, batch norm????

Convolutional neural networks have revolutionised computer vision through a design based on the structure of our own visual cortex. Recurrent neural networks (specifically the LSTM implementation) have transformed sequence & natural language processing by allowing the network to hold state and remember.  

Another key trend in machine learning algorithms is the availability of open source tools. Companies such as Alphabet or Facebook make many of their machine learning tools all open source and available.

While these technology companies share their tools, they don't share their data. This is because data is the crucial element in producing value from machine learning. World-class tools and computing power are not enough to deliver value from machine learning - you need data to make the magic happen.

## challenges

### data

The reliance on a large amount of labelled training data is a neural network's achilles heel.  You can't train large neural networks with small amounts of data.

Human beings are able to learn from small amounts of training data - burning yourself once on the oven is enough to learn not to touch it again. Many machine learning algorithms are not able to learn in this way.  This is known as **sample inefficiency** - the requirement for large amounts of data to get superhuman performance.

### interpretability

Another problem is **interpretability**. Neural networks don't lend themselves to explanation. The high dimensionality of the input and parameter space means that it's hard to pin down cause to effect. Industries such as finance are legally prevented from using uninterpretable models.  

Related to this is our current lack of a theoretical understanding. Many academics are uncomfortable with machine learning. We can empirically test if machine learning is working, but we don't really know why it is working.  Modern machine learning is an empirical science - we don't really know why one algorithm works and another doesn't, we can only run experiments and test one versus the other. 

Training a neural network is like raising a child.  It's hard to understand exactly what you have made, and it's hard to predict what you are going to get.  A neural network evolves it's understanding in a way that doesn't lend itself to linear input/output interpretation that humans use to understand the world.  

We can learn about the process of training neural networks, in the same way that we can learn about different ways to raise children.  But like raising a child, we can't purposefully design what the network will end up looking like.

### misuse of narrow AI by humans 

The major risk and challenge in artificial intelligence is not from a superhuman AI that turns the world into paperclips.  It's from groups of humans using the superhuman ability of machine learning in specific, narrow areas in immoral ways.  Generative adverserial networks (GANs) can now be used to generate fake video - the final destination of 'fake news'.  Advanced computer vision techniques can be used to improve the tracking citizens throughout cities.

The reason these risks are higher than a superhuman AI is that narrow AI progress will always be ahead of more general AI.  We already have the ability to generate fake video using GANs.  I can reccomend the blog post [What Worries Me About AI by François Chollet](https://medium.com/@francois.chollet/what-worries-me-about-ai-ed9df072b704) that expands upon this misue of narrow AI concept in the context of social media.

### worker displacement

Worker displacement is a challenge as old as the Industrial revolution.  Political innovation (such as the universal basic income) are needed to fight the inequality that already exists at a ridiculous and immoral level.

**It is possible for us to deploy automation and machine learning while increasing the quality of life for all of society.** The move towards a machine intelligent world will be a positive one if we **share the value created**.

The issue with machine driven value is that it supercharges the already broken self fufilling capitalist prophecy - the rich getting richer.  

### digitization of the energy industry

The progress of machine learning differs across industries, roughly proportional to how digitized the industry is.  The rate at which modern machine learning can be utilized to extract business value depends on how data is collected, the quality of data and if it's available.  So far the technology industries are far ahead of healthcare, education and energy. 

Digtization is the necessary first step before any sort of machine learning can be done.  It requires both equipment to measure temperatures, flow rates and pressures and equipment to store that data.  

The energy system is poor at managing data.  Often data is not captured, meters are left uncalibrated or data is only stored on site for a fixed length of time.  Throughout my career I've spent significant effort getting systems in place so that data can be collected. 

Ideally everything from sensor level data to prices are accessible to employees & machines, worldwide in near real time.

It's not about having a local site plant control system and historian setup. The 21st-century energy company should have all data available in the cloud in real time. This will allow machine learning models deployed to the cloud to help improve the performance of our energy system. It's easier to deploy a virtual machine in the cloud than to install & maintain a dedicated system on site.

Data is one of the most strategic assets a company can own. It's valuable not only because of the insights it can generate today, but also the value that will be created in the future. **Data is an investment that will pay off.**

[Part Two goes into detail on specific applications of machine learning in the energy industry] - forecasting, energy disaggregation and reinforcement learning.  We also take a look at one of the most famous applications of machine learning in an energy system - Google’s work in their own data centers.(http://adgefficiency.com/machine-learning-in-energy-part-two/)

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
