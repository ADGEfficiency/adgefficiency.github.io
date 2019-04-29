---
title: 'Machine Learning in Energy'
date: 2019-04-25
categories:
  - Energy
  - Machine Learning
excerpt:  What is machine learning anyway?

---

This post is aimed at introducing energy industry professionals with to machine learning, and answers the following questions:
- what is machine learning?
- why is machine learning so hot right now?
- what is driving performance?
- what are the challenges?

We will then take a survey current applications of energy and machine learning, and ponder what might happen in the future.

## What is machine learning?

> The business plans of the next 10,000 startups are easy to forecast: *Take X and add AI* - Kevin Kelly

> Machine learning is the biggest advance in how we can do engineering since the scientific method - Steve Juvertson

The hype has officially peaked - deep learning is right at the top of the peak of inflated expectations.

![]({{ '/assets/ml_energy/hype.png' }})

[The 2018 Gartner Hype Cycle](https://www.gartner.com/smarterwithgartner/5-trends-emerge-in-gartner-hype-cycle-for-emerging-technologies-2018/)

Deep learning is foundation of the hype in modern machine learning.  Deep learning uses complex, many layered neural networks to learn patterns from large datasets.  This is the primary intelligence of machine learning - pattern recognition.  Giving computers the ability to learn high dimensional patterns from data defines modern machine learning.

Machine learning has blown past the previous state of the art across a wide range of problems, and is impacting a wide range of industries.  The ability of machine learning to generalize across industries stems from the ability of neural networks to learn from images and text - the same raw high dimensional data that we use and learn from.

So where are we today?  Currently, machine learning forms a core part of the breakthrough performance of computer vision and language processing.  In both cases, deep neural networks are used to transform high dimensional data into valuable insights.  Examples include patterns include predicting power generation and dispatching battery storage.

## Machine learning versus artificial intelligence

So far machine learning has provided a number of examples of superhuman narrow artificial intelligence (AI), in areas such as computer vision, language processing and playing complex board games such as chess and Go. But in all these cases, the system has been specially designed for that single tasks.

While it's likely that machine learning will form part of a more general artificial intelligence, much more is needed to provide an intelligent machine that can perform a variety of tasks.

This means that machine learning and artificial intelligence (AI) shouldn't be used interchangeably.  Machine learning is only a subset of the broader field of AI.  AI encompasses multiple distinct approaches that are beyond both the scope of this article.

One reason for this is how well modern machine learning is performing - some researchers even think it's all we will need to solve the general intelligence problem.  What exactly is needed is unclear, and we are many breakthroughs away from providing general intelligence.

![fig2]({{ '/assets/ml_energy/ai_ml.png' }})

## Three branches of machine learning

To learn these patterns, machine learning makes use of three distinct learning signals, which can be used to separate machine learning into three branches.

The first is **supervised learning**, where the machine used labelled training data to learn how to predict the labels of unseen data.  Examples include time series forecasting, computer vision and language translation.  Supervised learning is the reason why Facebook can tell which of your friends is in your photo, or why Google can translate text from on a photo on your smart phone.

![fig2]({{ '/assets/ml_energy/fig1.png' }})

*Recent progress in computer vision on the ImageNet benchmark - [The Malicious Use of Artificial Intelligence: Forecasting, Prevention, and Mitigation](https://arxiv.org/pdf/1802.07228.pdf)*

The second is **unsupervised learning**, where the machine is able to generate new data without the supervision of labels.  Examples include artistic style transfer and generating realistic faces.

Generative Adversarial Networks (GANs) learn to generate realistic images using two competing neural networks.  One network generates images (the generator) and a second network has to decide if the image is real or fake.  This kind of adversarial learning is can be effective.

![fig1]({{ '/assets/ml_energy/gans.png' }})

*All the images to the right are machine generated - [Karras et. al (2019) A Style-Based Generator Architecture for Generative Adversarial Networks (NVIDIA)](https://arxiv.org/pdf/1812.04948.pdf)*

Adversarial learning can also be used in our final branch of machine learning - **reinforcement learning**.  In reinforcement learning the machine learns to select actions with the supervision of a scalar reward signal.  Reinforcement learning is applicable to a wide range of decision making problems with a reward signal, such as cost or carbon emissions.

The crown jewel of modern reinforcement learning is the 2016 AlphaGo victory over Lee Sedol.  Go was the last great challenge for AI in board games - creating a superhuman Go computer was thought to be a decade away.  AlphaGo stands in contrast to Deep Blue, the computer that solved chess with a 1996 victory over Garay Kasparov.

![]({{ '/assets/ml_energy/alphago.jpg' }})

*Lee Sedol in his eventual 4-1 loss to AlphaGo*

All of Deep Blue's intelligence originated from a team of programmers and chess Grandmasters, who handcrafted moves into the machine.  Contrast this to the supervision that AlphaGo used - learning by playing itself at the game of Go.

DeepMind (the London based lab behind AlphaGo) then introduced AlphaZero, which beat the previous version 100-0.  This version never relied on a dataset of human expert moves, and learnt entirely from adversarial self-play.

## Why is machine learning so hot right now?

Each of these three branches is undergoing a tremendous period of performance, research activity and hype.  The power behind all of these is deep learning - the use of multiple layer neural networks as complex function approximators.  

These artificial neural networks are inspired by the biological neural networks in our own brains.  The artificial neural networks used by machines are much simpler - yet they are powering the performance of modern machine learning.

A neural network is like any other function - it takes some input and produces an output.  Reducing a high dimensional sample of data to a lower dimension is the fundamental process in machine learning.  Examples include predicting solar power generation from satellite images, or dispatching a battery from grid data.

![fig1]({{ '/assets/ml_energy/nn_black_box.png' }})

Neural networks are general purpose.  If neural networks were only applicable in computer vision, this would still be a huge deal.  Yet neural networks are pushing the boundaries in multiple directions.  This generalizability of neural networks has allowed them to be state of the art across a wide range of problems.

The atomic unit of a neural network is the perceptron - a simple model that combines input from other perceptrons, squeezes it through a non-linear function (such as a sigmoid or rectifier) and sends output to child perceptrons.  The mathematics of this are beyond the scope of this article - the key takeaway is that stacking these perceptrons together into many layers allows neural networks to learn complex functions from large amounts of data.

The depth is what allows neural networks to break through the limits of classical machine learning and increase model capacity by adding more layers.  There is more to machine learning than just deep neural networks - algorithms like logistic regression and random forests are suitable for many business problems.  The problem with classical machine learning is that it doesn't benefit from massive amounts of data.

![fig3]({{ '/assets/ml_energy/fig2.png' }})

*Deep neural networks are able to learn from massive amounts of data  - [adapted from 'AI is the New Electricity' (Andrew Ng)](https://www.youtube.com/watch?v=21EiKfQYZXc)*

The performance of deep neural networks do scale with data.  Layers can be selected to take advantage of the structure in raw data.  Three common layers are fully connected, convolution and recurrent.

![]({{ '/assets/ml_energy/conv.png' }})

*Deep convolutional neural network used in the [2015 DeepMind Atari work](https://github.com/ADGEfficiency/dsr_rl/blob/master/literature/reinforcement_learning/2015_Minh_Atari_Nature.pdf)*

The convolutional layer is inspired by our own visual cortex, and is what powers modern computer vision.  They allow machines to see.  They can be used to classify the contents of the image, recognize faces and create captions for images.

![]({{ '/assets/ml_energy/recurr.png' }})

*An unrolled recurrent neural network - [colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)*

A recurrent layer processes input and generates output as sequences, and powers modern natural language processing.  Recurrent networks allow machines to understand the temporal structure in data, such as words in a sentence.

The ability to see and understand language not only drives performance, it also allows machine learning to generalize.  Vision and language understanding are low level skills used in essentially every domain of human life.  Mastering these low level skills means that machines can be useful in a range of industries.  Energy is no different.

## What's driving the performance of modern machine learning?

The performance of deep neural networks is driven by three things - access and availability of data, computational resources and algorithms and tools.

The ability for large capacity neural networks to learn from large amounts of data is what separates deep learning from classical machine learning.  This ability is also a dependence - deep neural networks need lots of data to learn anything useful.  This sample inefficiency problem is not shared in human learning.  We can learn not to touch the stove after doing it once - modern machine learning would require many samples to learn this.

The rise of the connected devices using the internet has lead to a combinatorial explosion of data generation.  This raw data is often high dimensional, such as satellite images.  The flexibility of neural networks allows the model to be setup to learn the structure in raw data.

This is another dividing line between modern and classical machine learning.  Hand crafted feature engineering, to preprocess the raw data before being used by a model such as a random forest, was common.  Neural networks do this feature engineering themselves, as part of the learning process.  In fact the more we let neural networks do, the better they get.  This 'getting out of the way of the machine' trend is present right at the edge of the state of the art.

There are two sub trends within computational hardware.  The first is the use of graphics processing units (GPUs) to speed up the training of deep neural networks.  In the early 2010's researchers repurposed GPUs designed for video games for training neural networks, and training times reduced from weeks to days.

This speed up is important. Most of our understanding of machine learning is empirical. This knowledge is built up faster by reducing the training time of machine learning models.

The second hardware trend is cloud computing.  The cloud gives access to computation on a fully variable cost basis.  Platforms such as Amazon Web Services allow on-demand access to a large amount of GPU-enabled computing power with cheap data storage alongside it.  This access to computing power works both vertically within large technology companies and for smaller companies.  It enables the balance sheet benefit of shifting a capital expense (building data centres) into an operating expense.

Our final trend driving modern machine learning is access to algorithms and tools.  Almost all the relevant literature for machine learning is available for free on sites like arXiv.  It's also possible to access high quality implementations of machine learning tools on GitHub.  This tendency for openness stands in stark contrast with the paywalls and licensed software of the energy industry.

## Challenges

There are a wide range of challenges in machine learning.  Examining them all is outside the scope of this article - issues such as interpretability, worker displacement and misuse of powerful narrow AI are significant issues and the focus of much research.  Instead the focus will be on challenges specific to using machine learning on energy problems.

The primary challenge is access to data.  The energy industry is still in the process of digitization - all my work in the energy has involved setting up the basic data collection infrastructure.  We've seen how important large amounts of data is to machine learning - a lack of historical data can be a show stopper for many energy and machine learning projects.

Forward thinking energy companies know that data can only be collected once.  It's more than just a local historian recording data from the site control system.  The 21st century energy company has everything from sensor level data to accounting data available to the employees and machines that need it, worldwide and in near real time.

The curation of large and interesting datasets is one of the few defensible advantages an energy company can build (another is brand).  These datasets are valuable not only because of how we can use them today, but because of the insights that can be generated tomorrow.

# Applications 

We've just had a whirlwind introduction to machine learning.  Key takeaways are:
- machine learning is the subset of AI that is working
- machine learning has three broad branches - supervised, unsupervised and reinforcement learning
- deep learning is powering modern machine learning
- convolution for vision, recurrent for sequences
- performance is driven by the availability of data, cloud compute and algorithms

Now we will dive into the details of a few applications of energy and machine learning.  Other applications include (but are not limited too) predictive maintenance, customer segmentation, churn prediction and minimization.

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

# Summary and key takeaways

# Sources and further reading

For more technical and non-technical machine learning resources, check out [ml-resources](https://github.com/ADGEfficiency/ml-resources).  For reinforcement learning resources, check out [rl-resources](https://github.com/ADGEfficiency/rl-resources).

Time series forecasting

- [Machine Learning in Energy (Fayadhoi Ibrahima)](http://large.stanford.edu/courses/2015/ph240/ibrahima2/)
- [7 reasons why utilities should be using machine learning](https://blogs.oracle.com/utilities/utilities-machine-learning)
- [Germany enlists machine learning to boost renewables revolution](http://www.nature.com/news/germany-enlists-machine-learning-to-boost-renewables-revolution-1.20251)
- [Weron (2014) Electricity price forecasting: A review of the state-of-the-art with a look into the future](http://www.sciencedirect.com/science/article/pii/S0169207014001083)
- [Re-dispatch costs in the German power grid](https://www.cleanenergywire.org/factsheets/re-dispatch-costs-german-power-grid)
- [This “duck curve” is solar energy’s greatest challenge](https://www.vox.com/2018/5/9/17336330/duck-curve-solar-energy-supply-demand-problem-caiso-nrel)

Thanks for reading!
