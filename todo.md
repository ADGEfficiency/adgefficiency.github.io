# Machine learning and energy project checklist
In general, if a rules based system can solve a problem, it is probably the way to go.

Data challenge
- data flows

maybe a separate post

---

should applications go further up?

- tense=present & together (let us learn etc)
- check lessons ML, lessons energy
- examples for EVERYTHING

core message = 

suprises = not the solution to everything

---

keep stats on blog posts

---
## world models post todo

Scroll bar
https://shaharkadmiel.github.io/Sticky-TOC-Sidebar/

(using the gym env.action_space.sample() or as a Brownian motion (see here). The Brownian motion action sampling is the default. - MISSING A )

- a small helper utility is given in worldmwodels/utils.py:

v2 (next redraft) - 
vae contributions
vae forward pass section
use the first paragraphs of each section in the short post
medium post = how I spent 3k reimplementing a paper
TOC - table contents
marcus Aurelius pic
link tf record files to the data section more
Didn't use spot at all!
Crashing of controller training
tree of project in every section?

what would the cost for a perfectly execuned project (minimum bound) = 200 generations + one vae + one memory

```
The statistics parameterized by the encoder are used to form a distribution over the latent space - a diagonal Gaussian.  

This diagonal Gaussian is a multivariate Gaussian with a diagonal covariance matrix - meaning that each variable is independent.

(is this enforcing a gaussian prior or posterior?)

This parameterized Gaussian is an approximation - using it will limit how expressive our latent space is.

$$z \sim P(z \mid x)$$

$$ z \mid x \approx \mathbf{N} \Big(\mu_{\theta}, \sigma_{\theta}\Big) $$

We can sample from this latent space distribution, making the encoding of an image $x$ stochastic.
```

