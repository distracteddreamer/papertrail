---
layout: post
title:  "Daily ML - Disconnected Manifold Learning for GANs"
date:   2019-08-01 11:51:32 +0100
categories: jekyll update
---
These are my notes on the paper [Disconnected Manifold Learning for Generative Adversarial Networks](https://arxiv.org/pdf/1806.00880). All mistakes are my own. 

# Introduction
Natural images may lie disconnected manifolds - for example, a dataset consisting of faces and beds. Interpolations pairs of points on the face manifold or on the bed manifold may be part of the data distribution but interpolating between a point on the face manifold and the bed manifold will most likely result in an out of distribution image. The distribution will have modes for the manifolds but be close to zero elsewhere. Stable training methods and regularisation typically encourage the generator to cover the whole manifold which will result in out of distribution images. The model can minimise the probability of points from regions that are not part of the data manifold but this may also lead to dropping some of the real manifolds. They also suggest that if the generator can't learn the right support of the real data distribution the equilibrium between the real and fake distributions might not be convergent.

# Model
One approach is to make the latent vector $Z$ disconnected e.g. by using a mixture of Gaussians. Another, which the paper considers, is to use a collection of independent generators $G_c$ with a shared latent space. The generators are encouraged to focus on submanifolds by maximising the mutual information between the generator and its generated data - so that each sample is encouraged to indicate perfectly which generator it came from. This is equivalent to minimising the cross entropy loss between $q(c|x)$ and $p(c)$ where $q(c|x)$ is a learnt approximation for $p(c|x)$. Instead of using a uniform prior, they instead learn $r(c; \zeta)$ minimising the cross entropy between $r(c)$ and the true posterior $p(c|x)$. They add an entropy regularisation term to prevent the probabilities for some generators going down to 0 at the start. To generate an image a generator is first sampled from r(c) and an image is sampled from the generator.

# Results
They measure the following:

1. Inter-class variation via the Jensen Shannon Divergence to compare the distributions over predicted classes from a pre-trained classifier for real and fake data to determine how well the distrbution over classes are captured. Smaller is better.

2. Intra-class variation for points from the same predicted class,using Euclidean distance in a small neighbourhood of each point to generate a graph that approximates the true image manifold and find the average of the shortest path distance for each pair of points. Larger is better (suggesting that the model is producing a wide range of samples from the class). 

3. They also evaluate sample quality by plotting the ratio of samples with classifier scores higher than some threshold for different thresholds. Larger ratio is better (indicating the classifier's confidence about the sample which in turn suggests it is of a good quality). 

They do experiments with MNIST and a Face-Bedroom datase consisting of equal numbers of images of faces and bedrooms which could reasonably be considered to lie on different manifolds. For both datasets the model performs better compared to other GANs on the metrics considered and also achieves lower Frechet Inception Distance (smaller is better). 