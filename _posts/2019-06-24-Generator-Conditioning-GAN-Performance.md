---
layout: post
title:  "Generator Conditioning and GAN Performance"
date:   2019-06-24 20:16:00 +0100
categories: jekyll update
---
These are my notes on the paper [Is Generator Conditioning Causally Related to GAN Performance?](https://arxiv.org/abs/1802.08768). All mistakes are my own. 

## Introduction
- Jacobian of a GAN: 

    $$J_z = \frac{\partial G(\mathbf{z})}{\partial \mathbf{z}}$$
- Metric tensor of a GAN: $M_z = J_z^TJ_z$
- $M_z$ is a Riemannian metric (*).

## Condition number
- Say you move from a point in the latent space $\mathbf{z} \in \mathcal{Z}$ along an eigenvectors of the metric tensor $M_z$ for a GAN.
- If this corresponds to a large eigenvalue then a small step in the latent space will lead to a large difference in the output i.e. $G(\mathbf{z} + \epsilon\mathbf{v}_k)$ for a small $\epsilon$:

    $$lim_{||\epsilon||\rightarrow 0}\frac{||G(\mathbf{z}) - G(\mathbf{z} + \epsilon\mathbf{v}_k)||}{||\epsilon\mathbf{v}_k||} = \sqrt{\lambda_k}$$
- Since hard to study spectrum of $M_z$ instead study condition number: ${\lambda_{\max}}/{\lambda_{\min}}$ - low means well-conditioned, high means poorly conditioned.
- Eigenvalues of $M_z$ are squared singular values of $J_z$.
- Plotting the average log condition number for fixed batch of latent vectors z for different random initialisations results in roughly half-models becoming more poor-conditioned and the other half becoming more well-conditioned.
- Log spectrum of average Jacobian for a batch also varies a lot for a GAN across different random initialisations compared to a VAE.

## Correspondence with metrics
- A high Inception Score and Frechet Inception distance (FID) imply that the GAN is performing well. 
- When the condition number is high, the score is high and the distance is low
- In fact for one run when the condition number is low at first, then goes up after a while, these scores behave in a similar way
- For larger datasets similar correspondence exists but failure modes are more dramatic
- Surprising because the "Inception" metrics are from a pre-trained classifier whilst the condition number depends only on the GAN parameters. 
- They take 360 samples for each of the randomly initialised models and predict the classes for these using the MNIST classifier used to the get the "Inception" metrics.
- It turns out that for well-conditioned GANs the distribution of scores is close to uniform whilst some classes are entirely missing for poorly conditioned ones suggestive of missing modes.

## Jacobian clamping
- Sample a mini-batch from $p_z$
- Make a second mini-batch by adding pertubations of size governed by $\epsilon$ (a hyperparameter)
- Take the ratio of the difference in outputs between the batches to the difference in inputs
- The model is penalised if  the loss does not lie between a pair of hyperparameters $\lambda_{\min}$ and $\lambda_{\max}$
- While this doesn't directly control condition number it uses the idea that for ill-conditioned metric tensors the ratio of differences in outputs to inputs is large. 
- Jacobian clamping improves condition number as well as the other metrics.
- **The fact that clamping improves these other metrics is evidence for a causal relationship between generator conditioning and model performance (as judged by the metrics)**
- Clamping also pushes together the log spectra.
- One notable aspect of the clamping method is how it comes bundled with a method to set hyperparameters:
     - The model is run several times as above with different random initialisations.
     - Then $\lambda_{\min}$ and $\lambda_{\max}$ can be chosen based on the rato $Q$ of the difference between $G(z)$ and $G(z')$ to the difference in inputs $z$ and $z'$ where $z'$ is a perturbed version of $z$ as described above.

## Impact on state of the art models
- They were able to train a conditional GAN with gradient penalty faster (with only 1 rather than 5 discriminator updates per generator update) using Jacobian clamping with only a small drop in the inception score (which might be because no hyperparameter tuning was done).


# (*) What is a Riemannian metric?
- According to Wikipedia it is a family of positive definite inner products of the form:

    $$g_x: T_xM \times T_xM \rightarrow \mathbb{R}, x \in M$$
- The entry also tells us that in a system of local coordinates on $M$ given by $n$ real-valued functions $\mathbf{z}^1, \mathbf{z}^2, \ldots, \mathbf{z}^n$, the vector fields

$$\left \{ \frac{\partial}{\partial z^1}\cdots\frac{\partial}{\partial z^n}\right \}$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;give a basis of tangent vectors at each point $x \in M$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(I have changed the notation slightly from Wikipedia to make it similar to the GAN notation).

- To be a Riemannian metric $g_x$ must satisfy the following properties for all pairs $u, v \in T_xM$:
    - $g(u,v) = g(v,u)$
    - $g(u,u) \geq 0$
    - $g(u,u) = 0$ if and only if $u=0$ 

- Informally we see how this applies to $M_z = J_z^TJ_z$.
- The columns of $J_z$

$$J_z = \left[\frac{\partial G(z)}{\partial z_1} ... \frac{\partial G(z)}{\partial z_{n_z}} \right]$$

- These give a basis of tangent vectors at the point $x = G(z)$
- The elements of the metric tensor are simply dot products of the tangent vectors 

    $$M_{z,ij} = (J_z^TJ_z)_{ij} = \sum_l^{n_x} \frac{\partial G(z)_l}{\partial z_i} \frac{\partial G(z)_l}{\partial z_j} = \frac{\partial G(z)}{\partial z_i} \cdot \frac{\partial G(z)}{\partial z_j} $$

- Since $M_z$ is an $n_z \times n_z$ matrix of real values of it has a real value for every pair in the Cartesian product of the tangent vectors i.e. every $\left(\frac{\partial G(z)}{\partial z_i}, \frac{\partial G(z)}{\partial z_j}\right)$, as required for a Riemannian metric.
- Then because $M_z$  positive definite, we can see it satisfies the properties noted above:
    - $M_{z,ij} = M_{z,ji}$
    - $M_{z,ii} = J_{z,i} \cdot J_{z,i} = \lVert J_{z,i}\rVert^2 \geq 0$
    - As $\lVert J_{z,i}\rVert^2$ is the squared length of the vector it is 0 if and only if $J_{z,i}$ is the zero vector.
    
## Topics

# ML
GAN, Inception score, Frechet Inception distance
# Maths
Jacobian, eigenvalues, eigenvectors, Riemannian metric












        
