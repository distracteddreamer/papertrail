---
layout: post
title:  "[WIP] Gauge Equivariant Convolutions"
date:   2020-04-11 17:00:53
categories: jekyll update
---


These are my notes on the papers [Gauge Equivariant Convolutional Networks and the Icosahedral CNN](https://arxiv.org/pdf/1902.04615v3.pdf) [1] and [Gauge Equivariant Mesh CNNs](https://arxiv.org/pdf/2003.05425v1.pdf) [2]. All mistakes are my own. I must particularly emphasise this caveat here since at this point I am still trying to grapple with the ideas in these papers and my interpretation of them is likely to be flawed. 

### What is a gauge?

- Informally we can think of it is as a mesh
- Formally it is defined as follows in [1]:

> [A] position-dependent invertible linear map $w_p: \mathbb{R}^d \rightarrow T_p M$, 
where $T_p$ is the tangent space of $M$ at $p$

- Informally we can break this down as follows:

  - Let M be a manifold in  $\mathcal{R}^d$ and let $p$ be a point on this manifold.
  - Let $T_pM$ be the space of tangent vectors to $M$ at the point $p$.
  - Further let $e_1 \ldots e_d$ be the standard basis vectors of $\mathbb{R}^d$
  - For each point $p$ a gauge consists of a set of basis vectors that are tangential to the $M$ at p.
  - These can be written via the mapping $w_p$ which maps each of $e_1 \ldots e_d$ to
  $w_p(e_1) \ldots w_p(e_d)$

