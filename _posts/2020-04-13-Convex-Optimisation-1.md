---
layout: post
title:  "[WIP] Convex Optimisation Week 1"
date:   2020-04-14 12:00:00
categories: jekyll update
---

Notes from my study of [Stanford's EE364a: Convex Optimization I](http://web.stanford.edu/class/ee364a/). This is not intended to be contain a summary of the topics but thoughts and notes related to the concepts in the course.

## Cones
- A **cone** is any set such that for a given vector $x$ in that set, all vectors pointing in the same direction as $x$ (but not necessarily the opposite direction), whatever their length, will be in that set. 
- A set $C$ is a **convex cone** if both convex and a cone meaning that for $x_1, \ldots x_N \in C$, $\theta_1, \ldots ,\theta_N \ge 0$, 
$\sum_i \theta_i x_i \in C$, which can be shown as follows:

    $$\sum_i \theta_i x_i = \frac{\sum_j \theta_j}{\sum_j \theta_j}\sum_i \theta_i x_i  
    = \sum_i \frac{\theta_i}{\sum_j \theta_j}\left(\sum_j \theta_j\cdot x_i\right)
    = \sum_i\alpha_i(t\cdot x_i)$$

- Since all $\theta_j \geq 0$, their sum $t \geq 0$ meaning that $t\cdot x_i \in C$ and as $\sum_i  \alpha_i = 1$, the sum is a convex combination of vectors in $C$.  

