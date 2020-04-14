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

## Connectivity
- A connected region is one where any two points within the region can be connected by a path that lies entirely within that region.
- Since for a convex region all the points in the straight line segment between any two points in that region are also in that region, a convex region is connected. 
- However a connected region is not necessarily convex since for example even in the non-simply connected region below a path exists from any point to any other point than does not leave the region (but is not necessarily a straight line).


    ![png]({{ site.baseurl }}/assets/Convex-Optimisation/connectivity.jpg)



