---
layout: post
title:  Norm Cones
date:   2020-05-14 17:04:57
categories: jekyll update
---



```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

**Definition**

Given a norm $\lVert\cdot\rVert$, a centre point $\mathbf{x_c} \in \mathbb{R}^n$ a **norm cone** is defined as

$$\left\{\mathbf{x} \in \mathbb{R}^n, t \in \mathbb{R}: 
\lVert\mathbf{x} - \mathbf{x}_c \rVert < t\right\}$$

Below we shall plot some norm cones for various norms, for $\mathbf{x} \in R^2$. They are all zero-centred i.e. $\mathbf{x_c} = (0, 0)$.


```python
def plot_norm_cone(norm_p, points, colour):
    fig = plt.figure(figsize=(12, 12))
    ax =fig.add_subplot(111, projection='3d')
    
    n_str = '\infty' if norm_p==np.inf else norm_p
    
    if norm_p==np.inf:
        n_exp = "max(|x|, |y|)"
    elif norm_p==1:
        n_exp = "|x| + |y|"
    else:
        n_exp = "(|x|_%s + |y|_%s)^{1/%s}" % ((n_str,)*3)
    
    px, py = np.meshgrid(*(points,) * 2)
    pz = np.linalg.norm(np.stack([px, py], axis=-1), norm_p, axis=-1)
    ax.plot_surface(
        px,
        py,
        pz,
        color=colour,
        alpha=0.6
    )
    
    px, py, pz = np.meshgrid(*(np.linspace(np.min(px), np.max(px), 11),)*2,
                             np.linspace(np.min(pz), np.max(pz), 11)
                            )
    
    norm = np.linalg.norm(np.stack([px, py], axis=-1), norm_p, axis=-1)
    
    
    m1 = (norm < pz)
    m2 = (norm > pz)
    
    ax.scatter(
        px[m1].ravel(), 
        py[m1].ravel(),
        pz[m1].ravel(),
        alpha=1.0,
        label='$%s < t$'%(n_exp)
    )
    
    ax.scatter(
        px[m2].ravel(), 
        py[m2].ravel(),
        pz[m2].ravel(),
        alpha=1.0,
        label='$%s > t$'%(n_exp)
    )
    fkw = dict(fontsize=12)
    ax.set_title('$L_%s$ norm cone'%(n_str), **fkw)
    ax.set_xlabel('$x$', **fkw)
    ax.set_ylabel('$y$', **fkw)
    ax.set_zlabel('$t$', **fkw)
    ax.legend();
```


```python
points = np.linspace(-0.5, 0.5, 101)
plot_norm_cone(1, points, 'slateblue')
```


![png]({{ site.baseurl }}/assets/NormCones/output_3_0.png)



```python
points = np.linspace(-0.5, 0.5, 101)
plot_norm_cone(2, points, 'gold')
```


![png]({{ site.baseurl }}/assets/NormCones/output_4_0.png)



```python
points = np.linspace(-0.5, 0.5, 101)
plot_norm_cone(3, points, 'darkgreen')
```


![png]({{ site.baseurl }}/assets/NormCones/output_5_0.png)



```python
points = np.linspace(-0.5, 0.5, 101)
plot_norm_cone(np.inf, points, 'firebrick')
```


![png]({{ site.baseurl }}/assets/NormCones/output_6_0.png)

