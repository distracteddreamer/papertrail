---
layout: post
title:  Norm Cones
date:   2020-05-14 12:00:58
categories: jekyll update
---



```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

# Norm cones


```python
fig = plt.figure(figsize=(8, 8))
points = np.linspace(-0.5, 0.6, 101)
px, py = [i.ravel() for i in np.meshgrid(*(points,)*2)]
x = np.stack([px, py], axis=-1)  #(N, 2)
tt = np.linspace(0, 0.5, 11) #(T,)

norm_l1 = np.abs(x).sum(-1) #(N,)

for t in tt[::-1]:
    plt.scatter(*x[norm_l1 < t].T)
plt.title('L1 norm cone');
```


![png]({{ site.baseurl }}/assets/NormCones/output_2_0.png)



```python
fig = plt.figure(figsize=(8, 8))
points = np.linspace(-0.5, 0.6, 101)
px, py = [i.ravel() for i in np.meshgrid(*(points,)*2)]
x = np.stack([px, py], axis=-1)  #(N, 2)
tt = np.linspace(0, 0.5, 11) #(T,)

norm_l2 = np.sqrt((x**2).sum(-1)) #(N,)

for t in tt[::-1]:
    plt.scatter(*x[norm_l2 < t].T)
plt.title('L2 norm cone');
```


![png]({{ site.baseurl }}/assets/NormCones/output_3_0.png)



```python
fig = plt.figure(figsize=(8, 8))
points = np.linspace(-0.5, 0.6, 101)
px, py = [i.ravel() for i in np.meshgrid(*(points,)*2)]
x = np.stack([px, py], axis=-1)  #(N, 2)
tt = np.linspace(-1, 1, 21) #(T,)

norm_l3 = np.cbrt((x**3).sum(-1)) #(N,)

for t in tt[::-1]:
    plt.scatter(*x[norm_l3 < t].T)
plt.title('L3 norm cone;')
```




    Text(0.5, 1.0, 'L3 norm cone;')




![png]({{ site.baseurl }}/assets/NormCones/output_4_1.png)



```python
fig = plt.figure(figsize=(8, 8))
points = np.linspace(-0.5, 0.6, 101)
px, py = [i.ravel() for i in np.meshgrid(*(points,)*2)]
x = np.stack([px, py], axis=-1)  #(N, 2)
tt = np.linspace(0, 0.5, 11) #(T,)

norm_l4 = np.power((x**4).sum(-1), 1/4) #(N,)

for t in tt[::-1]:
    plt.scatter(*x[norm_l4 < t].T)
plt.title('L4 norm cone;')
```




    Text(0.5, 1.0, 'L4 norm cone;')




![png]({{ site.baseurl }}/assets/NormCones/output_5_1.png)



```python
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
points = np.linspace(-0.5, 0.6, 101)
px, py = [i.ravel() for i in np.meshgrid(*(points,)*2)]
x = np.stack([px, py], axis=-1)  #(N, 2)
tt =[0, 1, 2, 3] #(T,)

norm_l0 = (x>=0).sum(-1) #(N,)

for ax, t in zip(axes.ravel(), tt):
    ax.scatter(*x[norm_l0 < t].T)
    ax.set_xlim([-0.5, 0.6])
    ax.set_ylim([-0.5, 0.6])
    ax.set_title("||x||_0 < {}".format(t))
plt.suptitle('L0 norm cone;');
```


![png]({{ site.baseurl }}/assets/NormCones/output_6_0.png)



```python
fig = plt.figure(figsize=(8, 8))
points = np.linspace(-0.5, 0.6, 101)
px, py = [i.ravel() for i in np.meshgrid(*(points,)*2)]
x = np.stack([px, py], axis=-1)  #(N, 2)
tt =np.linspace(-0.5, 0.6, 11) #(T,)

norm_linf = x.max(-1) #(N,)

for t in tt[::-1]:
    plt.scatter(*x[norm_linf < t].T)
    
plt.title('L$\infty$ norm cone;');
```


![png]({{ site.baseurl }}/assets/NormCones/output_7_0.png)

