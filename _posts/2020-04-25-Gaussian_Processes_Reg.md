---
layout: post
title:  Gaussian Processes for Regression
date:   2020-04-25 20:08:07
categories: jekyll update
---


In this notebook I am recreating the simple Gaussian Process example described in this tutorial. The results are not identical since the tutorial provides the values only for the inputs $\mathbf{x}$ and not for the outputs $\mathbf{y}$ for which I read off approximate values from the plot in Figure 1 of the tutorial.


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
np.random.seed(1)
```

We have the following $n = 6$ observations


```python
x = np.array([-1.5, -1, -.75, -.4, -.25, 0])
y = np.array([-1.62, -1.09, -0.3, 0.225, 0.55, 0.82])
noise = 0.3
plt.figure(figsize=(12, 8))
plt.errorbar(x, y, noise, fmt='ok', ecolor='r', elinewidth=1, capsize=3);
plt.title('6 noisy observations', fontsize=14);
```


![png]({{ site.baseurl }}/assets/Gaussian_Processes_Reg/output_4_0.png)


We assume that these observations come from an $n$-dimensional normal distribution as follows:

- There is an underlying function $\mathbf{f}(x) \sim \mathcal{N}(\mathbf{0}, K)$
- The observations $\mathbf{y}$ have additional noise associated with them such that

$$\mathbf{y} = \mathbf{f}(x) + \epsilon\\
\epsilon  \sim \mathcal{N}(\mathbf{0}, K)\\
\mathbf{y} \sim \mathcal{N}(\mathbf{0}, K + \sigma_n^2)
$$

The important part is the covariance matrix $K$ which is derived from the data and where each element $K_{vw}$ is defined via a **kernel** function $k(x_v, x_w)$. The function $k$ is based on our knowledge about the data distribution. Here we will use the squared exponential:

$$k(x, x') = \sigma_f^2\exp\left[\frac{-(x - x')^2}{2l^2}\right]$$

where $\sigma_f$ and $l$ are hyperparameters which we collectively denote $\theta$. Sometimes we might have prior knowledge about these. Actually $\sigma_n$ is also a hyperparameter but here we have been given a value for it of $\sigma_n = 0.3$. In the absence of such knowledge, we can estimate it e.g. via MAP estimate i.e.

$$\theta = \arg\max_{\theta} p(\theta \vert \mathbf{x}, \mathbf{y})\\
= \arg\max_{\theta} \frac{p(\mathbf{y} \vert \theta, \mathbf{x}) p(\theta \vert \mathbf{x})}{p(\mathbf{y} \vert \mathbf{x})}$$

If we don't have any prior knowledge about how $\theta$ should be, implying that $p(\theta \vert \mathbf{x})$ is uniform thus $p(\theta \vert \mathbf{x}) = \text{constant}$. Then since $p(\mathbf{y} \vert \mathbf{x})$ doesn't depend on $\theta$ we get

$$\theta = \arg\max_{\theta} p(\mathbf{y} \vert \theta, \mathbf{x}) = \arg\max_{\theta} \log p(\mathbf{y} \vert \theta, \mathbf{x})$$

Although we didn't explicitly specify the dependence on $\theta$ above, actually $p(\mathbf{y} \vert \theta, \mathbf{x}) = \mathcal{N}(\mathbf{0}, K + \sigma_n^2)$ which means 

$$ \log p(\mathbf{y} \vert \theta, \mathbf{x}) = 
-\frac{1}{2}\left(\log\left\lvert K + \sigma_n^2 \right\rvert  + \mathbf{y}^T\left(K + \sigma_n^2\right)^{-1}\mathbf{y}\right) + A
$$

where $A$ comprises terms that don't depend on $\theta$. Thus

$$\theta = \arg\max_{\theta}\left(-\log\left\lvert K + \sigma_n^2 \right\rvert  - \mathbf{y}^T\left(K + \sigma_n^2\right)^{-1}\mathbf{y}\right) = \arg\min_{\theta}\left(\log\left\lvert K + \sigma_n^2 \right\rvert  + \mathbf{y}^T\left(K + \sigma_n^2\right)^{-1}\mathbf{y}\right)$$


```python
from scipy.optimize import minimize
```


```python
def sq_exp_kernel(theta, x1, x2=None):
    if x2 is None:
        x2 = x1
    l_sq, sigma_f_sq = theta
    sq_diff = (x1[:, None] - x2) ** 2
    exp = np.exp(-sq_diff / (2 * l_sq))
    return (sigma_f_sq) * exp

def loss(theta, x, y, var_n):
    K = sq_exp_kernel(theta, x)
    K_noise = K + var_n * np.eye(len(x))
    return np.log(np.linalg.det(K_noise)) + (y.T@np.linalg.inv(K_noise))@y  
```


```python
t0 = [1., 1.]
res = minimize(loss, t0, args=(x, y, noise ** 2), method='Nelder-Mead', tol=1e-10)
l, sigma_f = np.sqrt(res.x)
l, sigma_f
```




    (0.9973985356503553, 1.2696785734036715)




```python
(sq_exp_kernel(res.x, x) + (noise ** 2) * np.eye(6)).round(2)
```




    array([[1.7 , 1.42, 1.22, 0.88, 0.74, 0.52],
           [1.42, 1.7 , 1.56, 1.35, 1.22, 0.98],
           [1.22, 1.56, 1.7 , 1.52, 1.42, 1.22],
           [0.88, 1.35, 1.52, 1.7 , 1.59, 1.49],
           [0.74, 1.22, 1.42, 1.59, 1.7 , 1.56],
           [0.52, 0.98, 1.22, 1.49, 1.56, 1.7 ]])



Suppose $m$ new input datapoints $\mathbf{x}^*$ are introduced, then we get a $(n + m)$ dimensional normal distribution whose noise variance is the same but whose other covariance matrix becomes:

$$K' = \begin{bmatrix}
K & K^{*T}\\
K^* & K^{**}
\end{bmatrix} \\
K^*_{uv} = k(x_u, x_v),\text{ }x_u \in \mathbf{x}, x_v \in \mathbf{x}^* \\
K^{**}_{uv} = k(x_u, x_v),\text{ }x_u, x_v \in \mathbf{x}^*
$$

where we abuse the element of $\in$ notation a bit to indicate that an element belongs to a vector. Basically constructing the matrix in this manner lets us reuse the previously computed $K$ instead of computing it again. 

Although we can write down the joint distribution for $\mathbf{y}$ and $\mathbf{y^*}$ as follows

$$\begin{bmatrix}
\mathbf{y}\\
\mathbf{y}^*
\end{bmatrix}  = 
\begin{bmatrix}
K & K^{*T}\\
K^* & K^{**}
\end{bmatrix}$$


we are actually interested in the conditional distribution. Letting $S' = K' + \sigma_nI$, with the $S, S^{*}, S^{**}$ the blocks of $S'$ corresponding to $K, K^{*}, K^{**}$:

$$p(\mathbf{y}^* | \mathbf{y}) = \mathcal{N}\left(S^*S^{-1}\mathbf{y}, S^{**} - S^{*}S^{-1}S^{*T}\right)$$


because using our data we would like to predict the output for a given input.


```python
x_star = np.array([0.2])
K_star = sq_exp_kernel(res.x, x_star, x)
K_star_star = sq_exp_kernel(res.x, x_star)
K = sq_exp_kernel(res.x, x)

K_p = np.concatenate(
    [np.concatenate([K, K_star.T], axis=-1),
     np.concatenate([K_star, K_star_star], axis=-1)]
)

S_p = K_p + (noise ** 2) * np.eye(K_p.shape[0])

S = S_p[:K.shape[0], :K.shape[1]]
S_star = S_p[K.shape[0]:, :K.shape[1]]
S_star_star = S_p[K.shape[0]:, K.shape[1]:]
assert np.allclose(S_star, S_p[:K.shape[0], K.shape[1]:].T)
```


```python
K_p.round(2)
```




    array([[1.61, 1.42, 1.22, 0.88, 0.74, 0.52, 0.38],
           [1.42, 1.61, 1.56, 1.35, 1.22, 0.98, 0.78],
           [1.22, 1.56, 1.61, 1.52, 1.42, 1.22, 1.02],
           [0.88, 1.35, 1.52, 1.61, 1.59, 1.49, 1.35],
           [0.74, 1.22, 1.42, 1.59, 1.61, 1.56, 1.46],
           [0.52, 0.98, 1.22, 1.49, 1.56, 1.61, 1.58],
           [0.38, 0.78, 1.02, 1.35, 1.46, 1.58, 1.61]])




```python
S_p.round(2)
```




    array([[1.7 , 1.42, 1.22, 0.88, 0.74, 0.52, 0.38],
           [1.42, 1.7 , 1.56, 1.35, 1.22, 0.98, 0.78],
           [1.22, 1.56, 1.7 , 1.52, 1.42, 1.22, 1.02],
           [0.88, 1.35, 1.52, 1.7 , 1.59, 1.49, 1.35],
           [0.74, 1.22, 1.42, 1.59, 1.7 , 1.56, 1.46],
           [0.52, 0.98, 1.22, 1.49, 1.56, 1.7 , 1.58],
           [0.38, 0.78, 1.02, 1.35, 1.46, 1.58, 1.7 ]])




```python
W = (S_star@np.linalg.inv(S))
y_star_hat = W@y
y_star_var = S_star_star - W@S_star.T
y_star_hat, y_star_var
```




    (array([0.92699289]), array([[0.20631961]]))




```python
def predict(x_star, noise, theta):
    K_star = sq_exp_kernel(theta, x_star, x)
    K_star_star = sq_exp_kernel(theta, x_star)
    K = sq_exp_kernel(theta, x)

    K_p = np.concatenate(
        [np.concatenate([K, K_star.T], axis=-1),
         np.concatenate([K_star, K_star_star], axis=-1)]
    )

    S_p = K_p + (noise ** 2) * np.eye(K_p.shape[0])

    S = S_p[:K.shape[0], :K.shape[1]]
    S_star = S_p[K.shape[0]:, :K.shape[1]]
    S_star_star = S_p[K.shape[0]:, K.shape[1]:]
    
    W = (S_star@np.linalg.inv(S))
    y_star_hat = W@y
    y_star_var = S_star_star - W@S_star.T
    
    return y_star_hat, y_star_var
```

Now we will repeat the above for 1000 different points. First we will just input each point individually and get a prediction.


```python
points = np.array(sorted(np.random.uniform(-1.6, 0.3, size=(1000,))))
pred_mean, pred_var = zip(*[predict(np.array([p]), noise, res.x) for p in 
                           points])
pred_mean = np.stack(pred_mean).squeeze()
pred_var = np.stack(pred_var).squeeze()
interval = 1.96 * np.sqrt(pred_var)
```


```python
# Use 95% confidence interval
plt.figure(figsize=(12, 8))
plt.errorbar(x, y, noise, fmt='ko', ecolor='r', elinewidth=1, capsize=3);
plt.errorbar(x_star[0], y_star_hat[0], 1.96 * np.sqrt(y_star_var[0]), fmt='bo', ecolor='lightgreen')
plt.plot(points, pred_mean, c='k')
plt.fill_between(points, pred_mean - interval, pred_mean + interval, color='red', alpha=0.2)
plt.title('6 noisy observations along with new points with pointwise 95% confidence intervals',
         fontsize=14);

```


![png]({{ site.baseurl }}/assets/Gaussian_Processes_Reg/output_21_0.png)


Now let us predict in one go.


```python
pred_mean_all, pred_var_all = predict(points, noise, res.x)
```

This time we can't use confidence intervals we have multivariate conditional where the covariance matrix is not diagonal as we can see below


```python
plt.figure(figsize=(8, 8))
plt.imshow(pred_var_all, cmap='hot', vmin=pred_var_all.min(), vmax=pred_var_all.max());
plt.axis('off');
```


![png]({{ site.baseurl }}/assets/Gaussian_Processes_Reg/output_25_0.png)


Instead we will sample a number of 1000-d vectors and plot those


```python
samples = np.random.multivariate_normal(mean=pred_mean_all, cov=pred_var_all, size=(100,))
```

As we can see in the plot above, the diagonal predominates over other values and below we notice that the shape of the plots of the samples roughly follows the envelope of the confidence intervals.


```python
plt.figure(figsize=(12, 8))
plt.plot(points, samples.T, c='lightcoral', alpha=0.2);
plt.plot(points[::10], samples[0][::10], c='b');
plt.plot(x, y, 'ko');
plt.title('6 noisy observations along with samples for new points',
         fontsize=14);
```


![png]({{ site.baseurl }}/assets/Gaussian_Processes_Reg/output_29_0.png)

