---
layout: post
title:  "Bellman equations example"
date:   2019-09-02 11:51:32 +0100
categories: jekyll update
---

 # Bellman equations in vectorised form

In this post we will go through the steps for example 3.8 in Sutton and Barto. 

### Theory

For Markov decision processes the value of a state $s$ under policy $\pi$ is given by

$$v_\pi(s) = \mathbb{E}_\pi[G_t \lvert S_t=s]=\mathbb{E}_\pi\left[\sum\limits_{k=0}^{\infty}\gamma^kR_{t + k + 1} \Bigg\vert S_t = s \right] \forall{s
}\in \mathcal{S}\\
$$

$$\mathbb{E}_\pi[R_{t+1}\lvert S_t=s]
= \sum_{r} r\cdot p(R_{t+1} = r \vert S_t = s)\\
= \sum_{s', r, a} r \cdot p(R_{t+1} = r, S_{t+1} = s' \vert A_t = a, S_t = s)\cdot \pi(A_t = a\vert S_t = s)$$

$$\mathbb{E}_\pi[G_{t+1}\lvert S_t=s]
= \sum_{s', a} \mathbb{E}_\pi[G_{t+1} \lvert S_{t+1}=s', S_t = s] \cdot
p(S_{t+1} = s' | S_t = s, A_t = a)\cdot \pi(A_t = a\vert S_t = s) \\= \sum_{s', a} \mathbb{E}_\pi[G_{t+1} \lvert S_{t+1}=s']\cdot p(S_{t+1} = s' | S_t = s, A_t = a)\cdot \pi(A_t = a\vert S_t = s)$$

(since given $S_{t+1}$, $G_{t+1}$ does not depend on $S_t$)

$$=\sum_{s', r, a} v_\pi(s')\cdot p(R_{t+1} = r, S_{t+1}=s' \vert S_t = s, A_t = a)\cdot \pi(A_t = a\vert S_t = s)$$

Thus we can write

$$v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \lvert S_t=s]\\
= \sum_a \pi(a\vert s) \sum_{s'}\sum_r p(s', r \vert s, a)\left[r + \gamma v_\pi(s')\right]$$

### Vectorised form

For $N$ discrete reward values, $\{r_0, r_1, \ldots, r_N\}$, define let be $\mathbf{r}$ a row vector where $\mathbf{r}_i = r_i$. Then assuming that $s = \{0, 1, ... ,\lvert\mathcal{S}\rvert-1\}$ (which can be achieved by mapping each state to an index):

$$\sum_{a, s', r} r\cdot \pi(a\vert s) \sum_r p(s', r \vert s, a)= \sum_i r_i \sum_{a, s'} \pi(a\vert s) \cdot p(s', r_i \vert s, a) = (\mathbf{r}\mathbf{Q})_s$$

$$\mathbf{Q}_{is} = \sum_{a, s'} p(s', r_i \vert s, a) \cdot \pi(a\vert s)$$


Similarly, with $\mathbf{v}$ as a row vector where $\mathbf{v}_{s'} = v_\pi(s')$:

$$\sum_a \pi(a\vert s) \sum_{s'}\sum_r \gamma v_\pi(s')\cdot p(s', r \vert s, a)
= \gamma\sum_{s'} v_\pi(s')\sum_{a, i} p(s', r_i \vert s, a) \cdot \pi(a\vert s) = \gamma(\mathbf{v} \mathbf{R})_s$$

$$\mathbf{R}_{s's} = \sum_{a, i} p(s', r_i \vert s, a) \cdot \pi(a\vert s)$$

Putting this together

$$\mathbf{v} = \mathbf{r}\mathbf{Q} + \gamma\mathbf{v} \mathbf{R}$$

Solving for $\mathbf{v}$:

$$\mathbf{v}(\mathbf{I} - \gamma\mathbf{R}) = \mathbf{r}\mathbf{Q} \implies \mathbf{v} = \mathbf{r}\mathbf{Q}(\mathbf{I} - \gamma\mathbf{R})^{-1}$$

### Simple gridworld application


```python
import numpy as np
import matplotlib.pyplot as plt
import sys
```

The states are the cells of a 5 by 5 grid.


```python
h = w = 5
states = np.stack(np.meshgrid(np.arange(h), np.arange(w))[::-1], axis=-1).reshape((25,2))
state2ind = dict(zip(map(tuple, states), range(len(states))))
```

The actions are move one cell up, down, left or right and each has equal probability for all states. Hence $\pi(a \vert s)$ can be represented by a $4 \times 25$ matrix where every element is 0.25.


```python
actions = np.array([[1,0],[0,1],[-1,0],[0,-1]])
pi = np.tile([[0.25]], [4, 25])
```

There are two special states $A$ and $B$ where every action leads to a jump to the next states $A'$ and $B'$. For every other state all actions lead to the neighbouring state that depends on the action - unless this action would lead to a state outside the grid in which case the next state is the same as the present state.


```python
state_a = np.array([0,1])
state_b = np.array([0,3])
next_a = np.array([4,1])
next_b = np.array([2,3])
```

For next states $A'$ the reward is 10 and for $B'$ it is $5$. For all other next states rewards are $0$ unless the action leads to a state outside the grid in which case the reward is $-1$ 


```python
rewards = np.array([-1,0,5,10])
rewards2ind = dict(zip(rewards, range(len(rewards))))
```


```python
def out_of_bounds(state, h, w):
    return np.logical_or(np.any(state >= [h,w], axis=-1), 
                          np.any(state < [0,0], axis=-1))
```


```python
out_of_bounds(states, h, w)
```




    array([False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False])



We will now create a $4D$ tensor for $p(s',r \vert s, a)$ i.e. at tensor with dimensions $\lvert\mathcal{S}\rvert \times \lvert\mathcal{R}\rvert \times \lvert\mathcal{S}\rvert \times
\lvert\mathcal{A}\rvert$.


```python
P = np.zeros((states.shape[0], rewards.shape[0], states.shape[0], actions.shape[0]))
```


```python
for s,state in enumerate(states):
    
    for a,action in enumerate(actions):
        if np.all(state == state_a):
            next_state = next_a
            reward = 10
        elif np.all(state == state_b):
            next_state = next_b
            reward = 5
        else:
            next_state = state+action
            reward = 0
            
        #Determine if next state is outside the grid boundaries
        if out_of_bounds(next_state, h, w):
            next_state = state
            reward = -1
        
        P[state2ind[tuple(next_state)], rewards2ind[reward], s, a] = 1
```

For both $\mathbf{Q}$ and $\mathbf{R}$ we need to find 

$$\sum_{a} p(s', r_i \vert s, a) \cdot \pi(a\vert s)$$


```python
M = np.sum(P*pi.T, axis=-1)
```


```python
Q = np.sum(M, axis=1)
```


```python
R = np.sum(M, axis=0)
```


```python
gamma = 0.9
```


```python
A = np.eye(Q.shape[0]) - gamma*Q
```


```python
y = np.dot(rewards, R)
```


```python
v = np.dot(y, np.linalg.inv(A))
```


```python
def make_grid(v, shape):
    return v.reshape(shape)
```


```python
def plot_grid(states, values, offx, offy, th):
    for state, value in zip(states, values):
        plt.text(state[1]+offx,state[0]+offy,np.round(value,1), 
                 color='white' if value < th else 'k')
```


```python
v_grid = make_grid(np.round(v,1),[5,5])
```

`v_grid` should be the same as in the diagram shown in the book:



```python
v_true = np.array([[ 3.3,  8.8,  4.4,  5.3,  1.5],
                   [ 1.5,  3. ,  2.3,  1.9,  0.5],
                   [ 0.1,  0.7,  0.7,  0.4, -0.4],
                   [-1. , -0.4, -0.4, -0.6, -1.2],
                   [-1.9, -1.3, -1.2, -1.4, -2. ]])
```


```python
def values_match(v_true, v):
    return np.all(v_true.reshape(-1)==np.round(v,1))
```


```python
print(values_match(v_true, v))
```

    True



```python
plt.imshow(v_grid)
plt.title('State-value function')
plt.colorbar();
plt.axis('off');
plot_grid(states, v, -0.25, 0, 8)
```


![png]({{ site.baseurl }}/assets/Bellman-Example/Bellman-Example_36_0.png)


# Optimal value function

### Bellman optimality equation
$$v_*(s) = \max_a \sum_{s',r}p(s', r \lvert s, a)\left[r + \gamma v_*(s')\right]$$

Below $\mathbf{P}$ is the $4D$ tensor defined above where $\mathbf{P}_{s'isa} = p(s',r_i\lvert s, a)$:

$$\mathbf{v}_* = \max_a \sum_{s'}\mathbf{r}\mathbf{P}_{s' : s a} + \gamma\sum_{i}\mathbf{v}_*\mathbf{P}_{: i s a}$$

This is not a linear equation so we can't solve it as before but we can estimate the solution iteratively. We will initialise $\mathbf{v}$ in different ways and keep updating it with the max over all actions $a$ of $\sum_{s'}\mathbf{r}\mathbf{P}_{s' : s a} + \gamma\sum_{i}\mathbf{v}_*\mathbf{P}_{: i s a}$ until the difference between consecutive value functions becomes negligible.

We will define a function to use later that will plot actions as arrows.


```python
def plot_actions(ox,oy,length,actions,clr='k',**kwargs):
    for action in actions:
        plt.arrow(ox,oy,length*action[0], length*action[1],color=clr,**kwargs)
```

#### Initialisation with $v_\pi(s)$ for uniform $\pi(a\lvert s)$

Here we will use the value of $\mathbf{v}$ found analytically above.


```python
def estimate_vstar(v_init, prob_next_reward, rewards, gamma = 0.9, tol=1e-12):
    iters = 0
    v_prev = v_init
    vs = [v_prev]
    diffs = []
    while True:
        iters+=1
        QQ = np.sum(np.einsum('j,ijkl->ikl',rewards, prob_next_reward), axis=0) 
        RR = np.sum(np.einsum('i,ijkl->jkl',v_prev, prob_next_reward), axis=0)
        v_next = np.max(QQ + gamma*RR, axis=-1)
        diff = np.square(v_prev-v_next)
        diffs.append(np.mean(diff))
        sys.stdout.write('\rIteration {}, mean squared difference {:.7f}'.format(iters, diffs[-1]))
        vs.append(v_next)
        if np.all(diff < tol):
            break
        v_prev = v_next 
    return v_next, vs, diffs, QQ, RR
```


```python
vstar_, vs, diffs, QQ, RR = estimate_vstar(v, P, rewards)
```

    Iteration 148, mean squared difference 0.0000000


```python
def plot_curves(vs, diffs):
    plt.figure(figsize=(12,4))
    plt.plot(vs,'-o',markersize=1);
    plt.title('$v_t(s)$ for all states across iterations')
    
    plt.figure(figsize=(12,4))
    plt.plot(diffs,'-o');
    plt.title('Mean squared difference $|v_t(s)-v_{t-1}(s)|^2$ for all states across iterations');
```


```python
plot_curves(vs, diffs)
```


![png]({{ site.baseurl }}/assets/Bellman-Example/Bellman-Example_46_0.png)



![png]({{ site.baseurl }}/assets/Bellman-Example/Bellman-Example_46_1.png)



```python
def plot_first_25(vs, states, th=8):
    vs_to_plot = vs[:25]
    vmin = np.min(vs_to_plot)
    vmax = np.max(vs_to_plot)
    plt.figure(figsize=(15,15))
    for i, vi in enumerate(vs_to_plot):
        plt.subplot(5,5, i+1)
        plt.imshow(make_grid(vi, [5,5]), vmin=vmin, vmax=vmax)
        plt.title('$v_\{\%i\}(s)$'\%i)
        plt.axis('off');
        plot_grid(states, vi, -0.25, 0, th)
```


```python
plot_first_25(vs, states)
```


![png]({{ site.baseurl }}/assets/Bellman-Example/Bellman-Example_48_0.png)



```python
def get_best_actions(QQ, RR, gamma=0.9):
    v_next_all = QQ + gamma*RR
    actions_best= []
    for vals in v_next_all:
        best_inds = np.where(vals==np.max(vals))
        actions_best.append(actions[best_inds])
    return actions_best
```


```python
actions_best = get_best_actions(QQ, RR, gamma=0.9)
```

**A note on the coordinate system**

Again we can confirm that out results match the ones shown in the book. Note we are using the $ij$ matrix based axis rather than the usual $xy$ coordinate axis. That means when we move `up` in $xy$-coordinates, this corresponds to a `down` for the matrix since the row number $i$ decreases. Although `left` and `right` are equivalent, whilst moving `left` or `right` in $xy$ decreases or increases the $x$ coordinate, here the column $j$ decreases or increases. So for plotting our states and actions will be reversed. In order to facilitate comparision of the best actions to those shown in the book, we will reverse the action vectors and use `down` where the diagram in the book has `up`. 


```python
#Directions in xy world 
right_ = np.array([[1,0]])
left_ = np.array([[-1,0]])
down_ = np.array([[0,-1]])
up_ = np.array([[0,1]])

#Directions in matrix world 
right = up_
left = down_
down = right_
up = left_

right_up = np.concatenate([right, up])
up_left = np.concatenate([up, left])

actions_best_true = [right, actions, left, actions, left,
                     right_up, up, up_left, left, left,
                     right_up, up, up_left, up_left, up_left,
                     right_up, up, up_left, up_left, up_left,
                     right_up, up, up_left, up_left, up_left]

```


```python
def actions_match(actions_true, actions):
    return (np.all(list(map(np.all, map(np.equal, actions_best, actions_best_true)))))
```


```python
actions_match(actions_best_true, actions_best)
```




    True



Confirm `v_next` is the same:


```python
vstar = np.array([[22. , 24.4, 22. , 19.4, 17.5],
                        [19.8, 22. , 19.8, 17.8, 16. ],
                        [17.8, 19.8, 17.8, 16. , 14.4],
                        [16. , 17.8, 16. , 14.4, 13. ],
                        [14.4, 16. , 14.4, 13. , 11.7]])

values_match(vstar, vstar_)
```




    True




```python
def plot_init_final(vs, states, actions_best):
    v_init_final = [vs[0], vs[-1]]
    vmin = np.min(v_init_final)
    vmax = np.max(v_init_final)
    plt.figure(figsize=(16,6))
    for i,(vi,title) in enumerate(zip(v_init_final, ['$v_0(s)$', '$v*(s)$'])):
        plt.subplot(1,2,i+1)
        plt.imshow(make_grid(vi,[5,5]),vmin=vmin,vmax=vmax)
        plt.title(title)
        plt.colorbar();

        if i == 1:
            for val, acts, (oy, ox) in zip(vi, actions_best, states):

                plot_actions(ox, oy, 0.2, acts[:,::-1], head_width=0.1)
        plot_grid(states, vi, 0.1, 0.4, 8)
        plt.axis('off');
```


```python
plot_init_final(vs, states, actions_best)
```


![png]({{ site.baseurl }}/assets/Bellman-Example/Bellman-Example_58_0.png)


#### Zero initialisation


```python
v_next_zero, vs_zero, diffs_zero, QQ_zero, RR_zero = estimate_vstar(np.zeros_like(v), P, rewards)
assert values_match(vstar, v_next_zero)
actions_best_zero =  get_best_actions(QQ_zero, RR_zero, gamma=0.9)
assert actions_match(actions_best, actions_best_zero)
plot_curves(vs_zero, diffs_zero)
plot_first_25(vs_zero, states)
plot_init_final(vs_zero, states, actions_best_zero)
```

    Iteration 154, mean squared difference 0.0000000


![png]({{ site.baseurl }}/assets/Bellman-Example/Bellman-Example_60_1.png)



![png]({{ site.baseurl }}/assets/Bellman-Example/Bellman-Example_60_2.png)



![png]({{ site.baseurl }}/assets/Bellman-Example/Bellman-Example_60_3.png)



![png]({{ site.baseurl }}/assets/Bellman-Example/Bellman-Example_60_4.png)


#### Random initialisation

Here each $v(s) \sim \mathcal{N}(\mu = 10, \sigma=1)$.


```python
v_next_rand, vs_rand, diffs_rand, QQ_rand, RR_rand = estimate_vstar(
    np.random.normal(loc=10,scale=1,size=v.shape), P, rewards)
assert values_match(vstar, v_next_rand)
actions_best_rand =  get_best_actions(QQ_rand, RR_rand, gamma=0.9)
assert actions_match(actions_best, actions_best_rand)
plot_curves(vs_rand, diffs_rand)
plot_first_25(vs_rand, states, 10)
plot_init_final(vs_rand, states, actions_best_rand)

```

    Iteration 151, mean squared difference 0.0000000


![png]({{ site.baseurl }}/assets/Bellman-Example/Bellman-Example_62_1.png)



![png]({{ site.baseurl }}/assets/Bellman-Example/Bellman-Example_62_2.png)



![png]({{ site.baseurl }}/assets/Bellman-Example/Bellman-Example_62_3.png)



![png]({{ site.baseurl }}/assets/Bellman-Example/Bellman-Example_62_4.png)



```python

```
