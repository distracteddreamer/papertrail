---
layout: post
title:  EM-Algorithm Convergence
date:   2020-05-23 14:56:28
categories: jekyll update
---


$$\newcommand{\Qz}{Q_i(z^{(i)})}$$
$$\newcommand{\Qstarz}{Q_i^{*}(z^{(i)})}$$
$$\newcommand{\Qzt}[1]{Q_i^{(#1)}(z^{(i)})}$$
$$\newcommand{\Ef}[2]{E_{#1}\left[#2\right]}$$
$$\newcommand{\EQt}[2]{E_{\Qzt{#1}}\left[#2\right]}$$
$$\newcommand{\losst}[2]{\sum_{i}\EQt{#1}{#2}}$$
$$\newcommand{\lt}[1]{l^{(#1)}}$$
$$\newcommand{\thetat}[1]{\theta^{(#1)}}$$
$$\newcommand \pr[1]{p\left(#1\right)}$$
$$\newcommand{\px}{\pr{x^{(i)}; \theta}}$$
$$\newcommand{\pxt}[1]{\pr{x^{(i)}; \theta^{(#1)}}}$$
$$\newcommand{\pxz}{\pr{x^{(i)}, z^{(i)}; \theta}}$$
$$\newcommand{\pxzt}[1]{\pr{x^{(i)}, z^{(i)}; \theta^{(#1)}}}$$
$$\newcommand{\pzgx}{\pr{z^{(i)}| x^{(i)}; \theta}}$$
$$\newcommand{\pzgxt}[1]{\pr{z^{(i)}| x^{(i)}; \theta^{(#1)}}}$$
$$\newcommand{\phizi}{\phi(z^{i})}$$

## Introduction

We wish to show that the log-likelihood $l(\theta)$ increases with each step of the EM-algorithm:

$$l(\theta) = \sum_{i}\log\Ef{\Qz}{\frac{\pxz}{\Qz}}$$

By Jensen's inequality

$$l(\theta) = \sum_{i}\log\Ef{\Qz}{\frac{\pxz}{\Qz}} \geq \sum_{i}\Ef{\Qz}{\log\frac{\pxz}{\Qz}} = f(\theta) \text{ }\text{ }\text{ }[1]$$

where $f(\theta)$ is the objective we maximise.

## Results from the algorithm

Let us consider the steps of iteration $t+1$. 

### E-step
Choose $\Qz$ using the parameters found in the previous iteration, $\thetat {t}$ to make the LHS and RHS of Jensen's inequality, equal at $\theta = \thetat t$

$$\Qzt{t+1} = \pzgxt {t}$$ 

Then we have that 

$$\sum_{i}\Ef{\Qzt {t+1}}{\log\frac{\pxzt t}{\Qzt {t+1}}} =
\sum_{i}\log\Ef{\Qzt {t+1}}{\frac{\pxzt t}{\Qzt {t+1}}} \text{ }\text{ }\text{ }[2]$$

### M-step
The objective to maximise with respect to $\theta$ is 

$$f(\theta) = \losst{t + 1}{\log \frac{\pxz}{\pzgxt t}}$$

This results in $\thetat {t+1} = \arg \max_\theta f(\theta)$


## Proof of convergence
With $\Qz = \Qzt {t+1}$ and $\theta = \thetat {t+1}$ we have by [1] above that:

$$l(\thetat {t+1}) \geq \sum_{i}\Ef{\Qzt {t+1}}{\log\frac{\pxzt {t+1}}{\Qzt {t+1}}} = f(\thetat {t + 1})$$

Since $\thetat {t + 1}$ maximises $f(\theta)$

$$f(\thetat {t + 1}) \geq f(\thetat {t}) = \sum_{i}\Ef{\Qzt {t+1}}{\log\frac{\pxzt {t}}{\Qzt {t+1}}} = \sum_{i}\log\Ef{\Qzt {t+1}}{\frac{\pxz}{\Qzt {t+1}}} = l(\thetat t)$$

where we use [2] above to equate $ f(\thetat {t})$ with $l(\thetat t)$

Thus $l(\thetat {t+ 1}) \geq l(\thetat t)$

