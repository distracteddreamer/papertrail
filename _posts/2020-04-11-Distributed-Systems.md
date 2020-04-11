---
layout: post
title:  "[WIP] Distributed Systems"
date:   2020-04-11 20:32:00
categories: jekyll update
---

Notes from my study of [MIT's 6.824: Distributed Systems](https://pdos.csail.mit.edu/6.824/index.html). This is not intended to be contain a summary of the topics but thoughts and notes related to the concepts in the course.

## Week 1 — MapReduce
- Whilst the method allows for workers failure, the master is not expected to fail. This is because there is only a single master and if there is a large number of computers, the probability of any a particular computer failing is low. However the probability that some machine will fail might not be negligible, hence the need to anticipate worker failure. 

- Let $p > 0$ be the probability that a given machine will fail and assume that the chance of failure is independent for each machine. Then for $N > 1$ machines, the probability that at least one machine in the cluster will fail, $P_\text{failure}$ = $1 - (1 - p)^N > 1 - (1 - p) = p$. As $N$ increases the probablity of failure increases whilst the probability that a given machine will fail is unchanged.  For example for p = 0.001, for N = 100, $P_\text{failure} \approx 0.01$ whilst for N = 1000, $P_\text{failure} \approx 0.63$.