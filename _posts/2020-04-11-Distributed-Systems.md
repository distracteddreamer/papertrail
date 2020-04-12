---
layout: post
title:  "[WIP] Distributed Systems"
date:   2020-04-11 20:32:00
categories: jekyll update
---

Notes from my study of [MIT's 6.824: Distributed Systems](https://pdos.csail.mit.edu/6.824/index.html). This is not intended to be contain a summary of the topics but thoughts and notes related to the concepts in the course.

## Week 1 — MapReduce
#### Master failure
- Whilst the method allows for workers failure, the master is not expected to fail. This is because there is only a single master and if there is a large number of computers, the probability of any a particular computer failing is low. However the probability that some machine will fail might not be negligible, hence the need to anticipate worker failure. 

- Let $p > 0$ be the probability that a given machine will fail and assume that the chance of failure is independent for each machine. Then for $N > 1$ machines, the probability that at least one machine in the cluster will fail, $P_\text{failure}$ = $1 - (1 - p)^N > 1 - (1 - p) = p$. As $N$ increases the probablity of failure increases whilst the probability that a given machine will fail is unchanged.  For example for p = 0.001, for N = 100, $P_\text{failure} \approx 0.01$ whilst for N = 1000, $P_\text{failure} \approx 0.63$.

#### Machine failure 
- If a worker fails during or after map task or during a reduce task, then the task status becomes "idle" and it becomes eligible for rescheduling elsewhere. A completed map task's results are stored locally so become inaccessible if the worker fails so it needs to be done again. 
- However those of a completed reduce task are stored in a global file system so it doesn't need to be re-run even if the worker fails after completing it. Other workers are informed on the failure and if they were meant to read data for a reduce task from worker A which has failed they will instead read it from worker B which has taken over the task.

#### Failure and determinism
- When map and reduce operators are deterministic every time you will get the same results as you would in a sequential execution of the whole program e.g. if you are doing word counts, you count the number of words in document 1, then count those in document 2 and increment the count, and so on. Even if there is a failure and you re-run some of the tasks this will be just the same.
- However in the absence of deterministic one reduce task R1's output might correspond to one sequential execution whilst another reduce task R2's might correspond to another sequential execution since each time the map task is re-run it might produce a different output. 