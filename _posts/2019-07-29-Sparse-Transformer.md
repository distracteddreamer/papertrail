---
layout: post
title:  "Sparse Transformer"
date:   2019-07-29 11:51:32 +0100
categories: jekyll update
---
These are my notes on the paper [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509). All mistakes are my own. 

## Motivation
In the standard Transformer (see this excellent tutorial to understand it in depth) self-attention,each position attends to all previous positions which is $O(n^2)$ making it prohibitively expensive for long sequences. It turns out that in practice at least some of the learned attention maps are quite sparse, sometimes attending only to specific points for certain categories or paying attention only to some columns or rows - the latter suggestive of some kind of factorisation of the attention maps. This motivates the design of sparse attention heads that for each point only attend to a subset of previous locations. However there are several such heads each with a different subset so that via a combination of heads, a path exists from each previous location to the present one. The complexity is now $O(n\sqrt(n))$ rather than $O(n^2)$.

## Model
For each attention type a cell attends to a fixed number of previous cells that depend on its location and via the cells to which a cell attends it gets summarised information from other cells. They use two attention heads and discuss three ways in which this could be implemented. 

1. Each residual block has one attention type and blocks with different attention types are interleaved. 2. The subset pixels of each attention type are combined across all the types and used in a single *merged* head.
3. Different attention patterns are run in parallel and the result is concatenated along the feature dimension - the attention types could be 

The architecture was modified to use a different kind of residual block and embedding layer compared to the original. 

## Results
The model achieved superior performance for density modelling tasks on CIFAR10 compared to other models in a briefer time. For density modelling on EnWik8  it was as good as the best model (Transformer-XL 277M) which used over twice as many parameters as the Sparse Transformer. The model also performed better than other models larger images in ImageNet 64x64. In addition they trained models for a classical music dataset but did not compare to other models because of the lack of details of dataset processing but obtained qualitatively good results of clips of around 5 seconds in length.



