---
layout: post
title:  "Daily ML - Gated-SCNN"
date:   2019-07-28 9:51:32 +0100
categories: jekyll update
---

These are my notes on the paper [Gated-SCNN](https://arxiv.org/abs/1907.05740). All mistakes are my own. 

## Model 
A segmentation network consisting of two streams - shape and boundary. One stream is like a usual segmentationn network producing a mask. The other parallel stream takes feature maps from the other stream and uses them in a gated attention block between residual blocks of its own. It starts from a layer in the shape stream. These are then combined in a fusion stream using atrous spatial pyramid pooling to produce the final mask.  

## Losses 
There are four losses. A binary CE for the boundary (class-agnostic loss that just tries to ensure that boundaries are found), a softmax CE loss for the masks (as usual). and two rregularisation losses. The first takes the argmax of the masks, passes a Gaussian filter through them, takes the image graadients and finds the boundary loss for this comapred to the boundary loss for this compared to the GT mask processed in the same way. The other loss is a softmax loss for the dense masks over all the pixels that are higher than some threshold in the boundary mask.

## Metrics
The model is trained and evaluated on Cityscapes using three different metrics, mean intersection over union (IoU averaged across all classes), F1 score for the boundary (with hit criteria based on a small slack factor) and a distance based mIoU. The purpose of the last metric is to evaluate models performance on objects distant from the camera by taking crops from roughly the centre of the image (around an approximate vanishing point) - so that  small objects will account for a larger fraction of pixels of their class - and then finding the IoU for these. 

## Results
For mean IoU, mean F-score at different slack factors and distance-based IoU for different sizes, the model outperforms the state of the art on Cityscapes, DeepLab v3+. It outperforms even models (including DeepLab v3+) that are trained on additional data consisting of the coarse annotations in Cityscapes (which are not suitable for boundary based training so not used for Gated-SCNN). 




