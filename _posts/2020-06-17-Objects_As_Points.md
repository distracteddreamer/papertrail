---
layout: post
title:  Objects As Points
date:   2020-06-17 13:56:28
categories: jekyll update
---

## Architecture
This paper introduces an architecture called CenterNet which is a fully convolutional approach to object detection. 

It takes some inspiration from key point detection like when you are predicting the joints when doing pose detection. It represents the bounding box as a centre point, height and width. It outputs a probability map for each pixel where the value indicates whether or not this point belongs to an object. This is done independently for each class. For each point it also predicts a height and width and an offset from the centre. The offset accounts for the fact that the output has a stride $S$ relative to the input. So the output is a $H/S \times W/S \times (C + 4)$ where they use $S=4$. 

This is very similar to anchor-based approaches where the parameters of a transform are predicited for each "anchor" box (which is a fixed size initial location for the object). Typically there is a set of anchor boxes of different sizes and aspect ratios evenly spaced across the image. 

![Diagram showing relationship between an anchor boxes and its target and a point and its target]({{ site.baseurl }}/assets/Objects_As_Points/anchors_points.png)

I think you could also add mask outputs, semantic segmentation style by adding a further mask head, which would let you do instance segmentation with this architecture. 

## Training
### Targets
The ground truth is obtained by projecting the centre points onto a pixel grid of size $H/S \times W/S$ and blurring them with a Gaussian kernel $\exp(-(x^2 + y^2)/\sigma^2)$. If Gaussians overlap they take the element wise max. So the ground truth is actually a probability map rather than a one-hot mask. 

### Losses
- Focal loss for the centre points 
    - A bit different from usual focal loss since the targets are probabilities rather than binary as is usually the case. 
    - Where probability is 1 behaves like usual focal loss
    - Otherwise it is weighted by (1-y)^beta to reduce contribution from pixels close to the centre.

- L1 loss for the bounding box dimensions where the ground truth is not normalised (only for the foreground points). 
- L1 loss for the offset, where ground truth is the difference between the real centre and the projected centre (only for the foreground points). 

## Inference
Select up-to top N (they use N=100) points whose confidence is higher than its neighbours. These points can be identified by max-pooling with a kernel-size of 3 after zero-padding with 1 to preserve the dimensions. You can then select those keypoint locations whose probability was chosen. As the offsets and dimensions are predicted per keypoint the corresponding values from these can also be selected in a similar manner.

![Diagram showing the CenterNet architecture along with target generation and inference steps]({{ site.baseurl }}/assets/Objects_As_Points/center_net.jpg)

## Results
It performs worse than most two-stage models but notably better than vanilla-MRCNN. For mAP, it is competitive or better than other one-stage models but not so good for small objects, even when they evaluate with different scales. However using a DLA-34 backbone it is faster than all the models compared at 28 FPS. 

## Discussion
### Simplicity 
A key advantage of this is that you can reduce the number of hyper-parameters quite a lot. For example in anchor based models, you need to decide how many anchor boxes to use, what sizes and scales. 

Another advantage is that targets are not assigned based on intersection over union. For RetinaNet/Faster-RCNN, you assign targets AFTER the bounding boxes have been predicted based on overlap with ground truth boxes which introduces further heuristics and hyperparameters.  

### Speed
For $C$ foreground classes, CenterNet has these many predictions: 

$$(H/4) \times (W/4) \times (C + 4) = \frac{1}{16} \cdot (H \times W \times (C + 4))$$ 

By contrast the original RetinaNet configuration has roughly $3$ times as many predictions, (for $A=9$)

$$\sum_{i=3}^{7}(H/2^i) \times (W/2^i) \times A \times (C + 4)
\\=\left(A\sum_{i=3}^{7}\frac{1}{4^i}\right)\cdot(H \times W \times (C + 4))
\\=\left(9\cdot\frac{1}{48}\left(1 - \frac{1}{4^5}\right)\right)\cdot(H \times W \times (C + 4))
\\ \approx \frac{3}{16}\cdot(H \times W \times (C + 4))$$

CenterNet could potentially be faster compared with a RetinaNet-like model with the same backbone but this would depend on the architecture of the decoder and the heads as well. In the paper they compare with a RetinaNet model using ResNet-101 and find that it is twice as fast with the same performance. 

### Concentric boxes
The main issue with this approach is when two objects have concentric bounding boxes then only one of these can be predicted. They address this by saying that this accounts for only a handful of the cases in their dataset but I think you can handle these corner case by assigning the ground truth to a neighbouring pixel and let the offset take care of this slight difference. 


