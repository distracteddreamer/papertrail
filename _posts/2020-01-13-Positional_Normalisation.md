---
layout: post
title:  Postional Normalisation
date:   2020-01-13 22:58:02
categories: jekyll update
---

These are my notes on the paper [Positional Normalization](https://arxiv.org/abs/1907.04312). All mistakes are my own.


```python
import tensorflow as tf
```

## Positional Normalisation (PONO)

- The feature map is normalised with channel-wise moments


```python
class PONO(tf.keras.layers.Layer):
    def __init__(self, channels_first=False, moments_only=False):
        self.channels_first = channels_first
        # In case we just want the moments to use for moment shortcut
        self.moments_only = moments_only
        super(PONO, self).__init__()
    
    @property
    def _channel_axis(self):
        return 1 if self.channels_first else -1
        
    def __call__(self, x):
        mu, var = tf.nn.moments(x, axis=self._channel_axis, keepdims=True)
        std = tf.sqrt(var)
        x = (x - mu) / std
        if moments_only:
            return mu, std
        return x, mu, std
```

## Moment shortcut (MS)

- Akin to skip connection exception that it uses the moments of the output from a layer of the encoder to transform the input to a corresponding layer in the decoder.
- Explicit bias given to decoder layer so that its activations have similar statistics to corresponding encoder layer.

### Dynamic moment shortcut (DMS)
- The difference here is that the moments are input to a small ConvNet that outputs the parameters used to transform the feature map


```python
class MomentShortcut(tf.keras.models.Model):
    def __init__(self, channels_first=False, 
                 net=None):
        self.channels_first = channels_first
        self.net = net
        super(MomentShortcut, self).__init__()
        
    @property
    def _channel_axis(self):
        return 1 if self.channels_first else -1
        
    def __call__(self, x, mu, std):
        # mu, std are outputs of PONO or other network
        if self.net is not None:
            out = self.net(tf.concat([mu, std]), axis=self._channel_axis)
            beta, gamma = tf.split(out, axis=self._channel_axis, num_or_size_splits=2)
        else:
            beta, gamma = mu, std
        x = gamma * x + beta
        return x
```

## Results

### Effect of the different components

- In almost all experiments combination of PONO-MS and other normalisation method improves FID of CycleGAN. 

- Using PONO-MS along with other methods also does better compared to PONO-MS alone.

- PONO-MS tends to be better than just MS, whilst PONO-DMS, with a suitable architecture (which seems to be task-dependent), is able to improve over PONO-DMS

### Image translation
- For **CycleGAN**, FID is better than baseline and best out all other models compared for three of four cases (for Map→Photo another model is better but PONO-MS is better for Photo→Map, which they try to explain in the paper)
- For **Pix2pix**, FID improves for all tasks although sometimes only marginally and LPIPS typically improves but where it doesn't is close to the baseline (i.e. does not make it worse).

### Style transfer
- Typical model 
    - Encoder to get content from source
    - Encoder to get style features from target
    - Decoder that takes both features and makes the output
- PONO-MS improves performance using both MUNIT and DRIT on almost all the performance metrics on both the experiments (Cat ↔ Dog, Portrait ↔ Photo)
- Improvement is only marginal for DRIT and least in both for FID (which is a data level metric whilst others are instance level and they suggest that this might be the reason)

