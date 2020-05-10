---
layout: post
title:  Transformer Notes
date:   2020-05-03 10:33:02
categories: jekyll update
---


An ongoing collection of notes on the [Transformer](https://arxiv.org/abs/1706.03762) architecture. I have used some code from [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) tutorial. 

## Masking 
When attending to other elements, elements are excluded as follows:
- For the source sequence, any pad elements
- For the target sequence, any pad elements plus any elements that appear after the present element. 

Masking is done in the following class (credit: [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) ):


```python
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
```

The target sequence is input to the model and masked in such a way that the model is not allowed to cheat which would cause it to fail to train properly. An element of the output must depend only on the input and on any earlier elements of the output. We deal with the corner case of the first element of the target sequence by padding the start of the sequence with a constant `START` symbol. As the image shows, the target mask lets the model look at this element but because it is a constant symbol no information about the target is leaked this way.
This is illustrated below for a single sequence where batch dimension is omitted for clarity:

![diagram-showing-source-target-masking]({{ site.baseurl }}/assets/Transformer_Notes/masking.png)
 