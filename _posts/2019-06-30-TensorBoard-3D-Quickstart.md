---
layout: post
title:  "TensorBoard for 3D data - quickstart"
date:   2019-06-30 11:51:32 +0100
categories: jekyll update
---

Right now the TensorBoard graphics functionality which allows you to display 3d data is part of the TensorBoard nightly build. 

So first install the TensorFlow nightly build:

{% highlight bash %}
pip3 install tf-nightly
{% endhighlight %}

Now import the mesh plugin from `tensorboard`

{% highlight python %}
import tensorflow as tf
from tensorboard.plugins.mesh import summary as mesh_summary
{% endhighlight %}

We will make use of the library meshzoo which as the name implies contains functionality to generate meshes for a variety of shapes. If you don't have the library install it via pip

{% highlight bash %}
pip3 install meshzoo
{% endhighlight %}

Let us create a summary for the mesh of a sphere and of a Moebius strip. First load the meshes (represented as set of vertices and edges) from mesh zoo.

{% highlight python %}
import meshzoo
points_sp, faces_sp = meshzoo.iso_sphere() # already between 1 and -1
print(points_sp.max(), points_sp.min())
#=> 1.0, -1.0
points_mb, faces_mb = meshzoo.moebius() # not between 1 and -1
print(points_mb.max(), points_mb.min())
#=> 15.0, -13.640587955517248
points_mb /= points_mb.max()
print(points_mb.max(), points_mb.min())
#=> 1.0 -0.9093725303678165
{% endhighlight %}

Then create a summary for each mesh. The mesh can have colours for each vertex. 
One important point to note is that the mesh inputs should all have a batch dimension, which we add below. We will also create a placeholder for a single colour that is then multiplied by the point coordinates - just to create an arbitrary set of colours for each point!

{% highlight python %}
clr_sp = tf.placeholder(dtype=tf.float32, shape=[3])
summary_sp = mesh_summary.op('sphere_mesh', 
                         vertices=tf.constant(points_sp[None]),
                         faces=tf.constant(faces_sp[None]),
                         colors=(clr_sp * points_sp)[None])
clr_mb = tf.placeholder(dtype=tf.float32, shape=[3])
summary_mb = mesh_summary.op('moebius_mesh', 
                            vertices=tf.constant(points_mb[None]),
                            faces=tf.constant(faces_mb[None]),
                            colors=(clr_mb * points_mb)[None])
{% endhighlight %}

Now it is possible to use this as a normal tensorboard summary. First we need to create a summary writer.

{% highlight python %}
writer = tf.summary.FileWriter('./summaries')
summary = tf.summary.merge([summary_sp, summary_mb])
{% endhighlight %}

Then create a session and run the summary op and add the summary string to the writer 

{% highlight python %}
with tf.Session() as sess:
    sum_str = sess.run(summary, 
                feed_dict={clr_sp: [255, 0, 0],  
                           clr_mb: [255,255,0]})
    writer.add_summary(sum_str, 0)
{% endhighlight %}

Now open Tensorboard as usual and view the rendered meshes.

{% highlight bash %}
tensorboard --logdir summaries
{% endhighlight %}

<img src="{{ site.baseurl }}/assets/tb-sphere.jpg">
<br>
<br>
<img src="{{ site.baseurl }}/assets/tb-moebius.jpg">