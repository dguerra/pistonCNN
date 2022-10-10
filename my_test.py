import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

probs_disc = tf.constant( [[0.5,0.25,0.25], [0.0, 0.0, 1.0], [0.3,0.1,0.6], [0.3,0.1,0.6]] )
#probs_disc = tf.constant( [0.5,0.4,0.1] )
print(probs_disc.shape)
#cat_dist = tf.contrib.distributions.Multinomial(logits=probs_disc, total_count=3.0)

p = [.2, .3, .5]
cat_dist = tf.contrib.distributions.Multinomial(total_count=1., probs=p, validate_args=True)

#m = cat_dist.sample([1])

#s = cat_dist.prob(0)
#print("s", s)
o = tf.ones(shape=[10,5])
o = tf.nn.softmax(o)

#samples = tf.random.categorical(tf.log([[10., 10.],[10., 10.],[10., 10.],[10., 10.],[10., 10.]]), 1)
m = tf.random.categorical(tf.log(o), 1)
print("hello, {}".format(m))
