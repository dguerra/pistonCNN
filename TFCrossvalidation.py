from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import numpy as np
import os


from tensorflow.python.framework import ops

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

max_count = 50   #Do several experiment to try out different hyperparameter configurations
for count in xrange(max_count):

  ops.reset_default_graph()
    
  one = tf.constant(1)
  global_step = tf.Variable(0, trainable=False)
  plus_one = tf.add(one,global_step)
  update = tf.assign(global_step, plus_one)
  
  # Try to find values for W and b that compute y_data = W * x_data + b
  # (We know that W should be 0.1 and b 0.3, but TensorFlow will
  # figure that out for us.)
  #W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
  W = tf.Variable(0.5)
  b = tf.Variable(tf.zeros([1]))
  y = W * x_data + b
    
  #crossvalidate learning rate value in the range [1e-5, 1e-3]
  INITIAL_LEARNING_RATE = tf.placeholder(shape=[], dtype=tf.float32, name="INITIAL_LEARNING_RATE")
    
  # Minimize either mean squared or l1 errors.
  loss = tf.reduce_mean(tf.square(y - y_data))   #l2 loss
  #loss = tf.reduce_mean(tf.abs(y - y_data))      #l1 loss
  optimizer = tf.train.GradientDescentOptimizer(INITIAL_LEARNING_RATE)
  #optimizer = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE)
  train = optimizer.minimize(loss)
  
  
  
  init = tf.global_variables_initializer()
  #initilize epoch counter within string_input_producer. Is deprecated, use tf.local_variables_initializer() instead
  init_local = tf.local_variables_initializer()  
  
  sess = tf.Session()
  sess.run(init)
  sess.run(init_local)
    
    
  # Start input enqueue threads.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  
  try:
    lr = 10**np.random.uniform(-3,-6)   #coarse search: (-3,-6)
    #reg = 10**np.random.uniform(-5,5)
    while not coord.should_stop():
      # Run training steps or whatever
      _, step_n, loss_val = sess.run([train, update, loss], feed_dict={INITIAL_LEARNING_RATE: lr})
      if step_n == 400:  #
        print("loss=", loss_val, "learning rate=", lr)
        coord.request_stop()
  
  
  except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
  finally:
    # When done, ask the threads to stop.
    coord.request_stop()
  
  # Wait for threads to finish.
  coord.join(threads)
  sess.close()
