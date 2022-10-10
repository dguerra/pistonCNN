from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import datetime
from zernike import Coefficient
from matplotlib import pyplot as plt

import threading
import tensorflow as tf
import numpy as np
import os
import time

from tensorflow.python.framework import ops

tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Number of points to process in a batch.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
                            
                           
NUM_CLASSES = 1

def create_data():
  x = np.random.uniform(low=-10.0, high=10.0)
  y = np.random.uniform(low=-10.0, high=10.0)
  # Create data from scratch using the known relation
  z = x * 5.0 + y * 6.0 + 7.0
  p = np.array([x, y], dtype=np.float32)
  return np.array(p, ndmin=1, dtype=np.float32), np.array(z, ndmin=1, dtype=np.float32)


def inputs(batch_size):
  # Create next batch. Each sample contains PSF with random Z4 coefficient (among others coefficients) and label the piece of data with this Z4 value
  point, label = tf.py_func(create_data, [], [tf.float32, tf.float32])
  point.set_shape([2])
  label.set_shape([1])
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.1  #0.4_default
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 
                           min_fraction_of_examples_in_queue)
  
  print ('Wait while it fills up the queue with %d PSF examples. ' % min_queue_examples)
  # Generate a batch of points and labels by building up a queue of examples.
  num_preprocess_threads = 16
  # use tf.train.batch() instead if no shuffling needed
  points, label_batch = tf.train.shuffle_batch(
        [point, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

  return points, tf.reshape(label_batch, [batch_size, NUM_CLASSES])


ops.reset_default_graph()
W = tf.get_variable(name="Weights", shape=[2,1])
b = tf.get_variable(name="Bias", shape=[1])


INITIAL_LEARNING_RATE = tf.placeholder(shape=[], dtype=tf.float32, name="INITIAL_LEARNING_RATE")
WEIGHT_DECAY_FACTOR =  tf.placeholder(shape=[], dtype=tf.float32, name="WEIGHT_DECAY_FACTOR")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10   # whole data set size used for training 400000

# Get next image and label batch from the queue
points, labels = inputs(tf.app.flags.FLAGS.batch_size)

# Build a Graph that computes the logits predictions from the inference model
logits = tf.nn.bias_add(tf.matmul(points, W), b)

# Calculate loss
loss = tf.reduce_mean(tf.nn.l2_loss(labels-logits, name='l2_loss'))

# Build a Graph that trains the model with one batch of examples and updates the model parameters
train_op = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(loss)

init = tf.global_variables_initializer()
#initilize epoch counter within string_input_producer. Is deprecated, use tf.local_variables_initializer() instead
init_local = tf.local_variables_initializer()  
  
sess = tf.Session(config=tf.ConfigProto(log_device_placement=tf.app.flags.FLAGS.log_device_placement))


sess.run(init)
sess.run(init_local)
    
# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
  lr = 0.01 #10**np.random.uniform(-1,-8)   #0.07   #coarse search: (-3,-6) (Default)
  reg = 0.0   #10**np.random.uniform(-5,5) #sample when crossvalidating
  for global_step in range(tf.app.flags.FLAGS.max_steps):
    # Run training steps or whatever
    _, loss_value, W_i, b_i = sess.run([train_op, loss, points, labels], feed_dict={INITIAL_LEARNING_RATE: lr, WEIGHT_DECAY_FACTOR: reg})
    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    if global_step % 10 == 0:        #step == 50: (crossvalidation)    #step % 10 == 0: (training)
      print(datetime.now(), "global_step=", global_step, "loss=", loss_value, "learning rate=", lr)
      print(W_i, b_i)

    if global_step == tf.app.flags.FLAGS.max_steps:        #step == 50: (crossvalidation)      #step == tf.app.flags.FLAGS.max_steps:  (training)
      coord.request_stop()
  
  
except tf.errors.OutOfRangeError:
  print('Done training -- epoch limit reached')
finally:
  # When done, ask the threads to stop.
  coord.request_stop()
  
# Wait for threads to finish.
coord.join(threads)
sess.close()
