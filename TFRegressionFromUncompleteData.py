from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import datetime
from matplotlib import pyplot as plt
from simulacion_piston import axial_to_matrix
from simulacion_piston import cube_to_axial

import threading
import tensorflow as tf
import numpy as np
import os
import time
from math import *
import random

from tensorflow.python.framework import ops

tf.app.flags.DEFINE_integer('batch_size', 256,
                            """Number of points to process in a batch.""")
tf.app.flags.DEFINE_integer('max_steps', 10001, #1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
                            
                           
LAMBDA_RANGES = 11
nrings = 3
nsegs = (3 * nrings * (nrings + 1)) + 1
ninput = 2 * 3 * 27

def inference(points):
  dense1 = tf.layers.dense(inputs=points, units=256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
  
  dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
  dense3 = tf.layers.dense(inputs=dense2, units=nsegs, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

  return tf.reshape(dense3, [tf.app.flags.FLAGS.batch_size, nsegs])


def create_data():
  pstn = np.random.uniform(-LAMBDA_RANGES * pi, LAMBDA_RANGES * pi, [(2*nrings)+1, (2*nrings)+1])

  x_stack = []
  y_stack = []
  offset_value = pstn[axial_to_matrix(0, 0, nrings)]

  pstn = pstn - offset_value
  rand_indx = random.sample(range(0,54), 0)

  count = 0
  for dx in range(-nrings, nrings):
    for dy in range( max(-nrings, -dx-nrings), min(nrings, -dx+nrings)):
      dq, dr = cube_to_axial(dx,dy,-dx-dy)
      pA = pstn[axial_to_matrix(dq,   dr,   nrings)]  
      pB = pstn[axial_to_matrix(dq,   dr-1, nrings)]   #upper hexagon
      pC = pstn[axial_to_matrix(dq+1, dr-1, nrings)]   #upper right hexagon
      if count in rand_indx:
        x_i = np.stack([0.0, 0.0, 0.0], axis=0)        #set to zero one entry randomly
      else:
        x_i = np.stack([pA-pB, pB-pC, pC-pA], axis=0)
      x_stack.append(x_i)
      count = count + 1

      dqf, drf = cube_to_axial(-dx, -(-dx-dy), -dy)
      pAf = pstn[axial_to_matrix(dqf,   drf,   nrings)]  
      pBf = pstn[axial_to_matrix(dqf,   drf-1, nrings)]   #upper hexagon
      pCf = pstn[axial_to_matrix(dqf-1, drf,   nrings)]   #upper LEFT hexagon
      if count in rand_indx:
        x_if = np.stack([0.0, 0.0, 0.0], axis=0)        #set to zero one entry randomly
      else:
        x_if = np.stack([pAf-pBf, pBf-pCf, pCf-pAf], axis=0)
      x_stack.append(x_if)

      count = count + 1

  
  for dx in range(-nrings, nrings+1):
    for dy in range( max(-nrings, -dx-nrings), min(nrings, -dx+nrings)+1):
      dz = -dx-dy
      dq, dr = cube_to_axial(dx,dy,dz)
      p = pstn[axial_to_matrix(dq, dr, nrings)]
      y_i = np.stack([p - offset_value], axis=0)
      y_stack.append(y_i)

        
  x_stack = np.stack(x_stack, axis=0)
  x = np.reshape(x_stack, [-1])
  y_stack = np.stack(y_stack, axis=0)
  y = np.reshape(y_stack, [-1])
  
  #x[np.random.randint(0, np.size(x)-1)] = np.random.uniform( -2.0 * LAMBDA_RANGES * pi, 2.0 * LAMBDA_RANGES * pi )
  
  #x = np.random.uniform(low=-10.0, high=10.0)
  #y = (x * 5.0) + 7.0    # Create data from scratch using the known relation
  #p = np.array([x], dtype=np.float32)
  return np.array(x, ndmin=1, dtype=np.float32), np.array(y, ndmin=1, dtype=np.float32)


def inputs(batch_size):
  # Create next batch
  point, label = tf.py_func(create_data, [], [tf.float32, tf.float32])
  point.set_shape([ninput])
  label.set_shape([nsegs])
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 
                           min_fraction_of_examples_in_queue)
  
  print ('Wait while it fills up the queue with %d examples. ' % min_queue_examples)
  # Generate a batch of points and labels by building up a queue of examples.
  num_preprocess_threads = 16
  # use tf.train.batch() instead if no shuffling needed
  points, label_batch = tf.train.shuffle_batch(
        [point, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

  return points, tf.reshape(label_batch, [batch_size, nsegs])
  

ops.reset_default_graph()
INITIAL_LEARNING_RATE = tf.placeholder(shape=[], dtype=tf.float32, name="INITIAL_LEARNING_RATE")
WEIGHT_DECAY_FACTOR =  tf.placeholder(shape=[], dtype=tf.float32, name="WEIGHT_DECAY_FACTOR")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10   # whole data set size used for training 400000

# Get next image and label batch from the queue
points, labels = inputs(tf.app.flags.FLAGS.batch_size)

# Build a Graph that computes the logits predictions from the inference model
#logits = tf.nn.bias_add(tf.matmul(points, W), b)
logits = inference(points)

# Calculate loss
#loss = tf.reduce_mean(tf.nn.l2_loss(labels-logits, name='l2_loss'))
loss = tf.losses.mean_squared_error(labels, logits)

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
  lr = 0.001 #10**np.random.uniform(-1,-8)   #0.07   #coarse search: (-3,-6) (Default)
  reg = 0.0   #10**np.random.uniform(-5,5) #sample when crossvalidating
  for global_step in range(tf.app.flags.FLAGS.max_steps):
    # Run training steps or whatever
    _, loss_value, po, la, lo = sess.run([train_op, loss, points, labels, logits], feed_dict={INITIAL_LEARNING_RATE: lr, WEIGHT_DECAY_FACTOR: reg})
    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    if global_step % 10 == 0:        #step == 50: (crossvalidation)    #step % 10 == 0: (training)
      #print(datetime.now(), "global_step=", global_step, "loss=", loss_value, "learning rate=", lr)
      print("*"+str(global_step)+"*"+str(loss_value).replace(".", ","))


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


