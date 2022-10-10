from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import datetime
from zernike import Coefficient
from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np
import os
import time
import threading

from tensorflow.python.framework import ops

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
                            
                           
NUM_CLASSES = 2
NUM_STEPS_FOR_CROSSVALIDATION = 100
CROSSVALIDATION = False

HEIGHT = 61
WIDTH = HEIGHT
CHANNELS = 1

def create_data():
  # Could also be implemented with threads, see below:
  #https://stackoverflow.com/questions/34594198/how-to-prefetch-data-using-a-custom-python-function-in-tensorflow
  c4 = np.random.uniform(low=0.0, high=3.0)
  c5 = np.random.uniform(low=0.0, high=3.0)
  c4 = np.random.uniform(low=0.0, high=3.0)
  c5 = np.random.uniform(low=0.0, high=3.0)
  
  zmap = Coefficient(Z4=c4, Z5=c5).psf(l = 31)
  zmap = np.expand_dims(zmap, axis=-1)
  return np.array(zmap, ndmin=3, dtype=np.float32), np.array([c4, c5], ndmin=1, dtype=np.float32)

def inputs(batch_size):
  image, label = tf.py_func(create_data, [], [tf.float32, tf.float32])
  image.set_shape([HEIGHT, WIDTH, CHANNELS])
  label.set_shape([NUM_CLASSES])
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 
                           min_fraction_of_examples_in_queue)
  
  print ('Wait while it fills up the queue with %d samples. ' % min_queue_examples)
  # Generate a batch of images and labels by building up a queue of examples.
  num_preprocess_threads = 16
  # use tf.train.batch() instead if no shuffling needed
  images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

  return images, tf.reshape(label_batch, [batch_size, NUM_CLASSES])


#Implementation with layers:
def inference(images):
  conv1 = tf.layers.conv2d(inputs=images, filters=16, kernel_size=[7, 7], padding="SAME", activation=None, name="conv1", kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
  conv1_bn = tf.nn.relu(tf.layers.batch_normalization(conv1))

  conv2 = tf.layers.conv2d(inputs=conv1_bn, filters=16, kernel_size=[7, 7], padding="SAME", activation=None, name="conv2", kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
  conv2_bn = tf.nn.relu(tf.layers.batch_normalization(conv2))

  conv3 = tf.layers.conv2d(inputs=conv2_bn, filters=16, kernel_size=[7, 7], padding="SAME", activation=None, name="conv3", kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
  conv3_bn = tf.nn.relu(tf.layers.batch_normalization(conv3))

  conv4 = tf.layers.conv2d(inputs=conv3_bn, filters=16, kernel_size=[7, 7], padding="SAME", activation=None, name="conv4", kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
  conv4_bn = tf.nn.relu(tf.layers.batch_normalization(conv4))

  poolN_flat = tf.reshape(conv4_bn, [tf.app.flags.FLAGS.batch_size, -1])
  dense1 = tf.layers.dense(inputs=poolN_flat, units=128, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
  dense2 = tf.layers.dense(inputs=dense1, units=32, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
  logits = tf.layers.dense(inputs=dense2, units=NUM_CLASSES, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

  return tf.reshape(logits, [tf.app.flags.FLAGS.batch_size, NUM_CLASSES])

def compute_loss(logits, labels):
  tf.losses.mean_squared_error(labels, logits, loss_collection=tf.GraphKeys.LOSSES)
  return tf.losses.get_total_loss(add_regularization_losses=True, name='total_loss')

def train(total_loss):
  # Compute gradients.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(total_loss)

  return train_op



max_count = 400   #Do several experiment to try out different hyperparameter configurations
for count in xrange(max_count):
  ops.reset_default_graph()

  INITIAL_LEARNING_RATE = tf.placeholder(shape=[], dtype=tf.float32, name="INITIAL_LEARNING_RATE")
  WEIGHT_DECAY_FACTOR =  tf.placeholder(shape=[], dtype=tf.float32, name="WEIGHT_DECAY_FACTOR")
  NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10   # whole data set size used for training 400000
    
  # Get next image and label batch from the queue
  images, labels = inputs(tf.app.flags.FLAGS.batch_size)

  # Build a Graph that computes the logits predictions from the inference model
  logits = inference(images)

  # Calculate loss
  loss = compute_loss(logits, labels)

  # Build a Graph that trains the model with one batch of examples and updates the model parameters
  train_op = train(loss)

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
    if CROSSVALIDATION == True:
      lr = 10**np.random.uniform(-3,-6)   # Fine tuning: (-1,-8)
    else:
      lr = 0.0001379342215827322

    reg = 0.0   #10**np.random.uniform(-5,5) #sample when crossvalidating
    step = 1
    while not coord.should_stop():
      # Run training steps or whatever
      _, loss_value = sess.run([train_op, loss], feed_dict={INITIAL_LEARNING_RATE: lr, WEIGHT_DECAY_FACTOR: reg})
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if CROSSVALIDATION == False:
        if step % 10 == 0:
          print(datetime.now(), "global_step=", step, "loss=", loss_value, "learning rate=", lr)

        if step == tf.app.flags.FLAGS.max_steps:
          coord.request_stop()
      else:
        if step == NUM_STEPS_FOR_CROSSVALIDATION:
          print(datetime.now(), "global_step=", step, "loss=", loss_value, "learning rate=", lr)
          coord.request_stop()

      step = step + 1
  
  
  except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
  finally:
    # When done, ask the threads to stop.
    coord.request_stop()
  
  # Wait for threads to finish.
  coord.join(threads)
  sess.close()
