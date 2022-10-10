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

tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
                            
                           
NUM_CLASSES = 2
#NUM_EPOCHS_FOR_CROSSVALIDATION

HEIGHT = 3
WIDTH = HEIGHT
CHANNELS = 1

def create_data():
  # Could also be implemented with threads, see below:
  #https://stackoverflow.com/questions/34594198/how-to-prefetch-data-using-a-custom-python-function-in-tensorflow 
  c4 = np.random.uniform(low=-10.0, high=10.0)
  c5 = np.random.uniform(low=-10.0, high=10.0)
  
  z = Coefficient(Z4 = c4, Z5 = c5)
  zmap = z.zernikematrix(l = HEIGHT)
  zmap = np.expand_dims(zmap, axis=-1)
  return np.array(zmap, ndmin=3, dtype=np.float32), np.array([c4, c5], ndmin=1, dtype=np.float32)

def inputs(batch_size):
  image, label = tf.py_func(create_data, [], [tf.float32, tf.float32])
  image.set_shape([HEIGHT, WIDTH, CHANNELS])
  label.set_shape([NUM_CLASSES])
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.1  #0.4_default
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
  logits = tf.layers.conv2d(inputs=images, filters=2, kernel_size=[HEIGHT, WIDTH], padding="VALID", activation=None, name="conv1", kernel_initializer=tf.ones_initializer)
  return tf.reshape(logits, [tf.app.flags.FLAGS.batch_size, NUM_CLASSES])

def compute_loss(logits, labels):
  tf.losses.mean_squared_error(labels, logits, loss_collection=tf.GraphKeys.LOSSES)
  return tf.losses.get_total_loss(add_regularization_losses=True, name='total_loss')

def train(total_loss):
  # Compute gradients.
  apply_gradient_op = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(total_loss)
  
  with tf.control_dependencies([apply_gradient_op]):   #do not call train_op until apply_gradient_op finishes
    train_op = tf.no_op(name='train')
  return train_op



max_count = 1 #400   #Do several experiment to try out different hyperparameter configurations
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
    lr = 0.001 #10**np.random.uniform(-3,-6)   #10**np.random.uniform(-1,-8)   #0.07   #coarse search: (-3,-6) (Default)
    reg = 0.0   #10**np.random.uniform(-5,5) #sample when crossvalidating
    step = 1
    while not coord.should_stop():
      # Run training steps or whatever
      _, loss_value, img, lbl, lgt = sess.run([train_op, loss, images, labels, logits], feed_dict={INITIAL_LEARNING_RATE: lr, WEIGHT_DECAY_FACTOR: reg})
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 100 == 0:        #step == 50: (crossvalidation)    #step % 10 == 0: (training)
        print(datetime.now(), "global_step=", step, "loss=", loss_value, "learning rate=", lr)
        gr = tf.get_default_graph()
        print("img", img.shape, img)
        print("lbl", lbl.shape, lbl)
        print("lgt", lgt.shape, lgt)
        conv1_kernel_val = gr.get_tensor_by_name('conv1/kernel:0').eval(session=sess)
        conv1_bias_val = gr.get_tensor_by_name('conv1/bias:0').eval(session=sess)
        print("conv1_kernel_val", conv1_kernel_val)
        print("conv1_bias_val", conv1_bias_val)

      if step == tf.app.flags.FLAGS.max_steps:        #step == 50: (crossvalidation)      #step == tf.app.flags.FLAGS.max_steps:  (training)
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
