from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import datetime
from zernike import Coefficient
from matplotlib import pyplot as plt
from scipy import misc
from scipy import signal

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
 
LOGDIR = "/tmp/phasedivcnn/1"
#https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial/blob/master/mnist.py
#floyd login
#floyd init phasedivcnn
#floyd run --gpu --env tensorflow-1.3 --data dguerra/datasets/ground_image/1:/my_data "python TFTrainCNN.py"
                           
NUM_CLASSES = 2
NUM_STEPS_FOR_CROSSVALIDATION = 100
CROSSVALIDATION = False

HEIGHT = 128
WIDTH = HEIGHT
CHANNELS = 2

IMG = misc.imread("/my_data/vray_osl_simplex_noise_10.png", flatten=True).astype(np.float32)
IMG = IMG/(0.01*IMG.max())

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def conv_layer(inputs, filters, kernel_size=7, name="conv", dtype=tf.float32, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), wd=None):
  with tf.variable_scope(name):
    #TO-DO: Check input dims are 4 (batch, height, width, channels)
    input_channels = inputs.get_shape()[-1].value   #data_format = channels_last: means (batch, height, width, channels)
    weights = tf.get_variable("weights", shape=[kernel_size, kernel_size, input_channels, filters], initializer=kernel_initializer, dtype=dtype)
    biases = tf.get_variable("biases", shape=[filters], initializer=tf.constant_initializer(0.0), dtype=dtype)
    if wd is not None:
      weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
      tf.add_to_collection(tf.GraphKeys.LOSSES, weight_decay)

    conv = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases)      
    bn = tf.layers.batch_normalization(pre_activation)
    act = tf.nn.relu(bn)
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("biases", biases)
    tf.summary.histogram("activations", act)
    #tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    return act


def fc_layer(inputs, units, name="fc", dtype=tf.float32, kernel_initializer=tf.contrib.layers.xavier_initializer()):
  with tf.variable_scope(name):
    #TO-DO: Check input dims are 2 (batch_size, -1)
    dim = inputs.get_shape()[-1].value
    weights = tf.get_variable(name="weights", shape=[dim, units], initializer=kernel_initializer, dtype=dtype)
    biases = tf.get_variable("biases", shape=[units], initializer=tf.constant_initializer(0.0), dtype=dtype)
    mat = tf.matmul(inputs, weights)
    pre_activation = tf.nn.bias_add(mat, biases)
    bn = tf.layers.batch_normalization(pre_activation)
    act = tf.nn.relu(bn)
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("biases", biases)
    tf.summary.histogram("activations", act)
    return act
 
def create_data():
  # Crop a square from a random location
  # Could also be implemented with threads, see below:
  #https://stackoverflow.com/questions/34594198/how-to-prefetch-data-using-a-custom-python-function-in-tensorflow
  c4 =  np.random.uniform(low=0.0, high=3.5)
  c11 = np.random.uniform(low=0.0, high=3.5)
  #Random location 
  xi = np.random.randint(IMG.shape[0]-WIDTH)
  yi = np.random.randint(IMG.shape[1]-HEIGHT)
  phasediv = 1.0
  i_1 = signal.fftconvolve(IMG[xi:xi+WIDTH, yi:yi+HEIGHT], Coefficient(Z4=c4 + 0.0*phasediv, Z11=c11).psf(r=1.0, l = 63), mode="same")
  i_2 = signal.fftconvolve(IMG[xi:xi+WIDTH, yi:yi+HEIGHT], Coefficient(Z4=c4 + 0.6*phasediv, Z11=c11).psf(r=1.0, l = 63), mode="same")
  
  #Two separate input images as inputs: {i_foc, i_defoc}
  dat = np.stack((i_1,i_2), axis=-1)

  return np.array(dat, ndmin=3, dtype=np.float32), np.array([c4, c11], ndmin=1, dtype=np.float32)

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
  try:
    images, label_batch = tf.train.shuffle_batch(
          [image, label],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples)
  except tf.errors.OutOfRangeError:
    print('Input queue is exhausted')

  return images, tf.reshape(label_batch, [batch_size, NUM_CLASSES])


#Implementation with layers:
def inference(images):
  conv1 = conv_layer(inputs=images, filters=16, name="conv1")
  conv2 = conv_layer(inputs=conv1, filters=16, name="conv2")
  conv3 = conv_layer(inputs=conv2, filters=16, name="conv3")
  conv4 = conv_layer(inputs=conv3, filters=16, name="conv4")
  conv5 = conv_layer(inputs=conv4, filters=16, name="conv5")
  conv6 = conv_layer(inputs=conv5, filters=16, name="conv6")
  conv7 = conv_layer(inputs=conv6, filters=16, name="conv7")
  conv8 = conv_layer(inputs=conv7, filters=16, name="conv8")
  conv9 = conv_layer(inputs=conv8, filters=16, name="conv9")
  conv10 = conv_layer(inputs=conv9, filters=16, name="conv10")
  conv11 = conv_layer(inputs=conv10, filters=16, name="conv11")
  conv12 = conv_layer(inputs=conv11, filters=16, name="conv12")

  poolN_flat = tf.reshape(conv12, [tf.app.flags.FLAGS.batch_size, -1])

  fc1 = fc_layer(inputs=poolN_flat, units=128, name="fc1")
  fc2 = fc_layer(inputs=fc1, units=32, name="fc2")
  logits = fc_layer(inputs=fc2, units=NUM_CLASSES, name="fc3")    #TO-DO: Last fully connected does not need activation relu!!!!

  return tf.reshape(logits, [tf.app.flags.FLAGS.batch_size, NUM_CLASSES])

def compute_loss(logits, labels):
  with tf.name_scope("loss"):
    the_loss = tf.losses.mean_squared_error(labels, logits, loss_collection=tf.GraphKeys.LOSSES)
    tf.summary.scalar('loss', the_loss)
    return tf.losses.get_total_loss(add_regularization_losses=True, name='total_loss')

def train(total_loss):
  # Compute gradients.
  with tf.name_scope("train"):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)   #Needed for tf.layers.batch_normalization to take effect
    with tf.control_dependencies(update_ops):
      apply_gradient_op = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(total_loss)
  
      with tf.control_dependencies([apply_gradient_op]):   #do not call train_op until apply_gradient_op finishes
        train_op = tf.no_op(name='train')
      return train_op


max_count = 400   #Do several experiment to try out different hyperparameter configurations
for count in xrange(max_count):
  ops.reset_default_graph()

  INITIAL_LEARNING_RATE = tf.placeholder(shape=[], dtype=tf.float32, name="INITIAL_LEARNING_RATE")
  WEIGHT_DECAY_FACTOR =  tf.placeholder(shape=[], dtype=tf.float32, name="WEIGHT_DECAY_FACTOR")
  NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100   # whole data set size used for training 400000
    
  # Get next image and label batch from the queue
  images, labels = inputs(tf.app.flags.FLAGS.batch_size)

  # Build a Graph that computes the logits predictions from the inference model
  logits = inference(images)

  # Calculate loss
  loss = compute_loss(logits, labels)

  # Build a Graph that trains the model with one batch of examples and updates the model parameters
  train_op = train(loss)

  sess = tf.Session(config=tf.ConfigProto(log_device_placement=tf.app.flags.FLAGS.log_device_placement))
  
  summ = tf.summary.merge_all()
  
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
    
  writer = tf.summary.FileWriter(LOGDIR)
  writer.add_graph(sess.graph)

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
      _, loss_value, s = sess.run([train_op, loss, summ], feed_dict={INITIAL_LEARNING_RATE: lr, WEIGHT_DECAY_FACTOR: reg})
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if CROSSVALIDATION == False:
        if step % 10 == 0:
          print(datetime.now(), "global_step=", step, "loss=", loss_value, "learning rate=", lr)
          #writer.add_summary(s, i)

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
