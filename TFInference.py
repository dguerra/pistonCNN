from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import datetime
#from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np
import os
import time

#More layers added
def inference(images):
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, 3, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)

    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))

    #for so-called "global normalization", used with convolutional filters with 
    #shape [batch, height, width, depth], pass axes=[0, 1, 2]. Otherwise axes=[0]
    batch_mean, batch_var = tf.nn.moments(conv,[0,1,2])
    scale = tf.Variable(tf.ones([64]))
    beta = tf.Variable(tf.zeros([64]))
    conv_norm = tf.nn.batch_normalization(conv,batch_mean,batch_var,beta,scale, 1e-3)

    pre_activation = tf.nn.bias_add(conv_norm, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv1)

  # pool1: 
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1') #Default ksize=[1, 3, 3, 1]
  # norm1
  #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32) #Default shape=[5, 5, 64, 64]
    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.1))

    batch_mean, batch_var = tf.nn.moments(conv,[0,1,2])
    scale = tf.Variable(tf.ones([64]))
    beta = tf.Variable(tf.zeros([64]))
    conv_norm = tf.nn.batch_normalization(conv,batch_mean,batch_var,beta,scale, 1e-3)

    pre_activation = tf.nn.bias_add(conv_norm, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv2)

  # norm2
  #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2:
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')  #Default ksize=[1, 3, 3, 1]

  ######  START NEW LAYERS:
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.1))

    batch_mean, batch_var = tf.nn.moments(conv,[0,1,2])
    scale = tf.Variable(tf.ones([64]))
    beta = tf.Variable(tf.zeros([64]))
    conv_norm = tf.nn.batch_normalization(conv,batch_mean,batch_var,beta,scale, 1e-3)

    pre_activation = tf.nn.bias_add(conv_norm, biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv3)

  with tf.variable_scope('conv4') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.1))

    batch_mean, batch_var = tf.nn.moments(conv,[0,1,2])
    scale = tf.Variable(tf.ones([64]))
    beta = tf.Variable(tf.zeros([64]))
    conv_norm = tf.nn.batch_normalization(conv,batch_mean,batch_var,beta,scale, 1e-3)

    pre_activation = tf.nn.bias_add(conv_norm, biases)
    conv4 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv3)

  # pool2
  pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
  ######  END NEW LAYERS:

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool4, [tf.app.flags.FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = tf.get_variable('weights', shape=[dim, 384], initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(weights), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    matmul = tf.matmul(reshape, weights)
    biases = tf.get_variable('biases', shape=[384], initializer=tf.constant_initializer(0.1))

    batch_mean, batch_var = tf.nn.moments(matmul,[0])
    scale = tf.Variable(tf.ones([384]))
    beta = tf.Variable(tf.zeros([384]))
    matmul_norm = tf.nn.batch_normalization(matmul,batch_mean,batch_var,beta,scale, 1e-3)    
    pre_activation = tf.nn.bias_add(matmul_norm, biases)

    local3 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = tf.get_variable('weights', shape=[384, 192], initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(weights), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    matmul = tf.matmul(local3, weights)
    biases = tf.get_variable('biases', shape=[192], initializer=tf.constant_initializer(0.1))

    batch_mean, batch_var = tf.nn.moments(matmul,[0])
    scale = tf.Variable(tf.ones([192]))
    beta = tf.Variable(tf.zeros([192]))
    matmul_norm = tf.nn.batch_normalization(matmul,batch_mean,batch_var,beta,scale, 1e-3)    
    pre_activation = tf.nn.bias_add(matmul_norm, biases)

    local4 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = tf.get_variable('weights', shape=[192, NUM_CLASSES], initializer=tf.truncated_normal_initializer(stddev=1/192.0, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(weights), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    matmul = tf.matmul(local4, weights)   
    biases = tf.get_variable('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))

    batch_mean, batch_var = tf.nn.moments(matmul,[0])
    scale = tf.Variable(tf.ones([NUM_CLASSES]))
    beta = tf.Variable(tf.zeros([NUM_CLASSES]))
    matmul_norm = tf.nn.batch_normalization(matmul,batch_mean,batch_var,beta,scale, 1e-3)    
    softmax_linear = tf.nn.bias_add(matmul_norm, biases)

    #_activation_summary(softmax_linear)

  return softmax_linear


#Four layers, no max_pool applied
def inference(images):
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, 3, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)

    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))

    #for so-called "global normalization", used with convolutional filters with 
    #shape [batch, height, width, depth], pass axes=[0, 1, 2]. Otherwise axes=[0]
    batch_mean, batch_var = tf.nn.moments(conv,[0,1,2])
    scale = tf.Variable(tf.ones([64]))
    beta = tf.Variable(tf.zeros([64]))
    conv_norm = tf.nn.batch_normalization(conv,batch_mean,batch_var,beta,scale, 1e-3)

    pre_activation = tf.nn.bias_add(conv_norm, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv1)

  # pool1: 
  #pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
  # norm1
  #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.1))

    batch_mean, batch_var = tf.nn.moments(conv,[0,1,2])
    scale = tf.Variable(tf.ones([64]))
    beta = tf.Variable(tf.zeros([64]))
    conv_norm = tf.nn.batch_normalization(conv,batch_mean,batch_var,beta,scale, 1e-3)

    pre_activation = tf.nn.bias_add(conv_norm, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv2)

  # norm2
  #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2:
  #pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')


  ####### START NEW LAYERS #######
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.1))

    batch_mean, batch_var = tf.nn.moments(conv,[0,1,2])
    scale = tf.Variable(tf.ones([64]))
    beta = tf.Variable(tf.zeros([64]))
    conv_norm = tf.nn.batch_normalization(conv,batch_mean,batch_var,beta,scale, 1e-3)

    pre_activation = tf.nn.bias_add(conv_norm, biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv2)

  # norm3
  #norm3 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
  # pool3:
  # pool3 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

  with tf.variable_scope('conv4') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.1))

    batch_mean, batch_var = tf.nn.moments(conv,[0,1,2])
    scale = tf.Variable(tf.ones([64]))
    beta = tf.Variable(tf.zeros([64]))
    conv_norm = tf.nn.batch_normalization(conv,batch_mean,batch_var,beta,scale, 1e-3)

    pre_activation = tf.nn.bias_add(conv_norm, biases)
    conv4 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv2)

  # norm3
  #norm3 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
  # pool3:
  #pool4 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
  ####### END NEW LAYERS #######

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(conv4, [tf.app.flags.FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = tf.get_variable('weights', shape=[dim, 384], initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(weights), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    matmul = tf.matmul(reshape, weights)
    biases = tf.get_variable('biases', shape=[384], initializer=tf.constant_initializer(0.1))

    batch_mean, batch_var = tf.nn.moments(matmul,[0])
    scale = tf.Variable(tf.ones([384]))
    beta = tf.Variable(tf.zeros([384]))
    matmul_norm = tf.nn.batch_normalization(matmul,batch_mean,batch_var,beta,scale, 1e-3)    
    pre_activation = tf.nn.bias_add(matmul_norm, biases)

    local3 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = tf.get_variable('weights', shape=[384, 192], initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(weights), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    matmul = tf.matmul(local3, weights)
    biases = tf.get_variable('biases', shape=[192], initializer=tf.constant_initializer(0.1))

    batch_mean, batch_var = tf.nn.moments(matmul,[0])
    scale = tf.Variable(tf.ones([192]))
    beta = tf.Variable(tf.zeros([192]))
    matmul_norm = tf.nn.batch_normalization(matmul,batch_mean,batch_var,beta,scale, 1e-3)    
    pre_activation = tf.nn.bias_add(matmul_norm, biases)

    local4 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = tf.get_variable('weights', shape=[192, NUM_CLASSES], initializer=tf.truncated_normal_initializer(stddev=1/192.0, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(weights), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    matmul = tf.matmul(local4, weights)   
    biases = tf.get_variable('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))

    batch_mean, batch_var = tf.nn.moments(matmul,[0])
    scale = tf.Variable(tf.ones([NUM_CLASSES]))
    beta = tf.Variable(tf.zeros([NUM_CLASSES]))
    matmul_norm = tf.nn.batch_normalization(matmul,batch_mean,batch_var,beta,scale, 1e-3)    
    softmax_linear = tf.nn.bias_add(matmul_norm, biases)

    #_activation_summary(softmax_linear)

  return softmax_linear


#Default configuration
def inference(images):
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, 3, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)

    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))

    #for so-called "global normalization", used with convolutional filters with 
    #shape [batch, height, width, depth], pass axes=[0, 1, 2]. Otherwise axes=[0]
    batch_mean, batch_var = tf.nn.moments(conv,[0,1,2])
    scale = tf.Variable(tf.ones([64]))
    beta = tf.Variable(tf.zeros([64]))
    conv_norm = tf.nn.batch_normalization(conv,batch_mean,batch_var,beta,scale, 1e-3)

    pre_activation = tf.nn.bias_add(conv_norm, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv1)

  # pool1: 
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
  # norm1
  #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = tf.get_variable('weights', shape=[5, 5, 64, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.1))

    batch_mean, batch_var = tf.nn.moments(conv,[0,1,2])
    scale = tf.Variable(tf.ones([64]))
    beta = tf.Variable(tf.zeros([64]))
    conv_norm = tf.nn.batch_normalization(conv,batch_mean,batch_var,beta,scale, 1e-3)

    pre_activation = tf.nn.bias_add(conv_norm, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv2)

  # norm2
  #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2:
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [tf.app.flags.FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = tf.get_variable('weights', shape=[dim, 384], initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(weights), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    matmul = tf.matmul(reshape, weights)
    biases = tf.get_variable('biases', shape=[384], initializer=tf.constant_initializer(0.1))

    batch_mean, batch_var = tf.nn.moments(matmul,[0])
    scale = tf.Variable(tf.ones([384]))
    beta = tf.Variable(tf.zeros([384]))
    matmul_norm = tf.nn.batch_normalization(matmul,batch_mean,batch_var,beta,scale, 1e-3)    
    pre_activation = tf.nn.bias_add(matmul_norm, biases)

    local3 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = tf.get_variable('weights', shape=[384, 192], initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(weights), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    matmul = tf.matmul(local3, weights)
    biases = tf.get_variable('biases', shape=[192], initializer=tf.constant_initializer(0.1))

    batch_mean, batch_var = tf.nn.moments(matmul,[0])
    scale = tf.Variable(tf.ones([192]))
    beta = tf.Variable(tf.zeros([192]))
    matmul_norm = tf.nn.batch_normalization(matmul,batch_mean,batch_var,beta,scale, 1e-3)    
    pre_activation = tf.nn.bias_add(matmul_norm, biases)

    local4 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = tf.get_variable('weights', shape=[192, NUM_CLASSES], initializer=tf.truncated_normal_initializer(stddev=1/192.0, dtype=tf.float32), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(weights), WEIGHT_DECAY_FACTOR, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    matmul = tf.matmul(local4, weights)   
    biases = tf.get_variable('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))

    batch_mean, batch_var = tf.nn.moments(matmul,[0])
    scale = tf.Variable(tf.ones([NUM_CLASSES]))
    beta = tf.Variable(tf.zeros([NUM_CLASSES]))
    matmul_norm = tf.nn.batch_normalization(matmul,batch_mean,batch_var,beta,scale, 1e-3)    
    softmax_linear = tf.nn.bias_add(matmul_norm, biases)

    #_activation_summary(softmax_linear)

  return softmax_linear

