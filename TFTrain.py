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


from tensorflow.python.framework import ops

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './phasediv_data',
                           """Path to the phase div data directory.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
                            
                           
NUM_CLASSES = 1
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
#NUM_EPOCHS_FOR_CROSSVALIDATION

def inputs(data_dir, batch_size):
  #filenames = [os.path.join(data_dir, 'data_batch128_%d.bin' % i)
  #             for i in xrange(1, 2)]

  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]
  
  for f in filenames:               
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  
  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  class PhaseDivRecord(object):
    pass
  record = PhaseDivRecord()
  
  datatype_size = 4   #Four bytes for float 32
  label_bytes = NUM_CLASSES   #Number of classes to classify
  record.height = 40  #32 Image height
  record.width = 40  #32 Image width
  record.depth = 2  # Number of channels in the image
  image_bytes = record.height * record.width * record.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes*datatype_size)

  # Get record identifier and raw record bytes
  record.key, value = reader.read(filename_queue)
  
  # Convert from a string to a vector of float32 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.float32, little_endian=True)

  # The first bytes represent the label, which we convert from uint8->int32.
  record.label = tf.slice(record_bytes, [0], [label_bytes])
  
  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                                 [record.depth, record.height, record.width])
  
  # Convert from [depth, height, width] to [height, width, depth].
  record.float32image = tf.transpose(depth_major, [1, 2, 0])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 
                           min_fraction_of_examples_in_queue)
  
  #print ('Wait while it fills up the queue with %d phase div examples. ' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  num_preprocess_threads = 16
  # use tf.train.batch() instead if no shuffling needed
  images, label_batch = tf.train.shuffle_batch(
        [record.float32image, record.label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)


  # Display the training images in the visualizer.
  #vis_imgs = tf.cast (tf.div(tf.mul( tf.sub( images, tf.reduce_min(images)), 255.0), 
  #                                   tf.sub(tf.reduce_max(images),tf.reduce_min(images) )), tf.uint8)
  #tf.image_summary('images', vis_imgs)

  return images, tf.reshape(label_batch, [batch_size, NUM_CLASSES])

#Implementation with layers:
def inference(images):

  # conv 1   #tf.truncated_normal_initializer(stddev=5e-2)    #have a look to separable_conv2d
  conv1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d() )
  #pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='SAME', name='pool1')

  conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='SAME', name='pool2')

  conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
  #pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, padding='SAME', name='pool3')

  convN = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
  poolN = tf.layers.max_pooling2d(inputs=convN, pool_size=[2, 2], strides=2, padding='SAME', name='poolN')
 
  poolN_flat = tf.reshape(poolN, [tf.app.flags.FLAGS.batch_size, -1])    #tf.reshape(pool2, [-1, 10*10*64])   #tf.app.flags.FLAGS.batch_size
  local3 = tf.layers.dense(inputs=poolN_flat, units=384, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

  local4 = tf.layers.dense(inputs=local3, units=192, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
 
  logits = tf.layers.dense(inputs=local4, units=NUM_CLASSES, kernel_initializer=tf.zeros_initializer())

  return logits


def compute_loss(logits, labels):
  #L2 loss, mean squared error:   lr candidate=0.00043725070
  mean_square = tf.reduce_sum(tf.pow(labels-logits, 2.0))/(2.0*tf.app.flags.FLAGS.batch_size)
  #L1 loss:
  #mean_square = tf.reduce_sum(tf.abs(labels-logits))/(tf.app.flags.FLAGS.batch_size)   #tf.reduce_mean(tf.abs(labels-logits))

  loss = tf.reduce_mean(mean_square, name='cross_entropy')
  tf.add_to_collection('losses', loss)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, global_step):
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / tf.app.flags.FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)

  # Compute gradients.
  opt = tf.train.AdamOptimizer(lr)
  #opt = tf.train.GradientDescentOptimizer(lr)
  grads = opt.compute_gradients(total_loss)

  # Apply gradients and increments global_step by one
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
  with tf.control_dependencies([apply_gradient_op]):   #do not call train_op until apply_gradient_op finishes
    train_op = tf.no_op(name='train')
  return train_op



max_count = 400   #Do several experiment to try out different hyperparameter configurations
for count in xrange(max_count):

  ops.reset_default_graph()
  global_step = tf.Variable(0, trainable=False)
  
  
  INITIAL_LEARNING_RATE = tf.placeholder(shape=[], dtype=tf.float32, name="INITIAL_LEARNING_RATE")
  WEIGHT_DECAY_FACTOR =  tf.placeholder(shape=[], dtype=tf.float32, name="WEIGHT_DECAY_FACTOR")
  NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000   # whole data set size used for training 400000

  # Get next image and label batch from the queue
  images, labels = inputs(tf.app.flags.FLAGS.data_dir, tf.app.flags.FLAGS.batch_size)

  # Build a Graph that computes the logits predictions from the inference model
  logits = inference(images)

  # Calculate loss
  loss = compute_loss(logits, labels)

  # Build a Graph that trains the model with one batch of examples and updates the model parameters
  train_op = train(loss, global_step)

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
    lr = 10**np.random.uniform(-3,-6)     #10**np.random.uniform(-1,-8)   #0.07   #coarse search: (-3,-6)
    reg = 0.0   #10**np.random.uniform(-5,5) #sample when crossvalidating
    while not coord.should_stop():
      # Run training steps or whatever
      _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={INITIAL_LEARNING_RATE: lr, WEIGHT_DECAY_FACTOR: reg})
      
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step == 1562:        #step == 50: (crossvalidation)    #step % 10 == 0: (training)
        print(datetime.now(), "global_step=", step, "loss=", loss_value, "learning rate=", lr)

      if step == 1562:        #step == 50: (crossvalidation)      #step == tf.app.flags.FLAGS.max_steps:  (training)
        coord.request_stop()
  
  
  except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
  finally:
    # When done, ask the threads to stop.
    coord.request_stop()
  
  # Wait for threads to finish.
  coord.join(threads)
  sess.close()
