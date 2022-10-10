from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
#from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np
import os

from tensorflow.python.framework import ops

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './phasediv_data',
                           """Path to the phase div data directory.""")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 128
def inputs(data_dir, batch_size):
  filenames = [os.path.join(data_dir, 'data_batch128_%d.bin' % i)
               for i in xrange(1, 2)]
  
  for f in filenames:               
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  
  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  class PhaseDivRecord(object):
    pass
  record = PhaseDivRecord()
  
  datatype_size = 4   #Four bytes for float 32
  label_bytes = 10   #Ten classes
  record.height = 40  #32 Image height
  record.width = 40  #32 Image width
  record.depth = 3  # Number of channels in the image
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

  # Ensure that the random shuffling has good mixing properties.Clinicana
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Wait while it fills up the queue with %d phase div examples. ' % min_queue_examples)

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

  return images, tf.reshape(label_batch, [batch_size, 10])
  



# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

max_count = 50   #Do several experiment to try out different hyperparameter configurations
for count in xrange(max_count):

  ops.reset_default_graph()
    
  # Get images and labels for CIFAR-10.
  images, labels = inputs(tf.app.flags.FLAGS.data_dir, tf.app.flags.FLAGS.batch_size)
  # tf.slice(input_, begin, size, name=None)
  tshape = tf.shape(tf.slice(images, [0,0,0,0],[1,-1,-1,-1]))
  #plt.imshow(images, interpolation='nearest')
  #plt.show()

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
      _, step_n, loss_val, img_dim = sess.run([train, update, loss, tshape], feed_dict={INITIAL_LEARNING_RATE: lr})
      print("loss=", loss_val, "learning rate=", lr)
      print("shape: ", img_dim, "step: ", step_n)
      if step_n == 2:  #
        coord.request_stop()
  
  
  except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
  finally:
    # When done, ask the threads to stop.
    coord.request_stop()
  
  # Wait for threads to finish.
  coord.join(threads)
  sess.close()
