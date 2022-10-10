from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import numpy as np
import os


from tensorflow.python.framework import ops

ops.reset_default_graph()
  
one = tf.constant(1)
global_step = tf.Variable(0, trainable=False)
plus_one = tf.add(one,global_step)
update = tf.assign(global_step, plus_one)

filenames = ['data_batch_%d.bin' % i for i in xrange(1, 6)]
filenames_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)
name = filenames_queue.dequeue()

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
  while not coord.should_stop():
    # Run training steps or whatever
    val, na = sess.run([update, name])
    print(val)
    print(na)
    if val == 100:  #
      coord.request_stop()


except tf.errors.OutOfRangeError:
  print('Done training -- epoch limit reached')
finally:
  # When done, ask the threads to stop.
  coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
