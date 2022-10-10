from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin

from simulacion_piston import axial_to_matrix
from simulacion_piston import cube_to_axial
from simulacion_piston import hex_to_pixel
from math import *
import tensorflow as tf
import numpy as np
from Mirror import Mirror

from tensorflow.python.framework import ops

max_steps = 1000000
#LOGDIR = "/output/phasedivcnn/1"
LOGDIR = ".."

#floyd login
#floyd init phasedivcnn
#floyd run --tensorboard --cpu --env tensorflow-1.13 "python PistonCNN.py"

            
NUM_CLASSES = 2   #number of values to be detected per image
HEIGHT = 256  #1024 #number of samples in each direction in the pupil plane [pixels]
INITIAL_LEARNING_RATE = 0.0006638080894086971   # 10**np.random.uniform(-1.0,-3.0) #Default: (-3,-6)   # Fine tuning: (-1,-8)
CHANNELS = 4
LAMBDA_RANGES = 1  #19  #only odd number of lambda ranges are allowed!! (1,3,5,7...)
if LAMBDA_RANGES % 2 == 0:
  raise NameError('LAMBDA_RANGES cannot be an even number. Only odd numbers are allowed.')


nrings = 1    #number of hexagon rings in the simmulation
time_steps = 1   #6*(nrings**2)   #Number of images to process in a sequence. Number of intersections annalised in the whole segmented mirror

mirror = Mirror(nrings, HEIGHT, CHANNELS, LAMBDA_RANGES)

segment_ind = np.empty([(2*nrings)+1, (2*nrings)+1])
segment_ind[:] = np.nan

ind = 0
for dx in range(-nrings, nrings+1):
  for dy in range( max(-nrings, -dx-nrings), min(nrings, -dx+nrings)+1):
    dz = -dx-dy
    dq, dr = cube_to_axial(dx,dy,dz)
    di, dj = axial_to_matrix(dq, dr, nrings)
    segment_ind[di, dj] = ind
    ind += 1

print(segment_ind)


def inputs(image, label):
  # use tf.data.Dataset.from_generator() instead in future versions
  image.set_shape([HEIGHT, HEIGHT, CHANNELS])
  label.set_shape([(2*nrings)+1, (2*nrings)+1])
  
  # Generate a batch of images and labels by building up a queue of examples.
  num_preprocess_threads = 16

  #Compute hexagon dimensionp
  hex_size = min( floor( (HEIGHT/2.0)/(1.0 + (nrings * 3.0/2.0)) ), floor( (HEIGHT / ((nrings*2.0) + 1.0))*(2.0/sqrt(3.0))*(1.0/2.0) ))
  hex_width = hex_size * 2.0
  hex_height = sqrt(3.0)/2.0 * hex_width
  crop_size = (2.0/4.0)*hex_size    #(2.0/4.0)*hex_size   #(3.0/4.0)*hex_size   #(3.0/2.0)*hex_size   #hex_height   #glimpse square crop

  counter = 0
  image_stack = []
  
  label_stack = [] 

  dq = 0
  dr = 0
  pA = label[axial_to_matrix(dq,   dr,   nrings)]  
  pB = label[axial_to_matrix(dq,   dr-1, nrings)]   #upper hexagon
  pC = label[axial_to_matrix(dq+1, dr-1, nrings)]   #upper RIGHT hexagon
  Ax, Ay = hex_to_pixel(dq, dr, hex_size)
  Bx, By = hex_to_pixel(dq,   dr-1, hex_size)
  Cx, Cy = hex_to_pixel(dq+1, dr-1, hex_size)

  #Compute location and size of the crop window
  center_width = (Ax + Bx + Cx)/3.0   #the origin is the center of the image
  center_height  = (Ay + By + Cy)/3.0  #the origin is the center of the image
  corner_height = center_height - (crop_size/2.0) + (HEIGHT/2.0)
  corner_width = center_width - (crop_size/2.0) + (HEIGHT/2.0)
  
  image_i = tf.image.crop_to_bounding_box(image, int(corner_height), int(corner_width), int(crop_size), int(crop_size))

  #better with this function but I couldn't make it work!!!!: tf.image.extract_glimpse(input, size, offsets, centered=True, normalized=True, uniform_noise=True, name=None)
      
  #label_i = tf.stack([pA-pB, pB-pC, pC-pA], axis=0)    #for NUM_CLASSES = 3
  label_i = tf.stack([pB-pC, pC-pA], axis=0)     #for NUM_CLASSES = 2
  #label_i = tf.stack([pA-pB], axis=0)    #for NUM_CLASSES = 1
     
  label_i.set_shape([NUM_CLASSES])
  label_stack.append(label_i)
  image_stack.append(image_i)
      
  label_stack = tf.stack(label_stack, axis=0)
  image_stack = tf.stack(image_stack, axis=0)

  # Display the training images in the visualizer.
  tf.summary.image('images', image_stack)
  #return images, tf.reshape(label_batch, [time_steps, NUM_CLASSES]), label_ttt_stack 
  return image_stack, label_stack


def inference_cnn(input_cnn, branch_name, num_filters=32):  
  conv1 = tf.layers.conv2d(inputs=input_cnn, filters=num_filters, kernel_size=7, activation=tf.nn.relu, use_bias=True, \
    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer(), name="conv1"+branch_name)
  
  conv2 = tf.layers.conv2d(inputs=conv1, filters=num_filters, kernel_size=5, activation=tf.nn.relu, use_bias=True, \
    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer(), name="conv2"+branch_name)

  conv3 = tf.layers.conv2d(inputs=conv2, filters=num_filters, kernel_size=3, activation=tf.nn.relu, use_bias=True, \
    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer(), name="conv3"+branch_name)

  output_cnn = tf.layers.max_pooling2d(conv3, 2, 2)
  
  return output_cnn


def compute_loss(logits_c, labels_p):
  with tf.name_scope("loss_xx"):
    scaled = tf.divide(tf.add(labels_p, ((LAMBDA_RANGES*2.0)) * pi ), pi)
    lambda_range = tf.floor(scaled)
    
    logits_c = tf.nn.softmax(logits_c)   #convert into probabilities

    '''
    #case supervised:
    pred_lambda_range = tf.reshape(tf.argmax(logits_c, axis=1), [1, NUM_CLASSES * time_steps])
    loss_c = tf.losses.sparse_softmax_cross_entropy(tf.reshape(tf.cast(lambda_range, tf.int32), [1 * NUM_CLASSES * time_steps]), tf.log(logits_c), weights=10.0, loss_collection=tf.GraphKeys.LOSSES)    
    accuracy_c = tf.reduce_sum( tf.cast(tf.equal(pred_lambda_range, tf.cast(lambda_range,tf.int64)),tf.int64)) / (1*NUM_CLASSES*time_steps)
    '''

    #case reinforcement:
    #loss_c = tf.losses.sparse_softmax_cross_entropy(tf.reshape(tf.cast(lambda_range, tf.int32), [1 * NUM_CLASSES * time_steps]), tf.log(logits_c), weights=10.0, loss_collection=None)    
    chosen_lambda_range = tf.reshape(tf.random.categorical(tf.log(logits_c), 1), [NUM_CLASSES * time_steps, 1])      #size = [NUM_CLASSES * time_steps, 1]
    #chosen_lambda_range = tf.reshape(tf.argmax(logits_c, axis=1), [NUM_CLASSES * time_steps, 1])   
  
    loss_c = tf.losses.mean_squared_error(tf.reshape(chosen_lambda_range, [1 * NUM_CLASSES * time_steps]),tf.reshape(tf.cast(lambda_range, tf.int32), [1 * NUM_CLASSES * time_steps]), loss_collection=None)
    #tf.assert(tf.less_equal(chosen_lambda_range, 4) )
    chosen_prob = tf.batch_gather(logits_c, chosen_lambda_range)
    chosen_prob = tf.reshape(chosen_prob,[time_steps, NUM_CLASSES])
    loss_actor = -tf.log(tf.reduce_prod(chosen_prob, axis=1) + 1e-5 ) * (-loss_c + 0.5)    
    chosen_lambda_range = tf.reshape(chosen_lambda_range, [1, NUM_CLASSES * time_steps])
    accuracy_c = tf.reduce_sum( tf.cast(tf.equal(chosen_lambda_range, tf.cast(lambda_range,tf.int64)),tf.int64)) / (1*NUM_CLASSES*time_steps)
    

    #return tf.losses.get_total_loss(add_regularization_losses=True, name='total_loss'), accuracy_c
    return loss_actor, accuracy_c

ops.reset_default_graph()
image_p = tf.placeholder(tf.float32, shape=(HEIGHT, HEIGHT, CHANNELS))
label_p = tf.placeholder(tf.float32, shape=((2*nrings)+1, (2*nrings)+1)) 

images, labels_pst = inputs(image_p, label_p)

conv_cls = inference_cnn(images, "classification", num_filters=128)
conv_cls = tf.contrib.layers.flatten(conv_cls)
  
#Approach:
logits_cls = tf.layers.dense(inputs=conv_cls, units=int((LAMBDA_RANGES*4) * NUM_CLASSES), name="fc_cls", activation=None, use_bias=True, \
  kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer() )

fco_cls = tf.reshape(logits_cls, [NUM_CLASSES * time_steps, int(LAMBDA_RANGES*4)])
labels_pst = tf.reshape(labels_pst, [1, NUM_CLASSES * time_steps])
 
loss, wrong_ranges = compute_loss(fco_cls, labels_pst)

# Build a Graph that trains the model with one batch of examples and updates the model parameters
train_op = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(loss)
  
sess = tf.Session()  
summ = tf.summary.merge_all()
  
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
    
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)
w_total = 0.0
for step in range(max_steps):
    # Run training steps or whatever
    pstn = np.random.uniform(-LAMBDA_RANGES * pi, LAMBDA_RANGES * pi, [(2*nrings)+1, (2*nrings)+1])     # np.zeros([(2*nrings)+1, (2*nrings)+1])
    #pstn = np.zeros([(2*nrings)+1, (2*nrings)+1])
    #pstn[axial_to_matrix(0,   0,   nrings)] = np.random.uniform(-LAMBDA_RANGES * pi, LAMBDA_RANGES * pi)
    #pstn = np.random.uniform(-pi/2.0, pi/2.0, [(2*nrings)+1, (2*nrings)+1])
    #pstn = pstn - pstn[axial_to_matrix(0, 0, nrings)]    #set reference to central hexagon piston value
    #pstn[axial_to_matrix(0, 0, nrings)] = 0.0
    image_p_feed = mirror.create_data(pstn)
 
    feed_dict = {
      image_p: image_p_feed,
      label_p: pstn   #np.array(pstn, ndmin=2, dtype=np.float32),
    }

    _, loss_value, s, w_ranges = sess.run([train_op, loss, summ, wrong_ranges], feed_dict=feed_dict)
    w_total += w_ranges

    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
  
    if step % 10 == 0:
      #print(datetime.now(), "global_step=", step, "loss=", loss_value, "learning rate=", lr)
      print("*"+str(step)+"*"+"*"+str(w_total/10.0).replace(".", ","))
      w_total = 0.0

      #saver.save(sess, '/tmp/model.ckpt')
      writer.add_summary(s, step)

    if step == max_steps:
      coord.request_stop()

