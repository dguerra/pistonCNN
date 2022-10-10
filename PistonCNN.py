from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import datetime

from simulacion_piston import axial_to_matrix
from simulacion_piston import cube_to_axial
from simulacion_piston import hex_to_pixel
#from matplotlib import pyplot as plt
from scipy.stats import norm
from math import *
import tensorflow as tf
import numpy as np
import os
import time
from Mirror import Mirror

from tensorflow.python.framework import ops

max_steps = 1000000
#LOGDIR = "/output/phasedivcnn/1"
LOGDIR = ".."

#floyd login
#floyd init phasedivcnn
#floyd run --tensorboard --cpu --env tensorflow-1.13 "python PistonCNN.py"

            
NUM_CLASSES = 3   #number of values to be detected per image
HEIGHT = 1024  #1024 #number of samples in each direction in the pupil plane [pixels]
INITIAL_LEARNING_RATE = 0.0006638080894086971   # 10**np.random.uniform(-1.0,-3.0) #Default: (-3,-6)   # Fine tuning: (-1,-8)
CHANNELS = 4
LAMBDA_RANGES = 11  #19  #only odd number of lambda ranges are allowed!! (1,3,5,7...)
if LAMBDA_RANGES % 2 == 0:
  raise NameError('LAMBDA_RANGES cannot be an even number. Only odd numbers are allowed.')


nrings = 3    #number of hexagon rings in the simmulation
time_steps = 6*(nrings**2)   #Number of images to process in a sequence. Number of intersections annalised in the whole segmented mirror

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
  reverse_img = tf.image.flip_left_right(image)
  for dx in range(-nrings, nrings):
    for dy in range( max(-nrings, -dx-nrings), min(nrings, -dx+nrings)):
      dq, dr = cube_to_axial(dx,dy,-dx-dy)
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
      
      label_i = tf.stack([pA-pB, pB-pC, pC-pA], axis=0)    #for NUM_CLASSES = 3
      #label_i = tf.stack([pB-pC, pC-pA], axis=0)     #for NUM_CLASSES = 2
      
      label_i.set_shape([NUM_CLASSES])
      label_stack.append(label_i)
      image_stack.append(image_i)

      #flipped counterpart:
      dqf, drf = cube_to_axial(-dx,-(-dx-dy), -dy)
      pAf = label[axial_to_matrix(dqf,   drf,   nrings)]  
      pBf = label[axial_to_matrix(dqf,   drf-1, nrings)]   #upper hexagon
      pCf = label[axial_to_matrix(dqf-1, drf,   nrings)]   #upper LEFT hexagon

      image_if = tf.image.crop_to_bounding_box(reverse_img, int(corner_height), int(corner_width), int(crop_size), int(crop_size))

      #better with this function but I couldn't make it work!!!!: tf.image.extract_glimpse(input, size, offsets, centered=True, normalized=True, uniform_noise=True, name=None)
      
      label_if = tf.stack([pAf-pBf, pBf-pCf, pCf-pAf], axis=0)    #for NUM_CLASSES = 3
      #label_if = tf.stack([pBf-pCf, pCf-pAf], axis=0)     #for NUM_CLASSES = 2

      label_if.set_shape([NUM_CLASSES])
      label_stack.append(label_if)
      image_stack.append(image_if)

      counter = counter + 2


      
  label_stack = tf.stack(label_stack, axis=0)
  image_stack = tf.stack(image_stack, axis=0)

  if counter != time_steps:
    print('Time steps must match %d' % counter)

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


def compute_loss(logits_r, logits_c, labels_p):
  with tf.name_scope("loss_xx"):
    scaled = tf.divide(tf.add(labels_p, ((LAMBDA_RANGES*2.0)) * pi ), pi)
    lambda_range = tf.floor(scaled)
    lambda_val = tf.multiply(tf.subtract(scaled,lambda_range), pi)   #all values between 0 and +2.0*pi
    even = tf.equal( tf.mod(lambda_range, 2), 0.0)
    lambda_val = tf.where(even, lambda_val, tf.subtract(pi, lambda_val))

    loss_c = tf.losses.sparse_softmax_cross_entropy(tf.reshape(tf.cast(lambda_range, tf.int32), [1 * NUM_CLASSES * time_steps]), logits_c, weights=10.0, loss_collection=tf.GraphKeys.LOSSES)
    loss_r = tf.losses.mean_squared_error(lambda_val, logits_r, weights=1.0, loss_collection=tf.GraphKeys.LOSSES)       # tf.constant(0.0)
    
    pred_lambda_range = tf.reshape(tf.argmax(logits_c, axis=1), [1, NUM_CLASSES * time_steps])
    accuracy_c = tf.reduce_sum( tf.cast(tf.equal(pred_lambda_range, tf.cast(lambda_range,tf.int64)),tf.int64)) / (1*NUM_CLASSES*time_steps)
    pred_even = tf.equal( tf.mod(tf.cast(pred_lambda_range, tf.float32), 2), 0.0)
    pred_lambda_val = tf.where(pred_even, logits_r, tf.subtract(pi, logits_r))

    final_pstn = tf.subtract(tf.add(tf.multiply(tf.cast(pred_lambda_range, tf.float32), pi), pred_lambda_val), ((LAMBDA_RANGES*2.0)) * pi)

    loss_p =  tf.losses.mean_squared_error(labels_p, final_pstn, loss_collection=None)
    tf.summary.scalar('loss', loss_p)

    return tf.losses.get_total_loss(add_regularization_losses=True, name='total_loss'), loss_p, loss_r, accuracy_c, final_pstn


def svd_solve(a, b):
    u, s, v_adj = np.linalg.svd(a, full_matrices=False)
    tol_indx = np.zeros_like(s, dtype=int)
    tol_indx[np.where(s >= 1e-16)] = 1
    r = np.sum(tol_indx) - 1    #####This minus one thing is weird... it does not seem to be necessary elsewhere
    return np.dot(v_adj[:r, :].T, np.dot(u[:, :r].T, b) / s[:r])

def final_loss(B, pstn_v):

  label_total_stack = []
  #go through hexagons within N rings
  for dx in range(-nrings, nrings+1):
    for dy in range( max(-nrings, -dx-nrings), min(nrings, -dx+nrings)+1):
      dz = -dx-dy
      dq, dr = cube_to_axial(dx,dy,dz)
      di, dj = axial_to_matrix(dq, dr, nrings)
      label_total_stack.append( pstn_v[di, dj] )
  label_total_stack = np.stack(label_total_stack, axis=0)
  #calculate number of segments
  nsegs = (3 * nrings * (nrings + 1)) + 1
  AA = np.zeros([time_steps * NUM_CLASSES, nsegs])
  consistency_threshold = 0.5
  B_check = np.reshape(B, [time_steps, NUM_CLASSES])
  consistency_check = np.absolute ( np.sum(B_check, axis=1) )
  ts = 0
  for dx in range(-nrings, nrings):
    for dy in range( max(-nrings, -dx-nrings), min(nrings, -dx+nrings)):
      if consistency_check[ts] < consistency_threshold:
        dq, dr = cube_to_axial(dx,dy,-dx-dy)
        indA = int(segment_ind[axial_to_matrix(dq,   dr,   nrings)])
        indB = int(segment_ind[axial_to_matrix(dq,   dr-1, nrings)])   #upper hexagon
        indC = int(segment_ind[axial_to_matrix(dq+1, dr-1, nrings)])  #upper RIGHT hexagon
        
        # xx.stack([pA-pB, pB-pC, pC-pA], axis=0)    #for NUM_CLASSES = 3
        AA[(ts*NUM_CLASSES)+0, indA] =  1.0
        AA[(ts*NUM_CLASSES)+0, indB] = -1.0

        AA[(ts*NUM_CLASSES)+1, indB] =  1.0
        AA[(ts*NUM_CLASSES)+1, indC] = -1.0

        AA[(ts*NUM_CLASSES)+2, indC] =  1.0
        AA[(ts*NUM_CLASSES)+2, indA] = -1.0

      ts += 1

      
      if consistency_check[ts] < consistency_threshold:
        #flipped counterpart:
        dqf, drf = cube_to_axial(-dx,-(-dx-dy), -dy)
        indAf = int(segment_ind[axial_to_matrix(dqf,   drf,   nrings)])  
        indBf = int(segment_ind[axial_to_matrix(dqf,   drf-1, nrings)])   #upper hexagon
        indCf = int(segment_ind[axial_to_matrix(dqf-1, drf,   nrings)])   #upper LEFT hexagon
        
        # xx.stack([pAf-pBf, pBf-pCf, pCf-pAf], axis=0)    #for NUM_CLASSES = 3
        AA[(ts*NUM_CLASSES)+0, indAf] =  1.0
        AA[(ts*NUM_CLASSES)+0, indBf] = -1.0

        AA[(ts*NUM_CLASSES)+1, indBf] =  1.0
        AA[(ts*NUM_CLASSES)+1, indCf] = -1.0

        AA[(ts*NUM_CLASSES)+2, indCf] =  1.0
        AA[(ts*NUM_CLASSES)+2, indAf] = -1.0

      ts += 1

  #print("A shape ", AA.shape, "B shape ", B.shape)
  #print(AA)
  #print(B)
  x = 0
  rnk = np.linalg.matrix_rank(AA)
  if rnk >= (nsegs-1):
    B = np.reshape(B, [-1])
    x = svd_solve(AA,B)
  
    #be suere piston values are measured with respect to the central hexagon
    x = x - x[int(segment_ind[axial_to_matrix(0, 0, nrings)])]
    floss = np.mean(np.absolute(label_total_stack - x)**2.0)
    #print("label_total_stack ", label_total_stack)
    #print("x ", x)
  else:
    floss = np.nan
  return floss, rnk, x, np.count_nonzero(np.sum(np.absolute(AA), axis=1))
 

ops.reset_default_graph()
image_p = tf.placeholder(tf.float32, shape=(HEIGHT, HEIGHT, CHANNELS))
label_p = tf.placeholder(tf.float32, shape=((2*nrings)+1, (2*nrings)+1)) 

images, labels_pst = inputs(image_p, label_p)

conv_cls = inference_cnn(images, "classification", num_filters=16)
conv_rgr = inference_cnn(images, "regression", num_filters=16)  #default 64    # time_steps, HEIGHT, HEIGHT, CHANNEL; tf.slice(images, [0, 0, 0, 0], [-1, -1, -1, 1]  

conv_cls = tf.contrib.layers.flatten(conv_cls)
conv_rgr = tf.contrib.layers.flatten(conv_rgr)
  
#Approach:
logits_cls = tf.layers.dense(inputs=conv_cls, units=int((LAMBDA_RANGES*4) * NUM_CLASSES), name="fc_cls", activation=None, use_bias=True, \
  kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer() )

logits_rgr = tf.layers.dense(inputs=conv_rgr, units=NUM_CLASSES, name="fc_rgr", activation=None, use_bias=True, \
  kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer() )

fco_rgr = tf.reshape(logits_rgr, [1, NUM_CLASSES * time_steps])
fco_rgr = tf.multiply( tf.sigmoid(fco_rgr), pi) # if it desired to enforce the output to be in certain range

fco_cls = tf.reshape(logits_cls, [NUM_CLASSES * time_steps, int(LAMBDA_RANGES*4)])
labels_pst = tf.reshape(labels_pst, [1, NUM_CLASSES * time_steps])
 
loss, loss_mse, the_loss, wrong_ranges, finalpstn = compute_loss(fco_rgr, fco_cls, labels_pst)

# Build a Graph that trains the model with one batch of examples and updates the model parameters
train_op = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(loss)
  
sess = tf.Session()  
summ = tf.summary.merge_all()
  
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
    
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)
   
for step in range(max_steps):
    # Run training steps or whatever
    pstn = np.random.uniform(-LAMBDA_RANGES * pi, LAMBDA_RANGES * pi, [(2*nrings)+1, (2*nrings)+1])     # np.zeros([(2*nrings)+1, (2*nrings)+1])
    #pstn = np.random.uniform(-pi/2.0, pi/2.0, [(2*nrings)+1, (2*nrings)+1])
    #pstn = pstn - pstn[axial_to_matrix(0, 0, nrings)]    #set reference to central hexagon piston value
    pstn[axial_to_matrix(0, 0, nrings)] = 0.0
    image_p_feed = mirror.create_data(pstn)
 
    feed_dict = {
      image_p: image_p_feed,
      label_p: pstn   #np.array(pstn, ndmin=2, dtype=np.float32),
    }

    _, loss_value, s, l_mse, the_l, w_ranges, fpstn = sess.run([train_op, loss, summ, loss_mse,  the_loss, wrong_ranges, finalpstn], feed_dict=feed_dict)
  
    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
  
    if step % 10 == 0:
      #print(datetime.now(), "global_step=", step, "loss=", loss_value, "learning rate=", lr)
      f_loss_, rnk_, xx, A_shape = final_loss(fpstn, pstn)
      print("*"+str(step)+"*"+str(l_mse).replace(".", ",")+"*"+str(the_l).replace(".", ",")+"*"+str(w_ranges).replace(".", ",")+"*"+str(f_loss_).replace(".", ",")+"*"+str(rnk_).replace(".", ",")+"*"+str(A_shape).replace(".", ",")+"*" )

      #saver.save(sess, '/tmp/model.ckpt')
      writer.add_summary(s, step)

    if step == max_steps:
      coord.request_stop()



'''
#Restore a model:
sess = tf_reset()
input_ph, output_ph, output_pred = create_model()
saver = tf.train.Saver()
saver.restore(sess,'tmp/model.ckpt')
ouput_pred_run = sess.run(outpuy_pred, feed_dict={input_ph: inputs})
'''

