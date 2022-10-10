import tensorflow as tf
import numpy as np
from math import *
from simulacion_piston import axial_to_matrix
from simulacion_piston import cube_to_axial
from simulacion_piston import hex_to_pixel



def segment_numbering(nrings):
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

  return segment_ind

def stack_junctions(image, nrings, HEIGHT, CHANNELS):
  #Compute hexagon dimensionp
  image.set_shape([HEIGHT, HEIGHT, CHANNELS])
  hex_size = min( floor( (HEIGHT/2.0)/(1.0 + (nrings * 3.0/2.0)) ), floor( (HEIGHT / ((nrings*2.0) + 1.0))*(2.0/sqrt(3.0))*(1.0/2.0) ))
  hex_width = hex_size * 2.0
  hex_height = sqrt(3.0)/2.0 * hex_width
  crop_size = (2.0/4.0)*hex_size    #(2.0/4.0)*hex_size   #(3.0/4.0)*hex_size   #(3.0/2.0)*hex_size   #hex_height   #glimpse square crop

  counter = 0
  image_stack = []
  
  reverse_img = tf.image.flip_left_right(image)
  for dx in range(-nrings, nrings):
    for dy in range( max(-nrings, -dx-nrings), min(nrings, -dx+nrings)):
      dq, dr = cube_to_axial(dx,dy,-dx-dy)
      Ax, Ay = hex_to_pixel(dq, dr, hex_size)
      Bx, By = hex_to_pixel(dq,   dr-1, hex_size)
      Cx, Cy = hex_to_pixel(dq+1, dr-1, hex_size)

      #Compute location and size of the crop window
      center_width = (Ax + Bx + Cx)/3.0   #the origin is the center of the image
      center_height  = (Ay + By + Cy)/3.0  #the origin is the center of the image
      corner_height = center_height - (crop_size/2.0) + (HEIGHT/2.0)
      corner_width = center_width - (crop_size/2.0) + (HEIGHT/2.0)
  
      image_i = tf.image.crop_to_bounding_box(image, int(corner_height), int(corner_width), int(crop_size), int(crop_size))
      image_stack.append(image_i)

      #flipped counterpart:
      image_if = tf.image.crop_to_bounding_box(reverse_img, int(corner_height), int(corner_width), int(crop_size), int(crop_size))
      image_stack.append(image_if)

      counter = counter + 2

      
  image_stack = tf.stack(image_stack, axis=0)
  #tf.summary.image('images', image_stack)
  return image_stack



def stack_piston_steps(label, nrings):
  # use tf.data.Dataset.from_generator() instead in future versions
  #label.set_shape([(2*nrings)+1, (2*nrings)+1])
  NUM_CLASSES = 1
  counter = 0
  label_stack = [] 

  for dx in range(-nrings, nrings):
    for dy in range( max(-nrings, -dx-nrings), min(nrings, -dx+nrings)):
      dq, dr = cube_to_axial(dx,dy,-dx-dy)
      pA = label[axial_to_matrix(dq,   dr,   nrings)]  
      pB = label[axial_to_matrix(dq,   dr-1, nrings)]   #upper hexagon
      pC = label[axial_to_matrix(dq+1, dr-1, nrings)]   #upper RIGHT hexagon
      
      #label_i = np.stack([pA-pB, pB-pC, pC-pA], axis=0)    #for NUM_CLASSES = 3
      label_i = np.stack([pB-pA], axis=0)     #for NUM_CLASSES = 2
      
      label_i = np.reshape(label_i, [NUM_CLASSES])
      label_stack.append(label_i)

      #flipped counterpart:
      dqf, drf = cube_to_axial(-dx,-(-dx-dy), -dy)
      pAf = label[axial_to_matrix(dqf,   drf,   nrings)]  
      pBf = label[axial_to_matrix(dqf,   drf-1, nrings)]   #upper hexagon
      pCf = label[axial_to_matrix(dqf-1, drf,   nrings)]   #upper LEFT hexagon

      
      #label_if = np.stack([pAf-pBf, pBf-pCf, pCf-pAf], axis=0)    #for NUM_CLASSES = 3
      label_if = np.stack([pBf-pAf], axis=0)     #for NUM_CLASSES = 2

      label_if = np.reshape(label_if, [NUM_CLASSES])
      label_stack.append(label_if)

      counter = counter + 2

      
  label_stack = np.stack(label_stack, axis=0)

  return label_stack


