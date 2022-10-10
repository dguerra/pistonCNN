from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from scipy.stats import uniform
from Mirror import Mirror
from math import *
import numpy as np
from simulacion_piston import axial_to_matrix
from simulacion_piston import cube_to_axial
from simulacion_piston import hex_to_pixel
from simulacion_piston import mirror_wavefront
import matplotlib.pyplot as plt
from foo import environment
from FourierOptics import propTF
from FourierOptics import focus
from Mirror import Mirror


physical_hex_size = 0.95
nrings = 3
HEIGHT = 1024
hex_size = min( floor( (HEIGHT/2.0)/(1.0 + (nrings * 3.0/2.0)) ), floor( (HEIGHT / ((nrings*2.0) + 1.0))*(2.0/sqrt(3.0))*(1.0/2.0) ))
pupscale = (physical_hex_size * 100.0) / hex_size     #[cm/pixel]
crop_size = (2.0/4.0) * hex_size    #(2.0/4.0)*hex_size   #(3.0/4.0)*hex_size   #(3.0/2.0)*hex_size   #hex_height   #glimpse square crop
CHANNELS = 3
LAMBDA_RANGES = 3

L = float ( (physical_hex_size / 2.0) * 2 * ((3 * nrings) + 2) ) 
lambda0 = 700.e-9
r0 = 0.2
xi_factor = 2.5
wlambda = [lambda0, lambda0 * 0.930, lambda0 * 0.860, lambda0 * 0.790]
z0  = 0.8*(r0**2.0)*(xi_factor**(-2.0))*(wlambda[3]**(7.0/5.0))/(500.e-9**(12.0/5.0))  
pstn = np.empty([(2*nrings)+1, (2*nrings)+1])
pstn[:] = np.nan
dq = 0
dr = 0
pstn[axial_to_matrix(dq,   dr,   nrings)] = 0.0
pstn[axial_to_matrix(dq,   dr-1, nrings)] = 0.23
pstn[axial_to_matrix(dq+1, dr-1, nrings)] = -0.5
wf = mirror_wavefront(L, HEIGHT, nrings, pstn, np.zeros_like(pstn), np.zeros_like(pstn))

u = propTF(np.nan_to_num(np.exp(1j* (wlambda[0] / wlambda[0]) * wf)), L, wlambda[0], z0)
i = np.absolute(u**2.0)

Ax, Ay = hex_to_pixel(dq, dr, hex_size)
Bx, By = hex_to_pixel(dq,   dr-1, hex_size)
Cx, Cy = hex_to_pixel(dq+1, dr-1, hex_size)

center_width = (Ax + Bx + Cx)/3.0   #the origin is the center of the image
center_height  = (Ay + By + Cy)/3.0  #the origin is the center of the image
corner_height = center_height - (crop_size/2.0) + (HEIGHT/2.0)
corner_width = center_width - (crop_size/2.0) + (HEIGHT/2.0)


env = environment(CHANNELS)
#img1, _ = env.reset([0.75, 0.45])
#img2 = env.reset_wf([1.0, 1.8])
I2, reward, wfc = env.reset([-0.6, 1.1])
#I2, reward = env.get_circ([-1.0,0.9])

i = I2[int(corner_height):int( corner_height+ crop_size),int(corner_width):int(corner_width+crop_size)]
        

print(reward)
#print(img1.shape)

fig = plt.figure()
plt.imshow(image_p_feed)
plt.show()

