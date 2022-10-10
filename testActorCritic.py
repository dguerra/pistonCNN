import tensorflow as tf
import numpy as np
from math import *
#from foo import environment
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import triang
from simulacion_piston import mirror_wavefront
from simulacion_piston import introduce_atmosphere
from FourierOptics import propTF
from FourierOptics import focus
from FourierOptics import propFF

import matplotlib.pyplot as plt

from simulacion_piston import axial_to_matrix
from simulacion_piston import axial_to_matrix
from simulacion_piston import hex_to_pixel


class environment():
    def __init__(self, nrings, HEIGHT, CHANNELS):
        self.nrings = nrings
        self.physical_hex_size = 0.95
        self.LAMBDA_RANGES = 0.5
        self.CHANNELS = CHANNELS
        self.pstn = np.zeros([(2*self.nrings)+1, (2*self.nrings)+1])     # np.zeros([(2*nrings)+1, (2*nrings)+1])
        dq = 0
        dr = 0

        self.L = float ( (self.physical_hex_size / 2.0) * 2 * ((3 * self.nrings) + 2) )    # physical side length of pupil plane in [m]
        self.HEIGHT = HEIGHT

        self.hex_size = min( floor( (self.HEIGHT/2.0)/(1.0 + (self.nrings * 3.0/2.0)) ), floor( (self.HEIGHT / ((self.nrings*2.0) + 1.0))*(2.0/sqrt(3.0))*(1.0/2.0) ))

        lambda0 = 700.e-9
        self.wlambda = [lambda0, lambda0 * 0.930, lambda0 * 0.860, lambda0 * 0.790]

        xi_factor = 2.5     #proportion of difraction signal width over image blur due to the turbulence
        self.r0 = 0.2            #worst atmospheric r0 considered
        self.z0  = 0.8*(self.r0**2.0)*(xi_factor**(-2.0))*(self.wlambda[3]**(7.0/5.0))/(500.e-9**(12.0/5.0))    #propagation distance (previously 15875.0)

        Ax, Ay = hex_to_pixel(dq, dr, self.hex_size)
        Bx, By = hex_to_pixel(dq,   dr-1, self.hex_size)
        Cx, Cy = hex_to_pixel(dq+1, dr-1, self.hex_size)
        self.pupscale = (self.physical_hex_size * 100.0) / self.hex_size     #[cm/pixel]
        
        self.crop_size = (2.0/4.0) * self.hex_size    #(2.0/4.0)*hex_size   #(3.0/4.0)*hex_size   #(3.0/2.0)*hex_size   #hex_height   #glimpse square crop
        print("crop size.", self.crop_size)

        x = np.linspace(-(self.pupscale * self.crop_size * 0.5), (self.pupscale * self.crop_size * 0.5), num=int(self.crop_size), endpoint=False)
        X, Y =  np.meshgrid(x, x, indexing='xy')
        self.foc_mask = np.sqrt((X * X) + (Y * Y))
        mask_radious = 10.0    #in [cm] if r0 = 0.2 it means a diameter of 20 cm and radiuous 10cm
        self.foc_mask[np.where(self.foc_mask<mask_radious)] = 1.0
        self.foc_mask[np.where(self.foc_mask>mask_radious)] = np.nan

        #Compute location and size of the crop window
        center_width = (Ax + Bx + Cx)/3.0   #the origin is the center of the image
        center_height  = (Ay + By + Cy)/3.0  #the origin is the center of the image
        self.corner_height = center_height - (self.crop_size/2.0) + (self.HEIGHT/2.0)
        self.corner_width = center_width - (self.crop_size/2.0) + (self.HEIGHT/2.0)
        print(self.pstn)


    def reset(self, vv, imageI=True, PSF=True):
        dq = 0
        dr = 0
        self.pstn[axial_to_matrix(dq,   dr-1, self.nrings)] = vv[0]  #upper hexagon
        self.pstn[axial_to_matrix(dq+1, dr-1, self.nrings)] = vv[1]  #upper RIGHT hexagon

        wf = mirror_wavefront(self.L, self.HEIGHT, self.nrings, self.pstn, np.zeros_like(self.pstn), np.zeros_like(self.pstn))
        wf_foc = wf[int(self.corner_height):int(self.corner_height+self.crop_size),int(self.corner_width):int(self.corner_width+self.crop_size)]
        
        r0_500nm = np.random.uniform(self.r0, 0.2)
        zf = 10000

        ax = np.random.randint(2)
        act_mask = np.roll(self.foc_mask, np.random.randint(3)-1, axis=ax)

        oi = []
        strehl = 0.0

        if imageI==True:
          for indx in range(self.CHANNELS):
              u = propTF(np.nan_to_num(np.exp(1j* (self.wlambda[0] / self.wlambda[indx]) * wf)), self.L, self.wlambda[indx], self.z0)
              i = np.absolute(u**2.0)
              i = introduce_atmosphere(i, r0_500nm, self.HEIGHT, self.L, self.wlambda[indx], self.z0)
              i = i[int(self.corner_height):int(self.corner_height+self.crop_size),int(self.corner_width):int(self.corner_width+self.crop_size)]
              oi.append(i)

          ii = np.stack(oi, axis=-1)
          ii = np.array(ii, ndmin=3, dtype=np.float32)
        else:
          ii = _

        if PSF==True:
          for indx in range(self.CHANNELS):
              uin = np.nan_to_num(act_mask * np.exp(1j* (self.wlambda[0] / self.wlambda[indx]) * wf_foc))
              u2 = propFF(uin, (self.pupscale * self.crop_size) / 100, self.wlambda[indx], zf)    # find psf by fraunhofer pattern
              I_foc = np.absolute(u2**2.0)
              strehl += np.amax(I_foc)
       
        else:
          strehl = 0.0

        return ii, strehl


#LOGDIR = "/output/phasedivcnn/1"

nrings = 1
cap = pi/2.0
HEIGHT = 512 #512 #256  #1024 #number of samples in each direction in the pupil plane [pixels]
CHANNELS = 3
LAMBDA_RANGES = 1

tf.reset_default_graph()

input_dims = 24   #49 #24   #3   #2    #24  #49  #12 
action_dims = 2
batch_size = 6 #6*(nrings**2)    #8 # 16  #1024  #8192  #4096

#state_placeholder = tf.placeholder(tf.float32, [batch_size, input_dims]) 
state_placeholder = tf.placeholder(tf.float32, [batch_size, input_dims, input_dims, CHANNELS])
#state_placeholder = tf.placeholder(tf.float32, [input_dims, input_dims, CHANNELS])

#mirror = Mirror(nrings, HEIGHT, CHANNELS, LAMBDA_RANGES)
env = environment(nrings, HEIGHT, CHANNELS)
i_s = np.array([1.0,2.0])
state_i, _ = env.reset(i_s, imageI=True, PSF=True)

fig = plt.figure(facecolor='w', edgecolor='k')
plt.imshow(np.squeeze(state_i))
plt.show()


