from simulacion_piston import mirror_wavefront
from simulacion_piston import introduce_atmosphere
from simulacion_piston import ripple
from FourierOptics import propTF
from math import *
import numpy as np

class Mirror():
    def __init__(self, nrings, HEIGHT, CHANNELS, LAMBDA_RANGES):
      self.nrings = nrings
      self.HEIGHT = HEIGHT
      self.LAMBDA_RANGES = LAMBDA_RANGES
      self.CHANNELS = CHANNELS
      self.physical_hex_size = 0.95   #distance from any vertix to the center of hexagon [m]
      self.L = float ( (self.physical_hex_size / 2.0) * 2 * ((3 * nrings) + 2) )    # physical side length of pupil plane in [m]

      hex_size = min( floor( (HEIGHT/2.0)/(1.0 + (nrings * 3.0/2.0)) ), floor( (HEIGHT / ((nrings*2.0) + 1.0))*(2.0/sqrt(3.0))*(1.0/2.0) ))    #pixel size of the hexagon

      lambda0 = 700.e-9  
      self.wlambda = [lambda0, lambda0 * 0.930, lambda0 * 0.860, lambda0 * 0.790]
      self.xi_factor = 2.5     #proportion of difraction signal width over image blur due to the turbulence
      self.r0 = 0.10            #worst atmospheric r0 considered
      self.z0  = 0.8*(self.r0**2.0)*(self.xi_factor**(-2.0))*(self.wlambda[3]**(7.0/5.0))/(500.e-9**(12.0/5.0))    #propagation distance (previously 15875.0)

      self.pupscale = (self.physical_hex_size * 100.0) / hex_size     #[cm/pixel] 

      self.tip_tilt_range = 0.0  #0.5 * 1.e-8    #produces maximum distance of +-20nm over the vertex [radiands]

      #polish_error = ripple(20.0, 20.0, HEIGHT, pupscale = pupscale, lambdanm = lambda0*1.e9)
      self.polish_error = ripple(20.0, 20.0, HEIGHT, pupscale = self.pupscale, lambdanm = lambda0*1.e9)


    def create_wavefront(self, pstn):
       return mirror_wavefront(self.L, self.HEIGHT, self.nrings, pstn, np.zeros_like(pstn), np.zeros_like(pstn))

    def create_data(self, pstn): 
      wf = mirror_wavefront(self.L, self.HEIGHT, self.nrings, pstn, np.zeros_like(pstn), np.zeros_like(pstn)) + self.polish_error
      r0_500nm = np.random.uniform(self.r0, 0.2)

      
      oi = []
      for indx in range(self.CHANNELS):
        u = propTF(np.nan_to_num(np.exp(1j* (self.wlambda[0] / self.wlambda[indx]) * wf)), self.L, self.wlambda[indx], self.z0)
        i = np.absolute(u**2.0)
        i_atmos = introduce_atmosphere(i, r0_500nm, self.HEIGHT, self.L, self.wlambda[indx], self.z0)
        #oi.append(i)
        oi.append(i_atmos)
      

      ii = np.stack(oi, axis=-1)
      
      return np.array(ii, ndmin=3, dtype=np.float32)


 
