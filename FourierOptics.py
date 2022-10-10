from math import *
import numpy as np
from six.moves import xrange
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


def rect(x):
  # rectangle function
  return np.absolute(x)<=1.0/2.0

def focus(uin,L,wlambda,zf):
  # converging or diverging phase-front
  # uniform sampling assumed
  # uin - input field
  # L - side length
  # wlambda - wavelength
  # zf - focal distance (+ converge, - diverge)
  # uout - output field

  [M,N] = uin.shape # get input field array size
  dx = L/M  # sample interval
  k = (2.0 * pi) / wlambda # wavenumber

  x = np.linspace(-L/2.0, L/2.0, num=M, endpoint=False)
  [X,Y] = np.meshgrid(x, x, indexing='xy')

  return uin * np.exp(-1j*k/(2*zf)*(X**2.0+Y**2.0))   # apply focus


def tilt(uin,L,wlambda,alpha,theta):
# tilt phasefront
# uniform sampling assumed
# uin - input field
# L - side length
# wlambda - wavelength
# alpha - tilt angle
# theta - rotation angle (x axis 0)
# uout - output field

  [M,N] = uin.shape #get input field array size
  dx = L/M #sample interval
  k = 2.0 * pi / wlambda #wavenumber

  x = np.linspace(-L/2.0, L/2.0, num=M, endpoint=False) #coords
  [X,Y] = np.meshgrid(x, x, indexing='xy')

  return uin * np.exp(1j*k*(X*np.cos(theta)+Y*np.sin(theta))*np.tan(alpha)) #apply tilt


def propFF(u1,L1,wlambda,z):
  # propagation - Fraunhofer pattern
  # assumes uniform sampling
  # u1 - source plane field
  # L1 - source plane side length
  # wlambda - wavelength
  # z - propagation distance
  # L2 - observation plane side length
  # u2 - observation plane field

  [M,N] = u1.shape #get input field array size
  dx1 = L1/M #source sample interval
  k = (2.0 * pi) / wlambda  #wavenumber

  L2 = wlambda * z / dx1  #obs sidelength
  dx2 = wlambda * z / L1  #obs sample interval

  x2 = np.linspace(-L2/2.0, L2/2.0, num=M, endpoint=False)
  [X2,Y2] = np.meshgrid(x2,x2, indexing='xy')

  c = 1.0/(1j*wlambda*z) * np.exp(1j*k/(2.0*z)*(X2**2.0+Y2**2.0))
  u2 = c * np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(u1))) * dx1**2.0
  return u2


def propTF(u1,L,wlambda,z):
  # propagation - transfer function approach, assumes same x and y side lengths and uniform sampling
  # u1 - source plane field
  # L - source and observation plane side length
  # wlambda - wavelength
  # z - propagation distance
  # u2 - observation plane field 
  #TODO: Check sampling rate is enough
  [M,N] = u1.shape #get input field array size
  assert N == M
  dx = L/M                #sample interval
  k = (2.0*pi)/wlambda    #wavenumber

  u1_dd = np.zeros((2*M,2*M), dtype=np.complex)
  u1_dd[int(M/2):int((M/2)+M), int(M/2):int((M/2)+M)] = u1

  fx = np.fft.fftfreq(2*M, dx)   #freq coords equivalent to np.linspace(-1.0/(2.0*dx), 1.0/(2.0*dx), num=M, endpoint=False) and shifted

  [FX,FY] = np.meshgrid(fx,fx, indexing='xy')

  H = np.exp(-1j*pi*wlambda*z*(FX**2.0 + FY**2.0))     #transfer function. It is already shifted by fftfreq

  U1 = np.fft.fft2(np.fft.fftshift(u1_dd))        #shift, fft src field
  U2 = H * U1                                  #multiply
  u2 = np.fft.ifftshift(np.fft.ifft2(U2))      #inv fft, center obs field

  u2 = u2[int(M/2):int((M/2)+M),int(M/2):int((M/2)+M)]
  return u2 


def sqr_beam():
  L1 = 0.5    # side length
  M = 250     # number of samples
  dx1 = L1/M
  w = 0.055 # rectangle half-width (m)
  x1 = np.linspace(-L1/2.0, L1/2.0, num=M, endpoint=False)

  wlambda = 0.5e-6       #wavelength
  k = 2.0 * pi/wlambda   #wavenumber


  [X1,Y1] = np.meshgrid(x1, x1, indexing='xy')
  u1 = np.outer( rect(x1/(2.0*w)), rect(x1/(2.0*w)) )   #source field
  I1 = np.absolute(u1**2.0)

  z = 2000               #propagation distance (m) 

  case = "FOC"   #compute Fraunhofer pattern

  if case == "FF":
    u2 = propFF(u1,L1,wlambda,z)
  elif case == "FOC":
    #apply focus phase
    zf = 2000              #focus distance (m)
    u1 = focus(u1, L1, wlambda, zf)
    #propagate a distance z
    u2 = propTF(u1, L1, wlambda, z)

  I2 = np.absolute(u2**2.0)


  fig = plt.figure()
  plt.imshow(I1) 
  plt.show()




#sqr_beam()




