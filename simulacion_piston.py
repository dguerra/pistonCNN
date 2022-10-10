from math import *
import numpy as np
from six.moves import xrange
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


def round_away_from_zero(x):
  if np.isscalar(x) == True:
    if(np.absolute(x - np.ceil(x)) == np.absolute(x - np.floor(x))):
      xx = np.sign(x) * np.ceil(np.absolute(x))
    else:
      xx = np.round(x)  
  else:
    xx = np.round(x)
    half = np.where(np.absolute(x - np.ceil(x)) == np.absolute(x - np.floor(x)))
    x_ceil = np.sign(x) * np.ceil(np.absolute(x))
    xx[half] = x_ceil[half]
  return xx

def paridad(x):
  if ((x % 2) == 0):
   par = 2 
  else: 
    par = 1
  return par

def indice(jfin):
  n_gra = np.zeros((jfin + 40,), dtype=np.int)
  m_azi = np.zeros((jfin + 40,), dtype=np.int)
  #;RADIAL DEGREES AND AZIMUTHAL FRENCUENCY ASSOCIATED TO POLYNOMIAL J INDEX.

  n = 1
  j = 0.0
  while j <= jfin:
    for mm in xrange(n+1):
      j = ((n * (n + 1.0)) / 2.0) + mm + 1.0
      if(paridad(n) != paridad(mm)):
        m = mm + 1
      else:
        m = mm
    
      n_gra[int(j)] = n
      m_azi[int(j)] = m
    n += 1

  return n_gra, m_azi

def zer_rad(ro, n, m):
  RR = 0.0
  ddif = round_away_from_zero((n - m) / 2.0)
  dsum = round_away_from_zero((n + m) / 2.0)
  for s in xrange(int(ddif+1)):
    numer = ((-1)**s) * factorial(n - s)
    denom = factorial(s) * factorial(dsum - s) * factorial(ddif - s)
    RR += ((ro**(n - (2.0 * s))) * numer) / denom

  return RR

def zernike_no_mask(j, M):
  x = np.linspace(-1.0, 1.0, num=M, endpoint=False)
  X, Y =  np.meshgrid(x, x, indexing='xy')
  ro = np.sqrt((X * X) + (Y * Y))
  teta = np.arctan2(Y, X)

  n_gra, m_azi = indice(j)
  n = n_gra[j]
  m = m_azi[j]

  if (m == 0):
    Z = np.sqrt(n + 1.0) * zer_rad(ro, n, m)

  else:
    if(paridad(j) == 2):
      Z = np.sqrt((2.0 * n) + 2.0) * zer_rad(ro, n, m) * np.cos(m * teta)
    else:
      Z = np.sqrt((2.0 * n) + 2.0) * zer_rad(ro, n, m) * np.sin(m * teta)

  return Z

#params: dim, diametro_pupila, lambda_, distancia
def generate_fasor_Fresnel( dim, L, lambda_, z):
  dim_ext     = 2 * dim

  if (L / float(dim) <= abs((lambda_ * z) / float(L))):
    print('Muestreo insuficiente')

  #freq = np.linspace(-(dim_ext - 1.0) / 2.0, (dim_ext - 1.0) / 2.0, num=dim_ext) * (1.0/(2.0*L))
  freq = np.fft.fftshift(np.fft.fftfreq(dim_ext, L/dim))
  fx, fy = np.meshgrid(freq, freq, indexing='xy')
  #transfer function: The exp(jkz) term is ignored. This term doesn't affect the transverse spatial structure of the observation plane result.
  fasor = np.exp(-1j*pi*lambda_*z*(fx**2.0 + fy**2.0))
  return fasor

def propagacionFresnel(plano, fasor):
  width = plano.shape[0]
  height = plano.shape[1]
  px_min = width / 2
  py_min = height / 2

  u0 = np.zeros((2*width, 2*height), dtype=np.complex)
  u0[int(px_min):int(px_min+width), int(py_min):int(py_min+height)] = plano

  u = np.fft.ifft2(fasor * np.fft.fftshift(np.fft.fft2(u0)))
  defoc_image = abs(u[int(px_min):int(px_min+width), int(py_min):int(py_min+height)])**2

  return defoc_image


def photon_noise(image, photons):

  if photons > 5e7: photons = 5e7   #set a maximum value for photons to avoid memory problems

  #random.seed(time.time())   #not needed seemingly
  xsize = image.shape[1]
  ysize = image.shape[0]
  noise_xproj = np.zeros((xsize), dtype=np.uint64)
  noise_image = np.zeros(image.shape, dtype=np.double)

  #Find Projection in x
  xproj = np.sum(image, axis=0)
  #introduce noise:
  Itot = np.sum(image) #sum intensity of whole image
  nsx = np.zeros(xsize, dtype=np.float)   #probability function for x-projection
  nsx[0] = xproj[0]/Itot
  for s in xrange(1, xsize): nsx[s]= nsx[s-1] + xproj[s]/Itot

  #distribute all photons randomly on the x-projection:
  fo = np.random.uniform(low=0.0, high=1.0, size=photons)
  #fo = np.ones((photons)) * 0.3
  for i in xrange(photons):
    whr = np.flatnonzero( nsx <= fo[i] )  #gives positions of nonzero values as 'where'
    if whr.size == 0: h = -1
    else: h = np.max(whr)
    if h == xsize - 1: h = xsize - 2
    noise_xproj[h+1] = noise_xproj[h+1] + 1

  #distribute the photons in the y-dimension, slice by slice:
  for j in xrange(xsize):
    slicephotons = noise_xproj[j] # the number of photons in each y-slice is given by its x-projection
    if slicephotons > 0:
      yslice = image[:,j]
      # distribute the photons randomly in the y-slice:  	
      Iy = np.sum(yslice) #sum intensity in y-slice
      nsy = np.zeros(ysize, dtype=np.float)
      nsy[0] = yslice[0]/Iy
      for s in xrange(1,ysize): nsy[s]= nsy[s-1] + yslice[s]/Iy
      
      fo = np.random.uniform(low=0.0, high=1.0, size=slicephotons)
      #fo = np.ones((slicephotons)) * 0.4
      for i in xrange(slicephotons):
        whr = np.flatnonzero( nsy <= fo[i] )  #gives positions of nonzero values as 'where'
        if whr.size == 0: h = -1
        else: h = np.max(whr)
        if h == ysize - 1: h = ysize - 2
        noise_image[h+1,j] = noise_image[h+1,j] + 1

  return noise_image



# w is rms ripple in nm (wavefront)
# l is scalesize (i.e. outer scale) in cm 
# pupscale is size of 1 pixel [cm]
#     default assumes pupscale is 4.2985 cm/pixel
# s is size of output array in pixels
#w -> Hsferr[0]: 	RMS ripple error [nm WF]
#l -> Hsferr[1]: 	Outer scale for ripple error [cm]
def ripple(w,l,s,pupscale=4.2985, lambdanm=500.0):
  #default pupscale for a=22 pixels=0.936 m	
  #want l in pixels, input in cm, use pupscale (cm/pixel)
  #factor of two ?
  #w  in nanometers
	
  l1=(l/(pupscale/2.0))	#pixels within outerscale

  arr = np.zeros((s,s), dtype=np.float)

  if (w > 0.0):
    #d=dist(s)*(1.0/s)
    f = np.fft.fftfreq(s)  #scale so max (1-d) f is 0.5 per pixel
    fx, fy =  np.meshgrid(f, f, indexing='xy')
    d = np.sqrt((fx * fx) + (fy * fy))
  
    wt=((1.0/l1)**2.0+(d**2.0))**(-11.0/6.0) #Van-K type spectrum
    r1 = np.random.normal(loc=0.0, scale=1.0, size=(s,s))
    #r1 = np.ones((s,s), dtype=np.float) * (0.6)
    r2 = np.random.normal(loc=0.0, scale=1.0, size=(s,s))
    #r2 = np.ones((s,s), dtype=np.float) * (-0.8)

    r = r1 + 1j * r2

    screen = (np.fft.ifft2(np.sqrt(wt) * r)).real
    m = np.var(screen)
    ss = np.sqrt(m)
	
    arr = screen * w/ss
    arr = arr - np.mean(arr)  #get mean-error=0
   
    #w input in nanometers, output screen in radians
    arr = (2.0 * pi / lambdanm) * arr # in radians NB MIRROR 
    
  return arr


def introduce_atmosphere(image, r0, dim, L, lambda_, z):
  r0 = r0*((lambda_/500.0e-9)**(6.0/5.0))

  #l_ft = fix(1./l*fftpoints/propdist*(Sm)^2./lambdam,type=15)
  #dx_ft = double(Sm)/double(l_ft)  ; pixel scale [m/pixel]
  #rho_otf = dist(fftpoints)*dx_ft
  dim_ext = 2 * dim
  freq = np.linspace(-(dim_ext - 1.0) / 2.0, (dim_ext - 1.0) / 2.0, num=dim_ext) * (1.0/(2.0*L)) * lambda_*z
  fx, fy = np.meshgrid(freq, freq, indexing='xy')
  rho_otf = np.sqrt(fx**2.0 + fy**2.0)

  otf_atmos = np.exp(-3.44*(rho_otf/r0)**(5.0/3.0))

  # normalise OTF: with abberations, w/o atmosphere
  #mx = max(otf)
  #otf = temporary(otf)  #/mx   #does it need to be normilzed or not?

  #otfti = otf_atmos * np.fft.fftshift(np.fft.fft2(image))      # combine Pupil OTF with atmospheric OTF



  width = image.shape[0]
  height = image.shape[1]
  px_min = width / 2
  py_min = height / 2

  image0 = np.zeros((2*width, 2*height), dtype=np.complex)
  image0[int(px_min):int(px_min+width), int(py_min):int(py_min+height)] = image

  u = np.fft.ifft2(otf_atmos * np.fft.fftshift(np.fft.fft2(image0)))
  return abs(u[int(px_min):int(px_min+width), int(py_min):int(py_min+height)])  #image with atmospheric abberations


def pixel_to_hex(x, y, hex_size):
  '''         __
           __/  \__
        __/  \__/  \__
     __/  \__/  \__/  \__
    /  \__/  \__/  \__/  \
    \__/  \__/  \__/  \__/
    /  \__/  \__/  \__/  \
    \__/  \__/  \__/  \__/
    /  \__/  \__/  \__/  \
    \__/  \__/  \__/  \  /
    /  \__/  \__/  \__/  \
    \__/  \__/  \__/  \__/
       \__/  \__/  \__/
          \__/  \__/  
             \__/
   
  '''
  q = x * 2.0/3.0 / hex_size
  r = (-x / 3.0 + np.sqrt(3.0)/3.0 * y) / hex_size
  return hex_round(q, r)

def axial_to_cube(q, r):
  x = q
  z = r
  y = -x-z
  return x, y, z

def cube_to_axial(x,y,z):
  q = x
  r = z
  return q, r

def hex_round(q, r):
  x, y, z = axial_to_cube(q, r)
  rx, ry, rz = cube_round( x, y, z )
  return cube_to_axial(rx, ry, rz)

def cube_round(x,y,z):
  rx = np.round(x)
  ry = np.round(y)
  rz = np.round(z)

  x_diff = np.abs(rx - x)
  y_diff = np.abs(ry - y)
  z_diff = np.abs(rz - z)

  #cond_a = x_diff > y_diff
  #cond_b = x_diff > z_diff
  #cond_c = y_diff > z_diff
  cond_a = np.all([x_diff > y_diff, x_diff > z_diff], axis=0)
  cond_b = np.all([cond_a == False, y_diff > z_diff], axis=0)
  rx[cond_a] = (-ry-rz)[cond_a]
  ry[cond_b] = (-rx-rz)[cond_b]
  rz[cond_b == False] = (-rx-ry)[cond_b == False]

  '''
  if x_diff > y_diff and x_diff > z_diff:
    rx = -ry-rz
  elif y_diff > z_diff:
    ry = -rx-rz
  else:
    rz = -rx-ry
  '''
  return rx, ry, rz


def hex_to_pixel(q, r, hex_size):
  x = hex_size * 3/2 * q
  y = hex_size * np.sqrt(3) * (r + (q/2.0))
  return x, y

def axial_to_matrix(q,r, nrings):
  i = r + nrings
  j = q + nrings
  return i, j



def mirror_wavefront(L, M, nrings, piston, tip, tilt):  
  '''
    ______
   /      \
  /    ____\  :hex_size: distance fron each corner to the center
  \        /
   \______/    
  '''

  #hex_size = (L/M) * min( floor( (M/2.0)/(1.0 + (nrings * 3.0/2.0)) ), floor( (M / ((nrings*2.0) + 1.0))*(2.0/sqrt(3.0))*(1.0/2.0) ))
  hex_size = (float(L)/float(M)) * float(  min( floor( (M/2.0)/(1.0 + (nrings * 3.0/2.0)) ), floor( (M / ((nrings*2.0) + 1.0))*(2.0/sqrt(3.0))*(1.0/2.0) ))  )

  x = np.linspace(-L/2.0, L/2.0, num=M, endpoint=True)  #endpoint = True to visualize wavefront properly
  X, Y =  np.meshgrid(x, x, indexing='xy')

  q,r = pixel_to_hex(X, Y, hex_size)
  wavefront = np.empty([M,M])


  wavefront[:] = np.nan
   
  max_tilt = 1.0e-8

  #go through hexagons within N rings
  for dx in range(-nrings, nrings+1):
    for dy in range( max(-nrings, -dx-nrings), min(nrings, -dx+nrings)+1):
      dz = -dx-dy
      dq, dr = cube_to_axial(dx,dy,dz)
      #print("qr:", dq, dr, " -> ij:", dr + nrings, dq + nrings)
      di, dj = axial_to_matrix(dq, dr, nrings)
      hxg = np.all([q==dq,r==dr],axis=0)    #selection of pixels that belong to the hexagon
      #compute random tilt:
      alpha = np.random.uniform(0.0, max_tilt)    #5.0e-5   #rad
      theta = np.random.uniform(0.0, 2.0*pi)
      tilt = 2.0*pi*(X*np.cos(theta)+Y*np.sin(theta))*np.tan(alpha)/700e-9

      wavefront[ hxg ] = piston[di, dj] +  ( tilt[hxg] - np.mean(tilt[hxg]) )

  return wavefront


