from math import *
import numpy as np
import random
nrings = 3
np.set_printoptions(threshold=np.nan)                           
LAMBDA_RANGES = 11
nrings = 3
nsegs = (3 * nrings * (nrings + 1)) + 1
ninput = 2 * 3 * 27
time_steps = 2 * 27
NUM_CLASSES = 3

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




def svd_solve(a, b):
    [U, s, Vt] = np.linalg.svd(a, full_matrices=False)
    r = max(np.where(s >= 1e-16)[0])
    temp = np.dot(U[:, :r].T, b) / s[:r]
    return np.dot(Vt[:r, :].T, temp)

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

  AA = np.zeros([time_steps * NUM_CLASSES, nsegs])
  consistency_threshold = 0.08
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
      else:
        print("Inconsistency detected")

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
      else:
        print("Inconsistency detected")

      ts += 1
  print("AA shape ", AA.shape, "B shape ", B.shape)
  #print(AA)
  #print(B)
  print("matrix_rank of AA: ", np.linalg.matrix_rank(AA))
  x = svd_solve(AA,B)
  #x = np.linalg.lstsq(AA, np.reshape(B, [-1, 1]), rcond=None)[0]
  print("x shape", x.shape)
  #be suere piston values are measured with respect to the central hexagon
  x = x - x[int(segment_ind[axial_to_matrix(0, 0, nrings)])]

  print("label_total_stack ", label_total_stack)
  print("x ", x)
  print("label_total_stack shape ", label_total_stack.shape, "x shape ", x.shape)
  return np.mean(np.absolute(label_total_stack - x)**2.0)


def create_data():
  pstn = np.random.uniform(-LAMBDA_RANGES * pi, LAMBDA_RANGES * pi, [(2*nrings)+1, (2*nrings)+1])

  x_stack = []
  offset_value = pstn[axial_to_matrix(0, 0, nrings)]

  pstn = pstn - offset_value
  rand_indx = random.sample(range(0,54), 53)

  count = 0
  for dx in range(-nrings, nrings):
    for dy in range( max(-nrings, -dx-nrings), min(nrings, -dx+nrings)):
      dq, dr = cube_to_axial(dx,dy,-dx-dy)
      pA = pstn[axial_to_matrix(dq,   dr,   nrings)]  
      pB = pstn[axial_to_matrix(dq,   dr-1, nrings)]   #upper hexagon
      pC = pstn[axial_to_matrix(dq+1, dr-1, nrings)]   #upper right hexagon
      if count in rand_indx:
        x_i = np.stack([pA-pB, pB-pC, 0.5*(pC-pA)], axis=0)        #set to wrong one entry randomly
      else:
        x_i = np.stack([pA-pB, pB-pC, pC-pA], axis=0)
      x_stack.append(x_i)
      count = count + 1

      dqf, drf = cube_to_axial(-dx, -(-dx-dy), -dy)
      pAf = pstn[axial_to_matrix(dqf,   drf,   nrings)]  
      pBf = pstn[axial_to_matrix(dqf,   drf-1, nrings)]   #upper hexagon
      pCf = pstn[axial_to_matrix(dqf-1, drf,   nrings)]   #upper LEFT hexagon
      if count in rand_indx:
        x_if = np.stack([pAf-pBf, pBf-pCf, 0.5*(pCf-pAf)], axis=0)        #set to wrong value entry randomly
      else:
        x_if = np.stack([pAf-pBf, pBf-pCf, pCf-pAf], axis=0)
      x_stack.append(x_if)

      count = count + 1

        
  x_stack = np.stack(x_stack, axis=0)
  x = np.reshape(x_stack, [-1])
  
  #x[np.random.randint(0, np.size(x)-1)] = np.random.uniform( -2.0 * LAMBDA_RANGES * pi, 2.0 * LAMBDA_RANGES * pi )
  
  #x = np.random.uniform(low=-10.0, high=10.0)
  #y = (x * 5.0) + 7.0    # Create data from scratch using the known relation
  #p = np.array([x], dtype=np.float32)
  return np.array(x, ndmin=1, dtype=np.float32), np.array(pstn, ndmin=2, dtype=np.float32)


B_, pstn_val = create_data()

print( final_loss(B_, pstn_val) )
