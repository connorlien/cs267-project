from scipy.optimize import linprog
import numpy as np
from math import log, floor, ceil

def words(kb):
  return kb * 1024/4

def words_tensor(size, data_size = 8):
  return size * data_size/4

def calc_2d_blocks(b, wi, hi, wo, ho, hf, wf, ci, ndw, base = np.e):

  B = b
  W_out = wo
  H_out = ho
  H_f = hf
  W_f = wf
  C_in = ci
  N_dw = ndw


  # cache size
  M = words(32) 

  num_loops = 7

  p_o = words_tensor(b * wo * ho * N_dw * C_in)
  p_f = words_tensor(wf * hf * N_dw * C_in)
  p_i = words_tensor(b * wi * hi * C_in)

  p_t = p_o + p_i + p_f
  t1 = 1 - log(p_t, M)
  t2 = 1 - log(4 * p_t, M)

  c = np.array([-1 for _ in range(9)])
  A_ub = np.array([[1, 0, 1, 1, 1, 0, 0, 0, 0],
                   [0, 1, 1, 0, 0, 1, 1, 1, 1],
                   [1, 1, 0, 1, 1, 0, 1, 0, 1],
                   [1, 1, 0, 1, 0, 0, 1, 1, 1],
                   [1, 1, 0, 0, 1, 1, 1, 0, 1],
                   [1, 1, 0, 0, 0, 1, 1, 1, 1]])
  
  
  b_ub = np.array([t1, t1, t2, t2, t2, t2])
  o = linprog(c, A_ub=A_ub, b_ub=b_ub)
  print(o.x)
  return list(map(floor, base ** (o.x)))