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

    c = np.array([-1 for _ in range(num_loops)])
    A_ub = np.array([[1,1,1,1,1,1],
                    [0,1,0,0,1,1],
                    [1,1,1,1,0,0],
                    [1,0,0,0,0,0],
                    [0,1,0,0,0,0],
                    [0,0,1,0,0,0], 
                    [0,0,0,1,0,0], 
                    [0,0,0,0,1,0],
                    [0,0,0,0,0,1],
                    [0,0,0,0,0,0]])
    b_ub = [1,1,1] + [log(i, M) for i in [B, C_in, W_out, H_out, W_f, H_f]]
    l = [0 for _ in range(num_loops)]
    o = linprog(c, A_ub=A_ub, b_ub=b_ub)
    return list(map(ceil, M**(o.x)))

def calc_point_blocks(b, w, h, co, ci, base = np.e):
  B = b
  W_in = w
  H_in = h
  C_out = co
  C_in = ci

  # cache size
  M = words(32) 
  num_loops = 5
  c = np.array([-1 for _ in range(num_loops)])
  A_ub = np.array([[1,0,1,1,1],
                   [0,1,0,0,1],
                   [1,1,1,1,0],
                   [1,0,0,0,0],
                   [0,1,0,0,0],
                   [0,0,1,0,0], 
                   [0,0,0,1,0], 
                   [0,0,0,0,1]])
  b_ub = [1,1,1] + [log(i, M) for i in [B, C_out, W_in, H_in, C_in]]
  l = [0 for _ in range(num_loops)]
  o = linprog(c, A_ub=A_ub, b_ub=b_ub)
  return list(map(ceil, M**(o.x)))
