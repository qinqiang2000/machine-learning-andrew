#Some utility functions. There are lot of flattening and
#reshaping of theta matrices, the input X matrix, etc...
#Nicely shaped matrices make the linear algebra easier when developing,
#but the minimization routine (fmin_cg) requires that all inputs

import numpy as np

def unrollParams(theta_set):
  v = []
  for t in theta_set:
      v.extend(t.flatten().tolist())
  return np.array(v)
