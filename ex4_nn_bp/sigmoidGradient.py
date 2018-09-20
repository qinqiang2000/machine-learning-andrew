import numpy as np
from scipy.special import expit #Vectorized sigmoid function

def sigmoidGradient(z):
  g = expit(z)
  return g * (1 - g)