import numpy as np
import matplotlib.pyplot as plt
from polyFeatures import *

""" PLOTFIT Plots a learned polynomial regression fit over an existing figure.
Also works with linear regression.
   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
   fit with power p and feature normalization (mu, sigma).
 """
def plotFit(min_x, max_x, mu, sigma, theta, p):
  # We plot a range slightly bigger than the min and max values to get
  # an idea of how the fit will vary outside the range of the data points
  x = np.arange(min_x - 20, max_x + 15, 0.05).reshape(-1, 1)

  # Map the X values 
  X_poly = polyFeatures(x, p)
  X_poly = (X_poly - mu) / sigma
  X_poly = np.insert(X_poly, 0, 1 , axis=1)     # Add Ones

  # Plot
  plt.plot(x, X_poly @ theta.T, 'b--')
