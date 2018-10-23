#  Exercise 8 | Anomaly Detection and Collaborative Filtering

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from estimateGaussian import estimateGaussian
from multivariateGaussian import multivariateGaussian
from visualizeFit import visualizeFit
from selectThreshold import selectThreshold

""" 
 ================== Part 1: Load Example Dataset  ===================
  We start this exercise by using a small dataset that is easy to
  visualize.

  Our example case consists of 2 network server statistics across
  several machines: the latency and throughput of each machine.
  This exercise will help us find possibly faulty (or very fast) machines.
"""
print("Visualizing example dataset for outlier detection.")

#  The following command loads the dataset. You should now have the
#  variables X, Xval, yval in your environment
mat = sio.loadmat('ex8data1.mat')

#  Visualize the example dataset
X, Xval, yval = mat['X'], mat['Xval'], mat['yval']
plt.plot(X[:,0], X[:,1], 'bx')
plt.axis([0, 30, 0, 30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
# plt.show()

""" 
 ================== Part 2: Estimate the dataset statistics ===================
  For this exercise, we assume a Gaussian distribution for the dataset.

  We first estimate the parameters of our assumed Gaussian distribution, 
  then compute the probabilities for each of the points and then visualize 
  both the overall distribution and where each of the points falls in 
  terms of that distribution.
 """
print("\nVisualizing Gaussian fit")

#  Estimate my and sigma2
mu, sigma2 = estimateGaussian(X)
print(mu.shape,sigma2.shape)
print(mu,sigma2)

#  Returns the density of the multivariate normal at each data point (row) 
#  of X
p = multivariateGaussian(X, mu, sigma2)

# Visualize the fit
visualizeFit(X,  mu, sigma2)
# plt.show()

""" 
================== Part 3: Find Outliers ===================
 Now you will find a good epsilon threshold using a cross-validation set
 probabilities given the estimated Gaussian distribution 
"""
pval = multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = selectThreshold(yval, pval)
print('\nBest epsilon found using cross-validation: %e'% epsilon)
print('Best F1 on Cross Validation Set:  %f'% F1)
print('   (you should see a value epsilon of about 8.99e-05)')
print('   (you should see a Best F1 value of  0.875000)')

