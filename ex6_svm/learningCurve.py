import numpy as np
from trainLinearReg import *
from linearRegCostFunction import *

""" LEARNINGCURVE Generates the train and cross validation set errors needed 
to plot a learning curve
   [error_train, error_val] = ...
       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
       cross validation set errors for a learning curve. In particular, 
       it returns two vectors of the same length - error_train and 
       error_val. Then, error_train(i) contains the training error for
       i examples (and similarly for error_val(i)).

   In this function, you will compute the train and test errors for
   dataset sizes from 1 up to m. In practice, when working with larger
   datasets, you might want to do this in larger intervals.
 """

def learningCurve(X, y, Xval, yval, lamda):
  # Number of training examples
  m = X.shape[0]

  # You need to return these values correctly
  error_train = np.zeros((m, 1))
  error_val   = np.zeros((m, 1))

  for i in range(1, m + 1):
    xSub = X[:i, :]
    ySub = y[:i, :]

    theta = trainLinearReg(xSub, ySub, lamda)
    error_train[i-1] = linearRegCostFunction(theta, xSub, ySub, 0)[0]
    error_val[i-1] = linearRegCostFunction(theta, Xval, yval, 0)[0]

  return error_train, error_val