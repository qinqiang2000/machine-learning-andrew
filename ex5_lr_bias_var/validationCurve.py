import numpy as np
from trainLinearReg import *
from linearRegCostFunction import *

""" VALIDATIONCURVE Generate the train and validation errors needed to
plot a validation curve that we can use to select lambda
   [lambda_vec, error_train, error_val] = ...
       VALIDATIONCURVE(X, y, Xval, yval) returns the train
       and validation errors (in error_train, error_val)
       for different values of lambda. You are given the training set (X,
       y) and validation set (Xval, yval).
 """
def validationCurve(X, y, Xval, yval):
  # Selected values of lambda (you should not change this)
  lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
  
  # You need to return these variables correctly.
  error_train = np.zeros(len(lambda_vec))
  error_val = np.zeros(len(lambda_vec))

  for i in range(len(lambda_vec)):
    lamda = lambda_vec[i]

    theta = trainLinearReg(X, y, lamda)
    error_train[i] = linearRegCostFunction(theta, X, y, 0)[0]  # 记得把lambda = 0
    error_val[i] = linearRegCostFunction(theta, Xval, yval, 0)[0]  # 记得把lambda = 0

  return lambda_vec, error_train, error_val
