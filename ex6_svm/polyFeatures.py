import numpy as np

""" POLYFEATURES Maps X (1D vector) into the p-th power
   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
   maps each example into its polynomial features where
   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
 """
def polyFeatures(X, p):
  X_poly = np.zeros((X.shape[0], p))
  for i in range(p):
    X_poly[:, i] = np.power(X, i + 1).flatten()

  return X_poly