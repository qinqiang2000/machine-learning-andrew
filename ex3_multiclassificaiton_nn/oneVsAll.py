import numpy as np
from lrCostFunction import lrCostFunction, lrGradient
from scipy import optimize

#ONEVSALL trains multiple logistic regression classifiers and returns all
#the classifiers in a matrix all_theta, where the i-th row of all_theta 
#corresponds to the classifier for label i
#   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
#   logistic regression classifiers and returns each of these classifiers
#   in a matrix all_theta, where the i-th row of all_theta corresponds 
#   to the classifier for label i

def oneVsAll(X, y, num_labels, lamda):
  # Add ones to the X data matrix
  X = np.c_[np.ones(X.shape[0]), X]
  
  # Some useful varibles
  m, n = X.shape
  all_theta = np.zeros((n, num_labels))

  # 每次只训练一个分类器，就相当于，y==i时，是1，其他等于0
  for i in range(num_labels):
    initial_theta = np.zeros(n)
    myargs = (X, y == i, lamda)
    res = optimize.minimize(fun=lrCostFunction, x0=initial_theta, args=myargs, method='CG', jac=lrGradient)
    all_theta[:, i] = res.x
    print(i,": " ,res.success)
  return all_theta