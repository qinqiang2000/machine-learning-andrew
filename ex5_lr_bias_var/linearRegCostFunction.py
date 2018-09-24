
""" LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
regression with multiple variables
   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
   cost of using theta as the parameter for linear regression to fit the 
   data points in X and y. Returns the cost in J and the gradisent in grad """

import numpy as np

def linearRegCostFunction(theta, X, y, lamda):
    m, n = X.shape

    theta = theta.reshape((n, 1))
    theta_reg = np.r_[np.zeros((1,1)), theta[1:,:]]
    # a = theta[1:,:] 
    # theta_reg = np.insert(a, 0, 0, 0)

    hx = X @ theta
    
    J = (np.sum(np.square(hx - y)) + lamda * (theta_reg.T @ theta_reg)) * 0.5 / m
    
    Grad = (X.T @ (hx - y) + lamda * theta_reg) / m

    return J, Grad.flatten()