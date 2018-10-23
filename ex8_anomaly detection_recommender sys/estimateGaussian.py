import numpy as np


def estimateGaussian(X, useMultivariate=False):
    """ This function estimates the parameters of a 
    Gaussian distribution using the data in X
    [mu sigma2] = estimateGaussian(X), 
    The input X is the dataset with each n-dimensional data point in one row
    The output is an n-dimensional vector mu, the mean of the data set
    and the variances sigma^2, an n x 1 vector
    """ 
    mu = np.mean(X, axis=0)
    
    if useMultivariate:    
        sigma = ((X-mu).T @ (X-mu)) / len(X)
    else:
        sigma = np.var(X, axis=0)  

    return mu, sigma


    