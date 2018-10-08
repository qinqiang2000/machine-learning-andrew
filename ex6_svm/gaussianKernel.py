import numpy as np

""" RBFKERNEL returns a radial basis function kernel between x1 and x2
   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
   and returns the value in sim """

def gaussianKernel(x1, x2, sigma):
    sim = np.sum(np.square(x1 - x2)) * -0.5 / pow(sigma, 2)
    sim = np.exp(sim)

    return sim