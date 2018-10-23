import numpy as np
import matplotlib.pyplot as plt
from multivariateGaussian import multivariateGaussian

def visualizeFit(X, mu, sigma2):
    """ 
    VISUALIZEFIT Visualize the dataset and its estimated distribution.
    VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the 
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
    """
    xx = np.arange(0, 35, 0.5)
    yy = np.arange(0, 35, 0.5)

    X1, X2 = np.meshgrid(xx, yy)

    points = np.c_[X1.ravel(), X2.ravel()]
    Z = multivariateGaussian(points, mu, sigma2)
    Z = Z.reshape(X1.shape)

    # 这个levels是作业里面给的参考,或者通过求解的概率推出来
    cont_levels = [10**h for h in range(-20,0,3)]

    plt.contour(X1, X2, Z, cont_levels)
