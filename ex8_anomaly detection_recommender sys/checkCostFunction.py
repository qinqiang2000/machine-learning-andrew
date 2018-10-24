import numpy as np
from computeNumericalGradient import *
from cofiCostFunc import cofiCostFunc

def checkCostFunction(params, Y, myR, nu, nm, nf, l = 0.):
    costFunc = lambda w: cofiCostFunc(w, Y, myR, nu, nm, nf, l)

    J, grad = costFunc(params)  
    # 计算数值梯度
    numgrad = computeNumericalGradient(costFunc, params)
        
    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your cost function implementation is correct,\n'
    'the relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\n'
    'lambda={}, Relative Difference: {}'.format(l, diff))