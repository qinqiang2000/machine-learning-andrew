import numpy as np

#COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
#and gives us a numerical estimate of the gradient.
#   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
#   gradient of the function J around theta. Calling y = J(theta) should
#   return the function value at theta.

# Notes: The following code implements numerical gradient checking, and 
#        returns the numerical gradient.It sets numgrad(i) to (a numerical 
#        approximation of) the partial derivative of J with respect to the 
#        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
#        be the (approximately) the partial derivative of J with respect 
#        to theta(i).)
#     
#定义一下数值梯度，用于梯度检查
def computeNumericalGradient(J, theta):
    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)

    e = 1e-4
    it = np.nditer(theta, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        p = it.multi_index
        # Set perturbation vector
        perturb[p] = e;
        loss1 = J(theta - perturb)[0]
        loss2 = J(theta + perturb)[0]
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e);
        perturb[p] = 0;
        it.iternext() 
        
    return numgrad   

"""     fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    e = 1e-4;
    
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + e # increment by e
        fxph = f(x) # evalute f(x + e)
        x[ix] = oldval - e
        fxmh = f(x) # evaluate f(x - e)
        x[ix] = oldval # restore
        
        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * e) # the slope
        it.iternext() # step to next dimension 

    return numgrad"""