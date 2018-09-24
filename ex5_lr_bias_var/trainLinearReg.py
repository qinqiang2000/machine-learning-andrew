import numpy as np
import scipy.optimize as op
from linearRegCostFunction import linearRegCostFunction 

""" TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
regularization parameter lambda
   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
   the dataset (X, y) and regularization parameter lambda. Returns the
   trained parameters theta. """
def trainLinearReg(X, y, lamda):
    # Initialize Theta
    initial_theta = np.zeros((X.shape[1], 1)) 

    # Create "short hand" for the cost function to be minimized
    costFunc = lambda w: linearRegCostFunction(X, y, w, lamda)

    # Now, costFunction is a function that takes in only one argument
    minimize_method = "CG" # CG, L-BFGS-B, TNC, ...
    opts = {'maxiter':200}
    if minimize_method == "TNC":
        opts['maxCGit'] = 0
        opts['stepmx'] = 500
    elif minimize_method == 'L-BFGS-B':
        opts['eps'] = 1e-8
    else:
        pass

    # Minimize using fmincg
    Result = op.minimize(fun = linearRegCostFunction,
                        x0 = initial_theta,
                        method = minimize_method,
                        args = (X, y, lamda),
                        jac = True,
                        options= opts)
                        
    return Result.x