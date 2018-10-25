import numpy as np
from unrollutility import unrollParams

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, l=0):
    """ 
    COFICOSTFUNC Collaborative filtering cost function
    [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
    num_features, lambda) returns the cost and gradient for the
    collaborative filtering problem.
    """  
    # unfold X and Theta matrices    
    X = params[:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)

    error = .5 * np.square((X @ Theta.T - Y) * R).sum()
    reg1 = .5 * l * np.square(Theta).sum()
    reg2 = .5 * l * np.square(X).sum()
    J = error + reg1 + reg2

    X_grad = (((X @ Theta.T - Y) * R) @ Theta) + l * X
    Theta_grad = (((X @ Theta.T - Y) * R).T @ X) + l * Theta

    grad = unrollParams([X_grad, Theta_grad])

    return J, grad.flatten()
