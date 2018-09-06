import numpy as np

# Compute cost for linear regression
# J = computeCost(X, y, theta) computes the cost of using theta as the
# parameter for linear regression to fit the data points in X and y
def computeCost(X, y, theta):
    m = y.size  # number of training examples
    return ((X.dot(theta) - y) ** 2).sum() / (2 * m)

# Performs gradient descent to learn theta
# theta = gradientDescent(X, y, theta, alpha, num_iters) updates theta by
# taking num_iters gradient steps with learning rate alpha
def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros((num_iters, 1))
    for iter in range(0, num_iters):
        theta = theta - X.T.dot(X.dot(theta) - y) * alpha / m

        # Save the cost J in every iteration
        J_history[iter] = computeCost(X, y, theta)

    return theta, J_history