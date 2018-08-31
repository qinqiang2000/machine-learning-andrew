import numpy as np
import matplotlib.pyplot as plt 

data = np.loadtxt("ex1data1.txt", delimiter=',') # read comma separated data
X = data[:, 0] # 向量
y = data[:, 1]
m = y.size # number of training examples
X = X.reshape(m, 1) # 向量转矩阵
y = y.reshape(m, 1)

#Plot the data to see what it looks like
# plt.plot(X, y, 'rx', markersize=10)
# plt.ylabel('Profit in $10,000s')
# plt.xlabel('Population of City in 10,000s')
# plt.show()

X = np.column_stack((np.ones(m), X)) # Add a column of ones to x
theta = np.zeros((2,1)) # initialize fitting parameters
iterations = 1500
alpha = 0.01

# Compute cost for linear regression
# J = computeCost(X, y, theta) computes the cost of using theta as the
# parameter for linear regression to fit the data points in X and y
def computeCost(X, y, theta):
    m = y.size  # number of training examples
    J = 0
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

    plt.plot(range(len(J_history)), J_history, 'bo')
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    plt.show()

    return theta, J_history

gradientDescent(X, y, theta, alpha, iterations)