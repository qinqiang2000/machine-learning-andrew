import numpy as np
import matplotlib.pyplot as plt
import featureNormalize as fNorm
import gradientDescentMulti as gdm
import normalEqn as neqn

# ================ Part 1: Feature Normalization ================
data = np.loadtxt("ex1data2.txt", delimiter=',') # read comma separated data
X = data[:, 0:-1] # 读取除最后一列外的所有列
y = data[:, -1:]
m = y.size # number of training examples

X_norm, mu, sigma = fNorm.featureNormalize(X)

# =================== Part 2: Cost and Gradient descent ===================
X_norm = np.column_stack((np.ones(m), X_norm)) # Add a column of ones to x
# Choose some alpha value
alpha = 0.01
num_iters = 1500

# Init Theta and Run Gradient Descent
theta = np.zeros((3, 1))

theta, J_history = gdm.gradientDescent(X_norm, y, theta, alpha, num_iters)

#Plot the convergence of the cost function
def plotConvergence(jvec):
    plt.figure()
    plt.plot(range(len(jvec)), jvec, 'bo')
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    plt.show()

#Plot convergence of cost function:
# plotConvergence(J_history)

#compute the gradient by using Normal Equations without feature scaling and gradient descent
X = np.column_stack((np.ones(m), X)) # Add a column of ones to x
theta = neqn.normalEqn(X, y)

print(theta)

print("$%0.2f" % np.dot(theta.T,[[1],[1650.],[3]]))