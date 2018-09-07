import numpy as np
from plotData import plotData
from costFunction import costFunction
from costFunction import gradient
from scipy.optimize import minimize

# Load Data
#  The first two columns contains the exam scores and the third column contains the label.
data = np.loadtxt("ex2data1.txt", delimiter=',') # read comma separated data
X = data[:, 0:-1] # 读取除最后一列外的所有列
y = data[:, -1:]

m = X.shape[0]

## ==================== Part 1: Plotting ====================
# plotData(X, y)

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. 

#  Setup the data matrix appropriately, and add ones for the intercept term
X = np.column_stack((np.ones(m), X)) # Add a column of ones to x
n = X.shape[1]

# Initialize fitting parameters
# initial_theta = np.zeros((n, 1))
initial_theta = np.zeros(n)
print(initial_theta.shape)

# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)

print('Cost at initial theta (zeros):\n', cost)
print('Gradient at initial theta (zeros): \n', grad)

## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.
result = minimize(fun=costFunction, x0=initial_theta, args=(X, y), method='TNC', jac=gradient)
optimalTheta = result.x

print('Cost at theta found by fminunc:', result.fun)
print('theta: \n', optimalTheta)