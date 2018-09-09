import numpy as np
from plotData import *
from costFunction import *
from mapFeature import *
from scipy.optimize import minimize
from scipy.optimize import fmin_cg

# Load Data
#  The first two columns contains the exam scores and the third column contains the label.
data = np.loadtxt("ex2data2.txt", delimiter=',') # read comma separated data
X = data[:, 0:-1]
y = data[:, -1:]

m = X.shape[0]

# plotData(X, y)

## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(X[:,0], X[:,1])	

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

# Set regularization parameter lambda to 1
lamda=1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost =costFunctionReg(initial_theta, X, y, lamda)
grad = gradientReg(initial_theta, X, y, lamda)

print('Cost at initial theta (zeros):\n', cost)
print('Gradient at initial theta (zeros): \n', grad[:5])

# Compute and display cost and gradient with all-ones theta and lambda = 10
test_theta = np.ones((X.shape[1],1))
print('Cost test again(with lambda = 10): should be 3.16: ', costFunctionReg(test_theta, X, y, 10))
print('Gradient at test theta - first five values only:\n', gradientReg(test_theta, X, y, 10)[:5]);
print('Expected gradients (approx) - first five values only:\n');
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#
input()

## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a sci optimize function (minimize) to find the
#  optimal parameters theta.
result = minimize(fun=costFunction, x0=initial_theta, args=(X, y), method='TNC', jac=gradient)
optimalTheta = result.x

print('Cost at theta found by minimize:', result.fun)
print('theta: ', optimalTheta)
print('Exam 1 score of 45 and an Exam 2 score of 85, admission probability:', out(np.array([1, 45, 85]), optimalTheta))

plotDecisionBoundary(optimalTheta, X, y, 'Admitted', 'Not admitted')	