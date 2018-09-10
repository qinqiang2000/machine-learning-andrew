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
print('Expected gradients (approx) - first five values only:');
print(' 0.3460 0.1614 0.1948 0.2269 0.0922');

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

## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a sci optimize function (minimize) to find the
#  optimal parameters theta.
result = minimize(fun=costFunctionReg, x0=initial_theta, args=(X, y, lamda), method='BFGS'
	, options={"maxiter":500, "disp":False} )
optimalTheta = result.x
print('Cost at theta found by minimize:', result.fun)
print('theta: ', optimalTheta)

#Build a figure showing contours for various values of regularization parameter, lambda#Build a 
#It shows for lambda=0 we are overfitting, and for lambda=100 we are underfitting
plt.figure(figsize=(9,10))

#lambda=1
plt.subplot(221)
plotDecisionBoundary(optimalTheta, X, y)	
plt.title("lambda=1")

#lambda=0
plt.subplot(222)
lamda = 0
result = minimize(fun=costFunctionReg, x0=initial_theta, args=(X, y, lamda), method='BFGS'
	, options={"maxiter":500, "disp":False} )
plotDecisionBoundary(result.x, X, y)	
plt.title("lambda=0")

#lambda=10
plt.subplot(223)
lamda = 10
result = minimize(fun=costFunctionReg, x0=initial_theta, args=(X, y, lamda), method='BFGS'
	, options={"maxiter":500, "disp":False} )
plotDecisionBoundary(result.x, X, y)	
plt.title("lambda=10")

#lambda=100
plt.subplot(224)
lamda = 100
result = minimize(fun=costFunctionReg, x0=initial_theta, args=(X, y, lamda), method='BFGS'
	, options={"maxiter":500, "disp":False} )
plotDecisionBoundary(result.x, X, y)	
plt.title("lambda=100")

plt.show()