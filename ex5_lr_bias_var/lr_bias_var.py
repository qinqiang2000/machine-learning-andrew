## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance

import numpy as np
import scipy.io as sio   # Used to load the OCTAVE *.mat files
import matplotlib.pyplot as plt
from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from plotData import plotData
from plotFit import plotFit
from validationCurve import validationCurve

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment
mat = sio.loadmat('ex5data1.mat')
X, y, Xval, yval, Xtest, ytest = mat['X'], mat['y'], mat['Xval'], mat['yval'], mat['Xtest'], mat['ytest']
m = X.shape[0]
print('X.shape=',X.shape)

# Plot training data
plotData(X, y)

X = np.insert(X, 0, 1, axis=1)

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#

theta = np.array([[1], [1]])
J, grad = linearRegCostFunction(theta, X, y, 1)

print('\nCost at theta = [1 ; 1]: %f \n(this value should be about 303.993192)' % J);

print('Gradient at theta = [1 ; 1]:  [%f; %f]'
         '\n(this value should be about [-15.303016; 598.250744])\n' % (grad[0], grad[1]) )

## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great fit.
#

#  Train linear regression with lambda = 0
lamda = 0
theta = trainLinearReg(X, y, lamda)

#  Plot fit over the data
pred = X @ theta
plt.plot(X[:,1:], pred, 'b--')
# plt.show()

## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
#

error_train, error_val = learningCurve(X, y, np.insert(Xval, 0, 1, axis=1), yval, 0)

plt.figure()
plt.plot(np.arange(1, m + 1), error_train, np.arange(1, m + 1), error_val)
plt.title('Learning curve for linear regression')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim((0, 13))
plt.ylim((0, 150))

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t%d\t\t%f\t%f\n' %(i, error_train[i], error_val[i]))

# plt.show()

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#
p = 6

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X[:, 1:], p)
X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
X_poly = np.insert(X_poly, 0, 1 , axis=1)     # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = (X_poly_test - mu) / sigma
X_poly_test = np.insert(X_poly_test, 0, 1 , axis=1)     # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = (X_poly_val - mu) / sigma
X_poly_val = np.insert(X_poly_val, 0, 1 , axis=1)     # Add Ones

## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with 
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

lamda = 0;
theta = trainLinearReg(X_poly, y, lamda)

# Plot training data and fit
plotData(X[:,1:], y)
plotFit(X[:,1:].min(), X[:,1:].max(), mu, sigma, theta, p)
plt.title('Polynomial Regression Fit (lambda = %.2f)' % lamda)

error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, 0)

plt.figure()
plt.plot(np.arange(1, m + 1), error_train, np.arange(1, m + 1), error_val)
plt.title('Polynomial Learing Curve (lambda = %.2f)' % lamda)
plt.xlabel('Number of training examples')
plt.ylabel('Error')

print('Polynomial Regression (lambda = %f)' % lamda)
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t%d\t\t%f\t%f\n' %(i, error_train[i], error_val[i]))

# plt.show()

## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of 
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.

lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

plt.figure()
plt.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.legend(['Train', 'Cross Validation'], loc = 0, ncol = 2)
plt.xlabel('lambda');
plt.ylabel('Error');

print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
	print(' %f\t%f\t%f\n' %(lambda_vec[i], error_train[i], error_val[i]))

plt.show()