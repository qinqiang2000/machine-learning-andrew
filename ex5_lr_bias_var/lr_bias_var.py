## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance

import numpy as np
import scipy.io as sio   # Used to load the OCTAVE *.mat files
import matplotlib.pyplot as plt
from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve

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
plt.plot(X, y, 'rx')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
# plt.show()

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
lamda = 0;
theta = trainLinearReg(X, y, lamda);

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

error_train, error_val = learningCurve(X, y, np.insert(Xval, 0, 1, axis=1), yval, 0);

plt.figure()
plt.plot(np.arange(1, m + 1), error_train, np.arange(1, m + 1), error_val);
plt.title('Learning curve for linear regression')
plt.legend('Train', 'Cross Validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim((0, 13))
plt.ylim((0, 150))

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t%d\t\t%f\t%f\n' %(i, error_train[i], error_val[i]))

plt.show()