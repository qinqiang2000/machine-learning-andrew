## Machine Learning Online Class
#  Exercise 6 | Support Vector Machines

import numpy as np
import scipy.io as sio   # Used to load the OCTAVE *.mat files
import matplotlib.pyplot as plt
from sklearn import svm
from plotData import *
from gaussianKernel import *
from dataset3Params import *

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.

# Load Training Data
print('Loading and Visualizing Dataset1 ...')

# Load from ex6data1: 
# You will have X, y in your environment
mat = sio.loadmat('ex6data1.mat')
X, y = mat['X'], mat['y']

# Plot training data
""" 
plt.figure(1)
plotData(X, y)
plt.show()
"""

## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
print('\nTraining Linear SVM ...')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)

plt.figure(figsize=(9,9))
for index, C in enumerate([1, 100]):
    clf = svm.SVC(C, kernel='linear') # model
    clf.fit(X, y.ravel()) # train

    #看一下训练的准确率
    y1_pred = clf.predict(X)
    acc_train = np.mean(y1_pred == y.ravel())
    print("C = %.1f, the accuracy of train data set: %f" %(C, acc_train))

    plt.subplot(211 + index)
    plotData(X, y)
    plotBoundary(clf, X)
    plt.title('SVM Decision Boundary with C = {} (Example Dataset 1)'.format(C))

# plt.show()

## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
print('\nEvaluating the Gaussian Kernel ...')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1]) 
sigma = 2
sim = gaussianKernel(x1, x2, sigma)

print("Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :"
    "\n\t%f\n(for sigma = 2, this value should be about 0.324652)" % (sigma, sim))

## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and 
#  plot the data.     

print('\nLoading and Visualizing Dataset2 ...')

# Load from ex6data2: 
# You will have X, y in your environment
mat = sio.loadmat('ex6data2.mat')
X, y = mat['X'], mat['y']

# Plot training data
plt.figure(figsize=(9,9))
plt.subplot(211)
plotData(X, y)
# plt.show()

## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the 
#  SVM classifier.

print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...')

#  SVM Parameters
C = 1; sigma = 0.1;

# I will use the of-course built-in gaussian kernel in my SVM software
# because it's certainly more optimized than mine.
# It is called 'rbf' and instead of dividing by sigmasquared,
# it multiplies by 'gamma'. As long as I set gamma = sigma^(-2),
# it will work just the same.
gamma = np.power(sigma, -2.)
clf_rbf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
clf_rbf.fit(X, y.ravel()) # train

#看一下训练的准确率
y1_pred = clf_rbf.predict(X)
acc_train = np.mean(y1_pred == y.ravel())
print("C = %.1f, the accuracy of train data set: %f" %(C, acc_train))

plt.subplot(212)
plotData(X, y)
plotBoundary(clf_rbf, X)
plt.title('SVM Decision Boundary with C = {} (Example Dataset 2)'.format(C))
# plt.show()

## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
print('\nLoading and Visualizing Dataset3 ...')

# Load from ex6data3: 
# You will have X, y in your environment
mat = sio.loadmat('ex6data3.mat')
X, y = mat['X'], mat['y']
Xval, yval = mat['Xval'], mat['yval']

# Plot training data
plt.figure(figsize=(9,9))
plt.subplot(211)
plotData(X, y)

## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.

print('\Training SVM with RBF Kernel (Dataset 3) ...')

# Try different SVM Parameters here
C, sigma = dataset3Params(X, y, Xval, yval)

# Train the SVM
clf_rbf = svm.SVC(C=C, kernel='rbf', gamma=np.power(sigma, -2.))
clf_rbf.fit(X, y.ravel()) 

plt.subplot(212)
plotData(X, y)
plotBoundary(clf_rbf, X)
plt.title('SVM Decision Boundary with C = {} (Example Dataset 3)'.format(C))
plt.show()