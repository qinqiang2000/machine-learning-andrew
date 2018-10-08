## Machine Learning Online Class
#  Exercise 6 | Support Vector Machines

import numpy as np
import scipy.io as sio   # Used to load the OCTAVE *.mat files
import matplotlib.pyplot as plt
from sklearn import svm
from plotData import *
from gaussianKernel import *

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

for C in [1, 100]:
    clf = svm.SVC(C, kernel='linear') # model
    clf.fit(X, y.ravel()) # train

    #看一下训练的准确率
    y1_pred = clf.predict(X)
    acc_train = np.mean(y1_pred == y.ravel())
    print("C = %.1f, the accuracy of train data set: %f" %(C, acc_train))

    plt.figure()
    plotData(X, y)
    plotBoundary(clf, X)
    plt.title('SVM Decision Boundary with C = {} (Example Dataset 1'.format(C))

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
plt.figure()
plotData(X, y)
# plt.show()

## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the 
#  SVM classifier.

printf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...')

#  SVM Parameters
C = 1; sigma = 0.1;
