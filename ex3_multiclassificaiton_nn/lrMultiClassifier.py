## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#     lrCostFunction.py (logistic regression cost function)
#     oneVsAll.py
#     predictOneVsAll.py
#     predict.py

import numpy as np
import scipy.io as sio   # Used to load the OCTAVE *.mat files
import random   #To pick random images to display
from displayData import *
from lrCostFunction import lrCostFunction, lrGradient
from oneVsAll import *
from predictOneVsAll import predictOneVsAll

## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
img_width = 20
num_labels = 10;          # 10 labels, from 1 to 10 # 

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#
mat = sio.loadmat('ex3data1.mat')
X, y = mat['X'], mat['y']
m = X.shape[0]

# 样本中，0用10替代了（MATLAB数组从1开始的原因），所以这里将其改回0
print(np.unique(y))
y[y==10] = 0
print(np.unique(y))

# Randomly select 100 data points to display
sel_indics = random.sample(range(m), 100)
sel = X[sel_indics, :]
# displayData(sel, img_width)

## ============ Part 2a: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.

# Test case for lrCostFunction
print('Testing lrCostFunction() with regularization')

theta_t = np.array([[-2], [-1], [1], [2]])
X_t = np.c_[np.ones((5, 1)), np.reshape(np.arange(1, 16)/10, (3, 5)).T]
y_t = np.array([[1], [0], [1], [0], [1]])
lambda_t = 3
J = lrCostFunction(theta_t, X_t, y_t, lambda_t)
grad = lrGradient(theta_t, X_t, y_t, lambda_t)

print('Cost: #f', J)
print('Expected cost: 2.534819')
print('Gradients:')
print(' #f ', grad)
print('Expected gradients:')
print(' 0.146561 -0.548558 0.724722 1.398003\n')

## ============ Part 2b: One-vs-All Traini1ng ============
lamda = 0.1

all_theta = oneVsAll(X, y, num_labels, lamda)

## ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X)

print('Training Set Accuracy: ', np.mean(y.ravel() == pred) )