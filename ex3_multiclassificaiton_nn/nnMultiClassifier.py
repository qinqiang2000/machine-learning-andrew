## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

#  Instructions
#  ------------
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy as np
import scipy.io as sio   # Used to load the OCTAVE *.mat files
import random   #To pick random images to display
from displayData import *
from lrCostFunction import lrCostFunction, lrGradient
from oneVsAll import *

## Setup the parameters you will use for this exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

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
#   displayData(sel, img_width)

# Load the weights into variables Theta1 and Theta2
mat2 = sio.loadmat('ex3weights.mat')
Theta1, Theta2 = mat2['Theta1'], mat2['Theta2']
print("Theta1 has shape:",Theta1.shape)
print("Theta2 has shape:",Theta2.shape)

## ================= Part 2: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);


## ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X)

print('Training Set Accuracy: ', np.mean(y.ravel() == pred) )