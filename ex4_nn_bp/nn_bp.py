## Machine Learning Online Class - Exercise 4 Neural Network Learning
#
import numpy as np
import scipy.io as sio   # Used to load the OCTAVE *.mat files
import random   #To pick random images to display
from displayData import displayData
from unrollutility import unrollParams
from nnCostFunction import nnCostFunction

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)
img_width = 20

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#
mat = sio.loadmat('ex4data1.mat')
X, y = mat['X'], mat['y']
m = X.shape[0]

# 样本中，0用10替代了(MATLAB数组从1开始的原因)
print(np.unique(y))

# Randomly select 100 data points to display
sel_indics = random.sample(range(m), 20)
sel = X[sel_indics, :]

# for idx, val in enumerate(y[sel_indics]):
#   print(idx, " ", val)

# displayData(sel, img_width)

## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

# Load the weights into variables Theta1 and Theta2
mat2 = sio.loadmat('ex4weights.mat')
Theta1, Theta2 = mat2['Theta1'], mat2['Theta2']
print("Theta1 has shape:",Theta1.shape, Theta1[0,0], Theta1[-1,-1])
print("Theta2 has shape:",Theta2.shape, Theta2[0,0], Theta2[-1,-1])

#  Unroll parameters 
nn_params = unrollParams([Theta1, Theta2])

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('Feedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
lamda = 0

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lamda)

print("Cost at parameters (loaded from ex4weights): ", J
      ,"(this value should be about 0.287629)")