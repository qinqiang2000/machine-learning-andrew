import numpy as np
from debugInitializeWeights import *
from nnCostFunction import *
from computeNumericalGradient import *

""" CHECKNNGRADIENTS Creates a small neural network to check the
backpropagation gradients
   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
   backpropagation gradients, it will output the analytical gradients
   produced by your backprop code and the numerical gradients (computed
   using computeNumericalGradient). These two gradient computations should
   result in very similar values.
 """
def checkNNGradients(lamda = 0):
  input_layer_size = 3;
  hidden_layer_size = 5;
  num_labels = 3;
  m = 5;

  # We generate some 'random' test data
  Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
  Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
  # Reusing debugInitializeWeights to generate X
  X = debugInitializeWeights(m, input_layer_size - 1)
  y = np.mod(np.arange(1, m + 1), num_labels).T + 1
  y = np.reshape(y, (y.shape[0], 1))

  # Unroll parameters
  nn_params = unrollParams([Theta1, Theta2])

  costFunc = lambda w: nnCostFunction(w, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lamda)

  # 计算梯度
  J, grad = costFunc(nn_params)  
  # 计算数值梯度
  numgrad = computeNumericalGradient(costFunc, nn_params)

  # print(np.c_[grad,numgrad])

  # Evaluate the norm of the difference between two solutions.  
  # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
  # in computeNumericalGradient.m, then diff below should be less than 1e-9
  diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
  print('If your backpropagation implementation is correct,\n'
  'the relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\n'
  'Relative Difference: {}'.format(diff))