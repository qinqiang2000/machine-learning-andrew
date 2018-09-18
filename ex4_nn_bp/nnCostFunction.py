#NNCOSTFUNCTION Implements the neural network cost function for a two layer
#neural network which performs classification
#   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, 
#   X, y, lambda) computes the cost and gradient of the neural network. The
#   parameters for the neural network are "unrolled" into the vector
#   nn_params and need to be converted back into the weight matrices. 
# 
#   The returned parameter grad should be a "unrolled" vector of the
#   partial derivatives of the neural network.
#
import numpy as np
from scipy.special import expit #Vectorized sigmoid function

def nnCostFunction(nn_params, 
                  input_layer_size, 
                  hidden_layer_size, 
                  num_labels, 
                  X, y, lamda):
  # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  # for our 2 layer neural network
  Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
  Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

  m = X.shape[0]

  # ====================== YOUR CODE HERE ======================
  # Instructions: You should complete the code by working through the
  #               following parts.
  #
  # Part 1: Feedforward the neural network and return the cost in the
  #         variable J. After implementing Part 1, you can verify that your
  #         cost function computation is correct by verifying the cost
  #         computed in ex4.m
  #
  # Part 2: Implement the backpropagation algorithm to compute the gradients
  #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
  #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
  #         Theta2_grad, respectively. After implementing Part 2, you can check
  #         that your implementation is correct by running checkNNGradients
  #
  #         Note: The vector y passed into the function is a vector of labels
  #               containing values from 1..K. You need to map this vector into a 
  #               binary vector of 1's and 0's to be used with the neural network
  #               cost function.
  #
  #         Hint: We recommend implementing backpropagation using a for-loop
  #               over the training examples if you are implementing it for the 
  #               first time.
  #
  # Part 3: Implement regularization with the cost function and gradients.
  #
  #         Hint: You can implement this around the code for
  #               backpropagation. That is, you can compute the gradients for
  #               the regularization separately and then add them to Theta1_grad
  #               and Theta2_grad from Part 2.
  #
  a1 =  np.c_[np.ones(m), X] # 加一列：bias
  z2 = a1.dot(Theta1.T)
  a2 = np.c_[np.ones(m), expit(z2)] # 加一列：bias
  z3 = a2.dot(Theta2.T)
  h = expit(z3)
  print("h(x) shape:", h.shape)
  
  # 首先把原先label表示的y变成向量模式的output
  y_vec = np.zeros((num_labels, m))
  for i in range(m):
    y_vec[y[i][0] - 1][i] = 1

  #每一training example的cost function是使用的向量计算，然后for loop累加所有m个training example  
  #的cost function 
  J = 0
  for i in range(m):
    J += np.log(h[i,:]).dot(y_vec[:,i]) + np.log(1 - h[i,:]).dot(1 - y_vec[:,i])
  
  J = -J / m

  return J