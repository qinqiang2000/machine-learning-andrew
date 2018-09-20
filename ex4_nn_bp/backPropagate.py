# 考虑到scipy的minimize函数不能接受一个函数返回cost和gradient，所以本模块单独计算gradient
import numpy as np
from scipy.special import expit #Vectorized sigmoid function
from sigmoidGradient import sigmoidGradient
from unrollutility import unrollParams

def backPropagate(nn_params, 
                  input_layer_size, 
                  hidden_layer_size, 
                  num_labels, 
                  X, y, lamda):
  
  # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  # for our 2 layer neural network
  Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
  Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

  m = X.shape[0]
  # Part 1: Feedforward the neural network and return the cost in the variable J.
  a1 = np.c_[np.ones(m), X] # 加一列：bias
  z2 = a1.dot(Theta1.T)
  a2 = np.c_[np.ones(m), expit(z2)] # 加一列：bias
  z3 = a2.dot(Theta2.T)
  a3 = expit(z3).T

  print("z2.shape:", z2.shape)
  # 首先把原先label表示的y变成向量模式的output
  y_vec = np.zeros((num_labels, m))
  for i in range(m):
    y_vec[y[i][0] - 1][i] = 1

  # Part 2: Implement the backpropagation algorithm to compute the gradients
  #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
  #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
  #         Theta2_grad, respectively. After implementing Part 2, you can check
  #         that your implementation is correct by running checkNNGradients
  delta3 = a3 - y_vec
  delta2 = Theta2[:,1:].T.dot(delta3) * sigmoidGradient(z2).T
  D2 = delta3.dot(a2)
  D1 = delta2.dot(a1)

  D2 /= m
  D1 /= m

  print("delta3", delta3.shape, "delta2:", delta2.shape, "D2:", D2.shape, "D1", D1.shape)

  return unrollParams([D1, D2])