import numpy as np
from scipy.special import expit #Vectorized sigmoid function

#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)

def predict(Theta1, Theta2, X):
  # Add ones to the X data matrix
  a1 = np.c_[np.ones(X.shape[0]), X]
  print(a1.shape, Theta1.shape, Theta2.shape)

  a2 = expit(Theta1.dot(a1.T))
  a2 = np.row_stack((np.ones(a2.shape[1]), a2))
  print("a2 shape:", a2.shape)

  a3 = expit(Theta2.dot(a2))
  print(a3.shape)

  pred = np.argmax(a3, axis=0) + 1 # 这里的数字是1-10，而不是0-9

  return pred
  

