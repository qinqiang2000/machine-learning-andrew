import numpy as np

#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)

def predict(Theta1, Theta2, X):
  # Add ones to the X data matrix
  X = np.c_[np.ones(X.shape[0]), X]