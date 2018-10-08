import matplotlib.pyplot as plt
import numpy as np

  # Plot training data
def plotData(X, y):
  label1 = np.where(y.ravel() == 1)
  label2 = np.where(y.ravel() == 0)

  plt.plot(X[label1, 0], X[label1, 1], 'k+', label="positive")
  plt.plot(X[label2, 0], X[label2, 1], 'yo', label="negetive")

  plt.xlabel('X1')
  plt.ylabel('X2')
  # plt.legend(loc = 'upper left')

def plotBoundary(clf,  X):
  # Make classification predictions over a grid of values
  x1plot = np.linspace(X[:,0].min() * 1.1, X[:,0].max() * 1.1, 500)
  x2plot = np.linspace(X[:,1].min() * 1.1, X[:,1].max() * 1.1, 500)
  X1, X2 = np.meshgrid(x1plot, x2plot)
  vals = clf.predict(np.c_[X1.ravel(), X2.ravel()])
  vals = vals.reshape(X1.shape)

  # Plot the SVM Boundary 
  plt.contour(X1, X2, vals)
