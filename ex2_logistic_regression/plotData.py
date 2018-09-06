import matplotlib.pyplot as plt
import numpy as np

## PLOTDATA Plots the data points X and y into a new figure 
#   PLOTDATA(x,y) plots the data points with + for the positive examples
#   and o for the negative examples. X is assumed to be a Mx2 matrix.
def plotData(X, y):
	pos = np.where(y.ravel() == 1)
	neg = np.where(y.ravel() == 0)
	
	plt.scatter(X[neg,0], X[neg,1], marker='o', color = 'y', label = 'Not admitted')
	plt.scatter(X[pos,0], X[pos,1], marker='+', color = 'k', label = 'admitted')
	plt.xlabel('Exam 1 score')
	plt.ylabel('Exam 2 score')
	plt.legend(loc = 'upper left')
	plt.show()