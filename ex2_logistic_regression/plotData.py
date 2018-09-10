import matplotlib.pyplot as plt
import numpy as np
from mapFeature import *

## PLOTDATA Plots the data points X and y into a new figure 
#   PLOTDATA(x,y) plots the data points with + for the positive examples
#   and o for the negative examples. X is assumed to be a Mx2 matrix.
def plotData(X, y, pos_label="+", neg_label="-"):
	pos = np.where(y.ravel() == 1)
	neg = np.where(y.ravel() == 0)
	
	plt.scatter(X[neg,0], X[neg,1], marker='o', color = 'y', label = neg_label)
	plt.scatter(X[pos,0], X[pos,1], marker='+', color = 'k', label = pos_label)
	plt.legend(loc = 'upper right') 
	return plt
	
#PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
#the decision boundary defined by theta
#   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
#   positive examples and o for the negative examples. X is assumed to be 
#   a either 
#   1) Mx3 matrix, where the first column is an all-ones column for the 
#      intercept.
#   2) MxN, N>3 matrix, where the first column is all-ones

def plotDecisionBoundary(theta, X, y, mylambda=0., pos_label="+", neg_label="-"):
	plotData(X[:,1:], y, pos_label, neg_label)

	if X.shape[1] <= 3:
		# Only need 2 points to define a line, so choose two endpoints
		plot_x = np.array([np.min(X[:,1]) - 2, np.max(X[:,1]) + 2])

		# Calculate the decision boundary line
		plot_y = -1 / theta[2] * (plot_x.dot(theta[1]) + theta[0])

		plt.plot(plot_x, plot_y, color='b')

		plt.show()
	else:
		u = np.linspace(-1,1.5,50)
		v = np.linspace(-1,1.5,50)
		z = np.zeros((len(u), len(v)))

		for i in range(0, len(u)):
			for j in range(0, len(v)):
				myfeaturesij = mapFeature(np.array([u[i]]), np.array([v[j]]))
				z[i][j] = np.dot(theta,myfeaturesij.T)

		z = z.transpose()
		x1, x2 = np.meshgrid(u, v)
		mycontour = plt.contour(x1, x2, z, [0])

