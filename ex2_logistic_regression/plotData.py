import matplotlib.pyplot as plt
import numpy as np

## PLOTDATA Plots the data points X and y into a new figure 
#   PLOTDATA(x,y) plots the data points with + for the positive examples
#   and o for the negative examples. X is assumed to be a Mx2 matrix.
def plotData(X, y, xlabel='', ylabel='', pos_label="+", neg_label="-"):
	pos = np.where(y.ravel() == 1)
	neg = np.where(y.ravel() == 0)
	
	plt.scatter(X[neg,0], X[neg,1], marker='o', color = 'y', label = neg_label)
	plt.scatter(X[pos,0], X[pos,1], marker='+', color = 'k', label = pos_label)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend(loc = 'upper rigt') 
	return plt
	
#PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
#the decision boundary defined by theta
#   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
#   positive examples and o for the negative examples. X is assumed to be 
#   a either 
#   1) Mx3 matrix, where the first column is an all-ones column for the 
#      intercept.
#   2) MxN, N>3 matrix, where the first column is all-ones

def plotDecisionBoundary(theta, X, y, pos_label="+", neg_label="-"):
	plotData(X[:,1:], y, 'exam 1', 'exam 2', pos_label, neg_label)

	if X.shape[1] <= 3:
		# Only need 2 points to define a line, so choose two endpoints
		plot_x = np.array([np.min(X[:,1]) - 2, np.max(X[:,1]) + 2])

		# Calculate the decision boundary line
		plot_y = -1 / theta[2] * (plot_x.dot(theta[1]) + theta[0])

		plt.plot(plot_x, plot_y, color='b')

		plt.show()









