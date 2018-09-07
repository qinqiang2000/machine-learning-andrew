import numpy as np

# calculate the sigmoid function
def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))

def h(thea, X):
	return X.dot(thea)

def costFunction(thea, X, y):
	m, n = X.shape
	thea = thea.reshape((n, 1))

	hx = sigmoid(h(thea, X))
	J =  -1 / m * (y.T.dot(np.log(hx)) + (1 - y).T.dot(np.log(1 - hx)))
	
	return np.sum(J.flatten())

def gradient(thea, X, y):
	m, n = X.shape
	
	thea = thea.reshape((n, 1))

	hx = sigmoid(h(thea, X))
	grad = X.T.dot(hx - y) / m
	return grad.flatten()