import numpy as np

# calculate the sigmoid function
def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))

def h(thea, X):
	return X.dot(thea)

def costFunction(thea, X, y):
	m = y.shape[0]
	hx = sigmoid(h(thea, X))
	J =  -1 / m * (y.T.dot(np.log(hx)) + (1 - y).T.dot(np.log(1 - hx)))
	grad = X.T.dot(hx - y) / m
	return J, grad