import numpy as np

# calculate the sigmoid function
def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))

def h(thea, X):
	return X.dot(thea)

def out(X,thea):
	# thea = thea.reshape(thea.shape[0], 1)
	return sigmoid(h(thea, X))

def lrCostFunction(thea, X, y, lamda):
	m, n = X.shape
	thea = thea.reshape((n, 1))
	theta_reg = np.r_[np.zeros((1,1)), thea[1:,:]]

	hx = sigmoid(h(thea, X))
	J =  -1 / m * (y.T.dot(np.log(hx)) + (1 - y).T.dot(np.log(1 - hx))) + lamda / (2*m) * theta_reg.T.dot(theta_reg)
	return J

def lrGradient(thea, X, y, lamda):
	m, n = X.shape
	thea = thea.reshape((n, 1))
	theta_reg=np.r_[np.zeros((1,1)), thea[1:,:]]

	hx = sigmoid(h(thea, X))
	grad = X.T.dot(hx - y) / m + (lamda / m) * theta_reg

	return grad.flatten()