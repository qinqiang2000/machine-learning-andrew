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

# other's code
def f(params,*args):
    X_train,y_train,reg = args
    m,n = X_train.shape
    J = 0
    theta = params.reshape((n,1))
    h = out(X_train,theta)
    theta_1 = theta[1:,:]
    J = -1*np.sum(y_train*np.log(h) + (1-y_train)*np.log((1-h))) / m +\
        + 0.5 * reg * theta_1.T.dot(theta_1) / m
    
    return J

def gradf(params,*args):
    X_train,y_train,reg = args
    m,n = X_train.shape
    theta = params.reshape(-1,1)
    h = out(X_train,theta)
    grad = np.zeros((X_train.shape[1],1))
    theta_1 = theta[1:,:]
    grad = X_train.T.dot((h-y_train)) / m
    grad[1:,:] += reg*theta_1/m  #theta0 without reg
    g = grad.ravel()
    return g  