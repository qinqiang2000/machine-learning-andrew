import numpy as np

# MAPFEATURE Feature mapping function to polynomial features
#
#   MAPFEATURE(X1, X2) maps the two input features
#   to quadratic features used in the regularization exercise.
#
#   Returns a new feature array with more features, comprising of 
#   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
#
#   Inputs X1, X2 must be the same size
#
def mapFeature(X1, X2, degree = 6):
	out = np.ones((X1.shape[0], 1))

	for i in range(1, degree + 1):
		for j in range(i + 1):
			tmp = (X1 ** (i-j)) * (X2 ** j)
			out = np.column_stack((out, tmp))

	return out