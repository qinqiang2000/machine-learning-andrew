import numpy as np

# NORMALEQN(X,y) computes the closed-form solution to linear
# regression using the normal equations.
# NORMALEQN Computes the closed-form solution to linear regression
def normalEqn(X, y):
    return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)