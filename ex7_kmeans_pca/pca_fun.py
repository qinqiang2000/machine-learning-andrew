import numpy as np

def pca(X):
    """ 
    PCA Run principal component analysis on the dataset X
    [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
    Returns the eigenvectors U, the eigenvalues (on diagonal) in S"""

    sigma = X.T @ X / len(X)
    U, S, V = np.linalg.svd(sigma) 

    return U, S, V

