import numpy as np

def multivariateGaussian(X, mu, sigma2):
    """ 
    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability 
    density function of the examples X under the multivariate gaussian 
    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
    treated as the covariance matrix. If Sigma2 is a vector, it is treated
    as the sigma^2 values of the variances in each dimension (a diagonal
    covariance matrix) """

    m, n = X.shape
    if(np.ndim(sigma2) == 1):
        sigma2 = np.diag(sigma2)
    
    # np.linalg.det()：矩阵求行列式（标量）
    norm = 1./(np.power((2*np.pi), n/2)*np.sqrt(np.linalg.det(sigma2)))
    exp = np.zeros((m,1))

    for row in range(m):
        exp[row] = np.exp(-0.5*((X[row]-mu).T).dot(np.linalg.inv(sigma2)).dot(X[row]-mu))

    return norm * exp