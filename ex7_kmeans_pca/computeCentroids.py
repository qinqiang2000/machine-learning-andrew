import numpy as np

""" 
COMPUTECENTROIDS returns the new centroids by computing the means of the 
data points assigned to each centroid.
   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
   computing the means of the data points assigned to each centroid. It is
   given a dataset X where each row is a single data point, a vector
   idx of centroid assignments (i.e. each entry in range [1..K]) for each
   example, and K, the number of centroids. You should return a matrix
   centroids, where each row of centroids is the mean of the data points
   assigned to it.
 """
def computeCentroids(X, idx, K):
    n = X.shape[1]

    centroids = np.zeros((K, n))

    for i in range(K):
        centroids[i,:] = np.mean(X[np.where(idx.ravel() == i + 1)], axis=0)

    return centroids