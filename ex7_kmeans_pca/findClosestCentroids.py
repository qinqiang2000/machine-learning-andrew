import numpy as np
import sys

""" 
FINDCLOSESTCENTROIDS computes the centroid memberships for every example
   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
   in idx for a dataset X where each row is a single example. idx = m x 1 
   vector of centroid assignments (i.e. each entry in range [1..K])
 """
def findClosestCentroids(X, centroids):
    K = centroids.shape[0] 
    m = X.shape[0]
    
    idx = np.zeros((m, 1))
    idx = idx.astype(np.int32)

    for i in range(m):
        shortest = sys.maxsize
        for j in range(K):
            dist = np.linalg.norm(X[i,] - centroids[j,])
            if dist < shortest:
                idx[i] = j + 1
                shortest = dist
    
    return idx