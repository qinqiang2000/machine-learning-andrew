import random as rd

def kMeansInitCentroids(X, K):
    """ 
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X centroids = KMEANSINITCENTROIDS(X, K) 
    returns K initial centroids to be used with the K-Means on the dataset X """

    # Take the first K examples as centroids
    randidx = rd.sample(range(X.shape[0]), K)
    centroids = X[randidx, :]

    return centroids