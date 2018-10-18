import numpy as np
import matplotlib.pyplot as plt
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids

def plotProgresskMeans(X, centroids, K, idx):
    # K = centroids.shape[0]

    colors = ("r","g","b","c","m","y","k","w")
    
    for i in range(K):
        t = np.where(idx.ravel() == i + 1)
        ic = i % len(colors)
        plt.plot(X[t,0], X[t,1], colors[ic] + "o", markerfacecolor='w')
        
    cs = np.array(centroids)
    x, y = [], []
    for c in cs:
        x.append(c[:,0])
        y.append(c[:,1])
    
    x = np.array(x)
    y = np.array(y)
    for i in range(x.shape[1]):
        plt.plot(x[:,i], y[:,i], "kx--")

    plt.show()


def runkMeans(X, initial_centroids, max_iters, plot_progress):
    """     
    RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
       [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
       plot_progress) runs the K-Means algorithm on data matrix X, where each 
       row of X is a single example. It uses initial_centroids used as the
       initial centroids. max_iters specifies the total number of interactions 
       of K-Means to execute. plot_progress is a true/false flag that 
       indicates if the function should also plot its progress as the 
       learning happens. This is set to false by default. runkMeans returns 
       centroids, a Kxn matrix of the computed centroids and idx, a m x 1 
       vector of centroid assignments (i.e. each entry in range [1..K]) 
    """
    if plot_progress: plt.figure()

    K = initial_centroids.shape[0]
    centroids = []
    idx = []
    centroids.append(initial_centroids)

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids[len(centroids) - 1])
        c = computeCentroids(X, idx, K)
        # print("{}:\n{}\n".format(i, c))

        centroids.append(c)
        if plot_progress: 
            plotProgresskMeans(X, centroids, K, idx)
            
    return centroids[len(centroids) - 1], idx