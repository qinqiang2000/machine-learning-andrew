import numpy as np

def normalizeRatings(Y, R):
    """ 
    NORMALIZERATINGS Preprocess data by subtracting mean rating for every 
    movie (every row)
       [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
       has a rating of 0 on average, and returns the mean rating in Ymean.
    """

    Yrating = Y * R
    Ymean = Yrating.sum(axis=1) / R.sum(axis=1)
    Ymean = Ymean.reshape(-1, 1)
    Ynorm = (Yrating - Ymean) * R

    return Ynorm, Ymean
