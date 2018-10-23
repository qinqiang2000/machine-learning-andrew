import numpy as np

def selectThreshold(yval, pval):
    """ 
    [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval). """

    bestF1 = 0
    bestEpsilon = 0

    stepsize = (pval.max() - pval.min()) / 1000
    # Instructions: Compute the F1 score of choosing epsilon as the
    #               threshold and place the value in F1. The code at the
    #               end of the loop will compare the F1 score for this
    #               choice of epsilon and set it to be the best epsilon if
    #               it is better than the current choice of epsilon.
    for epsilon in np.arange(pval.min(), pval.max(), stepsize):
        predictions = pval < epsilon

        tp = np.sum((predictions == 1) & (yval == 1))
        fp = np.sum((predictions == 1) & (yval == 0))
        fn = np.sum((predictions == 0) & (yval == 1))

        prec = tp / (tp + fp) if (tp + fp)  else 0
        rec = tp / (tp + fn) if (tp + fn)  else 0

        F1 = 2 * prec * rec / (prec + rec) if (prec + rec)  else 0
        if F1 > bestF1:
            bestEpsilon = epsilon
            bestF1 = F1

    return bestEpsilon, bestF1
