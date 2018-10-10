import numpy as np
from sklearn import svm

""" dataset3Params returns your choice of C and sigma for Part 3 of the exercise
where you select the optimal (C, sigma) learning parameters to use for SVM
with RBF kernel
   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
   sigma. You should complete this function to return the optimal C and 
   sigma based on a cross-validation set. """
def dataset3Params(X, y, Xval, yval):
    """  ====================== YOUR CODE HERE ======================
    Instructions: Fill in this function to return the optimal C and sigma
                learning parameters found using the cross validation set.
                You can use svmPredict to predict the labels on the cross
                validation set. For example, 
                    predictions = svmPredict(model, Xval);
                will return the predictions on the cross validation set.

    Note: You can compute the prediction error using 
            mean(double(predictions ~= yval)) """
    
    max_acc = 0
    CBest = 0
    sigmaBest = 0
    Cvals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigmaVals = Cvals
    for C in Cvals:
        for sigma in sigmaVals:
            gamma = np.power(sigma, -2.)
            clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
            clf.fit(X, y.ravel()) # train
            
            yval_pred = clf.predict(Xval)
            acc_train = np.mean(yval_pred == yval.ravel())
            if acc_train > max_acc:
                max_acc = acc_train
                CBest = C
                sigmaBest = sigma
                # print("({},{})".format(CBest, sigmaBest))
    
    print("Best C, sigma pair is (%.2f, %.2f) with a score of %f."
            % (CBest, sigmaBest, max_acc))
    return CBest, sigmaBest
            

