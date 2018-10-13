""" 
In this code, I will be using samples of spam and non-spam e-mails supplied 
by the SpamAssassin Public Corpus which I will subdivide into three groups 
as I mentioned earlier: a training set, a cross validation set, and a test set. 
An outline of what we're about to do is:
1. Read in the e-mails.
2. Preprocess the e-mails to turn their text into a format an SVM can understand.
3. Subdivide the e-mails into the three sets (training, CV, test).
4. Initialize an SVM, try different values of the C parameter, training the SVM on the training set each time.
5. Run each trained SVM on the cross validation set and determine which value of C is most appropriate.
6. Run the best trained SVM on the test set and report how well it worked.
 """

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

# we read training set, cross-validation set, and test set which we generated in extractFeatrue.py
X_train = pd.read_csv('X_train.csv').values
X_cv = pd.read_csv('X_cv.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values
y_cv = pd.read_csv('y_cv.csv').values
y_test = pd.read_csv('y_test.csv').values

print("X_train.shape={}, X_cv.shape={}, X_test.shape={}".format(X_train.shape, X_cv.shape, X_test.shape))
print("y_train.shape={}, y_cv.shape={}, y_test.shape={}".format(y_train.shape, y_cv.shape, y_test.shape))

Cvals = [ 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0 ]
cv_acc = []
test_acc = []

print("C\tcv accuracy\ttest accuracy")
for C in Cvals:
    # Make an instance of an SVM with C=myC and 'linear' kernel
    clf = svm.SVC(C=C, kernel='linear')

    # Now we fit the SVM to our X_train matrix, given the labels y_train
    clf.fit( X_train, y_train.flatten() )
    print(C, "\t%.4f"%clf.score(X_cv, y_cv), "\t\t%.4f" % clf.score(X_test, y_test))

    # Determine how well this SVM works by computing the
    # classification accuracy rate on the cross-validation set
    acc = clf.score(X_cv, y_cv)
    cv_acc.append(acc)

    # While we're at it, do the same for the training set error
    acc = clf.score(X_test, y_test)
    test_acc.append(acc)

plt.figure()
plt.plot(Cvals, cv_acc, "ro--", label='CV Set accuracy')
plt.plot(Cvals, test_acc, "bo--", label='Test Set accuracy')
plt.grid(True, 'both')
plt.xlabel('$C$ Value',)
plt.ylabel('Classification accuracy')
plt.title('Choosing a $C$ Value')
plt.xscale('log')
myleg = plt.legend()
plt.show()