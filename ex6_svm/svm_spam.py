"""  
Machine Learning Online Class
Exercise 6 | Spam Classification with SVMs """

import numpy as np
import scipy.io as sio   # Used to load the OCTAVE *.mat files
from sklearn import svm
from processEmail import processEmail, getVocabList

## ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.
print("Preprocessing and extracting features sample email (emailSample1.txt)")

# Extract Features
with open('emailSample1.txt', 'r') as f:
    features =processEmail(f.read())

print('length of vector = {}\nnum of non-zero = {}'
        .format(len(features), int(features.sum())))

## =========== Part 2: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
mat = sio.loadmat('spamTrain.mat')
X, y = mat['X'], mat['y']

print("\nTraining Linear SVM (Spam Classification)")
print("(this may take 1 to 2 minutes) ...")

C = 0.1
clf = svm.SVC(C, kernel='linear') # model
clf.fit(X, y.ravel()) # train

p = clf.predict(X)

print("C = %.1f, the accuracy of train data set: %f" %(C, clf.score(X, y)))

## =================== Part 3: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
mat = sio.loadmat('spamTest.mat')
Xtest, ytest = mat['Xtest'], mat['ytest']

print("C = %.1f, the accuracy of test data set: %f" %(C, clf.score(Xtest, ytest)))

## ================= Part 4: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#

# Sort the weights and obtin the vocabulary list
idx = np.argsort(clf.coef_, axis=None )[::-1] #[::-1] 表示数组反转
vocabList = getVocabList()

print('\nTop predictors of spam: ')
for i in range(15):
    j = idx[i]
    print("{} ".format(vocabList[j]))
