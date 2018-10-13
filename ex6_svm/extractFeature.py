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
import multiprocessing
from processEmail import processEmail
from buildVocab import readCorpus

vocabList = []

# 底下是并行计算
def extractFeature(mails):
    return np.array([processEmail(mail, vocabList) for mail in mails])

def parallelExtract(mails):
    m = len(mails)
    c = 8
    k = int(m / c)

    pool = multiprocessing.Pool(processes=c) 

    tasks = dict()
    for i in range(c):
        ti = pool.apply_async(extractFeature, args=(mails[i*k : (i+1)*k],))
        tasks[i] = ti

    if(m % c != 0):
        ti = pool.apply_async(extractFeature, args=(mails[(i+1)*k:m],))
        tasks[i+1] = ti

    pool.close()
    pool.join()

    tasks = sorted(tasks.items(), key=lambda k:k[0])
    tasks = dict(tasks)
    X = np.concatenate([t.get() for t in tasks.values()])

    return X

# Read corpus
nonspam_emails, spam_emails = readCorpus()
print("no spam size={}, spam size={}".format(len(nonspam_emails), len(spam_emails)))

#  Load Vocabulary
vocabList = pd.read_csv('myvocab.csv').values
print(len(vocabList), vocabList[:11])

# Choose the ratio of training set size to cross validation set size to test set size 
# to be roughly 60/20/20. 
# ==========Build the training X matrix and y vector===========
print("[{}]Start extract feature...".format(time.strftime('%H:%M:%S',time.localtime(time.time()))))
s1 = 0.6
s2 = 0.2
m_nonspam_train = int(len(nonspam_emails)*s1)
m_spam_train    = int(len(spam_emails)   *s1)

# 两个训练集合起来
mail_train = np.array(nonspam_emails[:m_nonspam_train] + spam_emails[:m_spam_train])

# 并发提取训练集的特征值
X_train = parallelExtract(mail_train)
y_train = np.concatenate((np.zeros((m_nonspam_train,1)), np.ones((m_spam_train,1))), axis=0)

# 保存结果
pd.DataFrame(X_train).to_csv('X_train.csv',index=False)
pd.DataFrame(y_train).to_csv('y_train.csv',index=False)

# ===========Build the CV X matrix and y vector===========
m_nonspam_cv = int(len(nonspam_emails)*s2)
m_spam_cv    = int(len(spam_emails)   *s2)
mail_cv = np.array(nonspam_emails[m_nonspam_train:m_nonspam_train+m_nonspam_cv] 
                + spam_emails[m_spam_train:m_spam_train+m_spam_cv])

X_cv = parallelExtract(mail_cv)
y_cv = np.concatenate((np.zeros((m_nonspam_cv,1)), np.ones((m_spam_cv,1))), axis=0)

pd.DataFrame(X_cv).to_csv('X_cv.csv',index=False)
pd.DataFrame(y_cv).to_csv('y_cv.csv',index=False)

# ===========Build the test X matrix and y vector===========
m_nonspam_test = len(nonspam_emails) - m_nonspam_train - m_nonspam_cv
m_spam_test    = len(spam_emails)    - m_spam_train - m_spam_cv

mail_test = np.array(nonspam_emails[-m_nonspam_test:] + spam_emails[-m_spam_test:])

X_test = parallelExtract(mail_test)
y_test = np.concatenate((np.zeros((m_nonspam_test,1)), np.ones((m_spam_test,1))), axis=0)

pd.DataFrame(X_test).to_csv('X_test.csv',index=False)
pd.DataFrame(y_test).to_csv('y_test.csv',index=False)

# X_train = np.array([processEmail(mail) for mail in mail_train])
print("[{}]End extract feature".format(time.strftime('%H:%M:%S',time.localtime(time.time()))))

# test
print('\n', y_train.shape, y_cv.shape, y_test.shape)
print('\n', X_train.shape, X_cv.shape, X_test.shape)
print('\n', set(pd.read_csv('X_train.csv').values.flatten()).difference(set(X_train.flatten()))
            ,set(pd.read_csv('X_cv.csv').values.flatten()).difference(set(X_cv.flatten()))
            ,set(pd.read_csv('X_test.csv').values.flatten()).difference(set(X_test.flatten())))