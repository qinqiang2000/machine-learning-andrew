import numpy as np

vocabList = []

# 底下是并行计算
def extractFeature(mails):
    print(vocabList)
    # return np.array([processEmail(mail, vocabList) for mail in mails])

def parallelExtract(mails):
    extractFeature(mails)

vocabList = [1,2,3,1,11,1]

parallelExtract("")