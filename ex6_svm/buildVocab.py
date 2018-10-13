import numpy as np
import pandas as pd
import os
import sys
import codecs
import time
from processEmail import processEmail, email2TokenList
from collections import Counter

def getEmails(dirs):
    emails = []
    for dir_ in dirs:
        files = os.listdir(dir_)
        for filename in files:
            if filename[0] == '.':
                continue

            df = codecs.open(dir_ + '/' +  filename, 'r',encoding= u'utf-8',errors='ignore')
            one_email = df.read()

            if "\n\n" not in one_email: continue
            
            emails.append(one_email)
    
    return emails

def readCorpus(data_dir = 'publiccorpus/'): 
    nonspam_emails = getEmails([data_dir + 'easy_ham', data_dir + 'easy_ham_2'])
    spam_emails = getEmails([data_dir + 'spam_2 2'])

    # Since all of the e-mails (spam and non-spam) have a double newline,
    #  I'm going to be assuming the first occurence of the double newline designates
    #  the end of the header garbage. Now I loop through all e-mails and erase the headers.
    nonspam_emails = [e[e.find("\n\n"):] for e in nonspam_emails]
    spam_emails = [e[e.find("\n\n"):] for e in spam_emails]

    return nonspam_emails, spam_emails

""" 
Build a feature matrix, Xtrain with shape ( n, 1000 ) [because my vocab dictionary is 1000 words long],
which is formed by stacking together a feature vector for each e-mail. 
This matrix will contain both spam and nonspam e-mails. 
"""
def buildVocab(size = 1000):
    nonspam_emails, spam_emails = readCorpus()

    # I accomplish this in a somewhat hacky way. 
    # I concatenate all emails to form one giant e-mail, tokenize and stem it, 
    # and use a function from within the collections module to count the number of occurrences of each word. 
    # I'll make a collections.Counter object, 
    # which is a high performance python container with super useful functionalities like the most_common(N) function, 
    # which returns the N most common tokens in the collection. I.E. exactly what I want!
    giant_nonspam_email = [ ' '.join(nonspam_emails[:]) ]
    giant_spam_email    = [ ' '.join(spam_emails[:])    ]
    giant_total_email   = giant_nonspam_email[0] + giant_spam_email[0]

    print("[{}]Start tokenizing...".format(time.strftime('%H:%M:%S',time.localtime(time.time()))))
    tokens = email2TokenList(giant_total_email)
    print("[{}]End tokenizing...".format(time.strftime('%H:%M:%S',time.localtime(time.time()))))

    word_counts = Counter(tokens)
    print('The total number of unique stemmed tokens is: ', len(word_counts))

    # The number of unique stemmed words throughout all of the e-mails is very large... 
    # perhaps too large for each unique stemmed word to be used as a feature for the SVM to 
    # consider when it attempts to classify an e-mail as spam or non-spam
    # Build our own vocabulary list (by selecting the high frequency words hat occur in the dataset)
    vocabList = [str(x[0]) for x in word_counts.most_common(size)]
    pd.DataFrame(vocabList).to_csv('myvocab.csv',index=False)

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        buildVocab(int(sys.argv[1]))
    else:
        buildVocab()             