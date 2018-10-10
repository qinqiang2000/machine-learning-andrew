import numpy as np
import pandas as pd
import re #regular expression for e-mail processing

# This is one possible porter stemmer 
# (note: I had to do a pip install stemming)
# https://pypi.python.org/pypi/stemming/1.0
# from stemming.porter2 import stem # 这是一个可用的英文分词算法(Porter stemmer)

# This porter stemmer seems to more accurately duplicate the
# porter stemmer used in the OCTAVE assignment code
# (note: I had to do a pip install nltk)
# I'll note that both stemmers have very similar results
import nltk, nltk.stem.porter  # 这个英文算法似乎更符合作业里面所用的代码，与上面效果差不多




"""
GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
cell array of the words
   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt 
   and returns a cell array of the words in vocabList. """
def getVocabList():
    # Store all dictionary words in cell array vocab{}
    # Total number of words in the vocab.txt is 1899
    df = pd.read_table('vocab.txt', names=['words'])
    vocab = df.values

    return vocab

""" 
我们做如下如处理：
  1. Lower-casing: 把整封邮件转化为小写。
  2. Stripping HTML: 移除所有HTML标签，只保留内容。
  3. Normalizing URLs: 将所有的URL替换为字符串 “httpaddr”.
  4. Normalizing Email Addresses: 所有的地址替换为 “emailaddr”
  5. Normalizing Dollars: 所有dollar符号($)替换为“dollar”.
  6. Normalizing Numbers: 所有数字替换为“number”
  7. Word Stemming(词干提取): 将所有单词还原为词源。例如，“discount”, “discounts”, “discounted” and “discounting”都替换为“discount”。
  8. Removal of non-words: 移除所有非文字类型，所有的空格(tabs, newlines, spaces)调整为一个空格.
 """
def preProcessEmail(email):
    # Make the entire e-mail lower case
    email = email.lower()

    # Strip html tags (strings that look like <blah> where 'blah' does not
    # contain '<' or '>')... replace with a space
    email = re.sub('<[^<>]+>', ' ', email)

    #Any numbers get replaced with the string 'number'
    email = re.sub('[0-9]+', 'number', email)
    # email = re.sub('[\d]+', 'number', email)

    # Anything starting with http or https:// replaced with 'httpaddr'
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email) # 匹配//后面不是空白字符的内容，遇到空白字符则停止

    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email);

    #The '$' sign gets replaced with 'dollar'
    email = re.sub('[$]+', 'dollar', email);

    return email

"""
    Function that takes in preprocessed (simplified) email, tokenizes it,
    stems each word, and returns an (ordered) list of tokens in the e-mail
"""
def email2TokenList( raw_email ):
    # I'll use the NLTK stemmer because it more accurately duplicates the
    # performance of the OCTAVE implementation in the assignment
    stemmer = nltk.stem.porter.PorterStemmer()

    email = preProcessEmail(raw_email)

    #Split the e-mail into individual words (tokens) (split by the delimiter ' ')
    #but also split by delimiters '@', '$', '/', etc etc
    #Splitting by many delimiters is easiest with re.split()
     # 将邮件分割为单个单词，re.split() 可以设置多种分隔符
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)

    #Loop over each word (token) and use a stemmer to shorten it,
    #then check if the word is in the vocab_list... if it is,
    #store what index in the vocab_list the word is
    tokenList = []
    for token in tokens:
        # Remove any non alphanumeric characters
        token = re.sub('[^a-zA-Z0-9]', '', token)

        # Use the Porter stemmer to stem the word
        stemmed = stemmer.stem(token)

        # 去除空字符串‘’，里面不含任何字符
        if not len(token): continue

        #Store a list of all unique stemmed words
        tokenList.append(stemmed)
    
    return tokenList

""" PROCESSEMAIL preprocesses the body of an email and
returns a list of word_indices 
   word_indices = PROCESSEMAIL(email_contents) preprocesses 
   the body of an email and returns a list of indices of the 
   words contained in the email.  """
def processEmail(email):
    # Preprocess and tokenList Email 
    tokenList = email2TokenList(email)

    #  Load Vocabulary
    vocabList = getVocabList()
    print(vocabList.shape)

    # Look up the word in the dictionary and add to word_indices if found
    word_indices = np.zeros(len(vocabList))

    """ index_list = [i for i in range(len(vocabList)) if vocabList[i] in tokenList]
    for idx in index_list:
        word_indices[idx] = 1 """

    for i, vocab in enumerate(vocabList):
        if(vocab in tokenList):
            word_indices[i] = 1
    
    return word_indices