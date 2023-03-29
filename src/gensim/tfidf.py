### IMPORT LIBS 

import os
import pdfplumber
import re
import gensim as gm
import numpy as np
import nltk as sl

from termcolor import colored

### DEFINE PATH and folder

path = os.getcwd()
corpus_path = '/data/corpus/corpus1'

files = os.listdir(path+corpus_path)

### EXTRACT .txt files from folder

def get_txt_from_folder(files):
    # Get all .txt files in list of files
    document_list = []
    for k in range(len(files)):
        if files[k][-4:] == ".txt":
            document_list.append(files[k])
    return document_list

### IMPORT CORPUS

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __iter__(self):
        for doc_path in files:
            print('Process doc : ' + doc_path)
            with open(path+corpus_path+"/"+doc_path, "r") as f:
                text = f.read()
            text = re.sub('[^a-zA-Z]+', ' ', text)
            # remove double space
            text = re.sub(' +', ' ', text)

            # lower case
            text = text.lower()

            # remove stop words
            stop_words = sl.corpus.stopwords.words('english')
            text = [w for w in text.split() if not w in stop_words]

            # stem words
            stemmer = sl.stem.PorterStemmer()
            result = []
            for w in text:
                if w=='cid' or stemmer.stem(w)=='cid':
                    pass
                result.append(stemmer.stem(w))
            yield text

files = get_txt_from_folder(files)

corpus = MyCorpus()

### BUILD DICTIONARY

dictionary = gm.corpora.Dictionary(corpus)

### BUILD BAG OF WORDS

bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]

### BUILD TFIDF MODEL

tfidf = gm.models.TfidfModel(bow_corpus)

### ORDER list of tuples by tfidf value

def order_tuples_by_tfidf(tuples):
    # Order tuples by tfidf value
    return sorted(tuples, key=lambda x: x[1], reverse=True)

### EXTRACT WORDS
for doc in tfidf[bow_corpus]:
    l = order_tuples_by_tfidf(doc)
    for k in range(5):
        print(dictionary[l[k][0]], l[k][1])
    print()

### SIM 

index = gm.similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=len(dictionary))

### for each doc, get the 5 most similar docs

for doc in tfidf[bow_corpus]:
    sims = index[doc]
    l = order_tuples_by_tfidf(enumerate(sims))
    print('Doc : ', files[l[0][0]])
    for k in range(1,15):
        value = str(int(l[k][1]*100)) + '%'
        print(files[l[k][0]], colored(value,'green'))
    print()