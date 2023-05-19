# Enhance LSI output

import os
import re
import gensim as gm
import numpy as np
import nltk as sl

path = os.getcwd()
corpus_path = '/data/corpus/corpus1'

# get list of txt file in folder :
corpus_list = os.listdir(path+corpus_path)
document_list = []

for k in range(len(corpus_list)):
    if corpus_list[k][-4:] == ".txt":
        document_list.append(corpus_list[k])

corpus_list = document_list

# Load corpus 

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __iter__(self):
        for doc_path in corpus_list:
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
            text = [stemmer.stem(w) for w in text]
            yield text

corpus = MyCorpus()

# Load dictionary/tfidf/lsi model

dictionary = gm.corpora.Dictionary.load(path+'/src/gensim/models/dictionary.dict')
tfidf = gm.models.TfidfModel.load(path+'/src/gensim/models/tfidf.model')
lsi = gm.models.LsiModel.load(path+'/src/gensim/models/lsi.model')

# Print the lsi model

topics = lsi.print_topics(-1)
i= 0
for doc in topics:
    print(corpus_list[i])
    print(doc)
    i +=1

print()
print()

# Get top N words of TFIDF model for a specific document 


N = 50
specific_doc = 'reportsstategy-statementdepartment-of-education-and-skills-statement-of-strategy-2015-2017.txt'


with open(path+corpus_path+"/"+specific_doc, "r") as f:
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
text = [stemmer.stem(w) for w in text]

doc2 = text
doc2_tfidf = tfidf[dictionary.doc2bow(doc2)]

# oder tfidf by relevance :

sorted_doc = []


def add_sort(l,element):
    # insersion sort
    k=0
    while k < len(l):
        if l[k][1] >= element[1]:
            return l[:k]+[element]+l[k:]
        if l[k][1]<element[1]:
            k += 1
    return l + [element]

for element in doc2_tfidf:
    sorted_doc = add_sort(sorted_doc,element)


for k in range(1,N):
    print(sorted_doc[-k])
    print(dictionary[sorted_doc[-k][0]])

