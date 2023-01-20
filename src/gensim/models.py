######## TUTORIAL 
# Different models in gensim

# [LINK REF](https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html#)

# Try to load corpus3 with streaming Interface

# Import all txt file in a directory

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

# Import all docs in folder as corpus with streaming interface

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

# Build a dictionary

dictionary = gm.corpora.Dictionary(corpus)

# Build a bag of words

bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]

# Build a tfidf model

tfidf = gm.models.TfidfModel(bow_corpus)

# Print the tfidf model

""" for doc in tfidf[bow_corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])
 """
# Build a LSI model

lsi = gm.models.LsiModel(tfidf[bow_corpus], id2word=dictionary, num_topics=20)

# Print the LSI topics 
lsi_topics = lsi.print_topics(-1)

#for doc in lsi_topics:
#    print(doc)

# Save Useful stuff 

dictionary.save(path+'/src/gensim/models/dictionary.dict')
tfidf.save(path+'/src/gensim/models/tfidf.model') 
lsi.save(path+'/src/gensim/models/lsi.model')
