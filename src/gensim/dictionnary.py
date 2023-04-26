# Import all txt file in a directory

import os
import re
import gensim as gm
import numpy as np
import nltk as sl

doc = """Je suis une phrase de démonstration. Je suis aussi un document"""
doc1 = """Je suis une nouvelle phrase de démonstration pour faire un corpus"""
doc2 = """Je suis un autre document pour faire un test"""
doc3 = """Me voici, un document étrange pour m'assurer que tout va bien"""
doc4 = """La démonstration est un processus mathématique logique"""

corpus = [doc, doc1, doc2, doc3, doc4]
# Build a dictionary

dictionary = gm.corpora.Dictionary([doc.lower().split() for doc in corpus])

# Build a bag of words

corpus_bow = [dictionary.doc2bow(doc.lower().split()) for doc in corpus]

# Build a tfidf model

tfidf = gm.models.TfidfModel(corpus_bow)

# Exemple avec une projection sur le BagOfWords formé par le corpus
words = "Je suis un autre document pour faire un test"

print(words)
words_bow = dictionary.doc2bow(words.lower().split())
print(words_bow)
print(tfidf[words_bow])

# Plot histogram of the documents 

import matplotlib.pyplot as plt




print()
