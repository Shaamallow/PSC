######## TUTORIAL 
# Basic tutorial of gensim core concepts

import pprint
import gensim as gm

### BASIC CONCEPTS
# Document : 
# litteraly a string (from tweet-long to book-long)

# Corpus :
# Collection of documents 
# 2 Roles : 
# - Input Training 
# - Input Inference


#### --- TEST --- ####

# - Define elements

document = "Human machine interface for lab abc computer applications"
text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

# - Preprocessing

# C'est la partie que j'avais implémenté de manière relativement nulle avec nltk
# Globalement on va regarder la doc de cette fonction [gensim.utils.simple_preprocess()](https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess)

# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once in corpus
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint.pprint(processed_corpus)

# Give ID to each word

dictionary = gm.corpora.Dictionary(processed_corpus)
print(dictionary)

# REMARQUE : 
# Ici on load tout le corpus dans la mémoire, en pratique c'est vachement pas opti
# Donc a regarder après : [Corpus Stream](https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#corpus-streaming-tutorial)

#### --- END TEST --- ####

#### --- VECTOR --- ####

# Comme avant, un document = vecteur de mots : Bag of Words
# On fait une représentation + légère avec les ID des mots (et pas les mots en eux-mêmes)
# On a les ID dans le dictionnaire
# doc_vec = [(1,5),(2,3),(3,1)] => 5 fois le mot 1, 3 fois le mot 2, 1 fois le mot 3
# On note ca le vecteur dense car on a l'information sur le mot dont on parle
# Il existe le vecteur creux si on sait à l'avance de quel mot on parle mais c'est pas très pratique je trouve
# En pratique on va garder le vecteur dense et omettre les mots qui n'apparaissent pas dans le document
# Tous les couples de la forme (x,0) pour un sérieux gain de place 

# GET IDS : 
pprint.pprint(dictionary.token2id)
# On note que c'est print dans l'ordre alphabétique et pas dans l'ordre d'apparition dans le corpus = Ordre des tokens

# New document => Vectorization
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)

# Corpus => Vectorization
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
pprint.pprint(bow_corpus)

#### --- END VECTOR --- ####

#### --- MODELS --- ####

# Tout plein d'exemples de modèles : (https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html#)

# - TF-IDF

tfidf = gm.models.TfidfModel(bow_corpus)

# Exemple avec une projection sur le BagOfWords formé par le corpus
words = "system minors".lower().split()
print(tfidf[dictionary.doc2bow(words)])

# Similarity : 

index = gm.similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)


# Test avec un nouveau doc 

query_document = 'system engineering'.split()
query_bow = dictionary.doc2bow(query_document)
sims = index[tfidf[query_bow]]

# List la similarité avec tous les documents du corpus
print(list(enumerate(sims)))