######## TUTORIAL 
# Basic tutorial of gensim core concepts

import pprint
import logging
import gensim as gm
from collections import defaultdict

# Pourquoi logging :
# c'est un auxiliaire pour avoir plus d'info que des prints un peu partout 
# recommendation du tuto => OK j'écoute

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Corpus creation (cf core_concept for details)

documents = [
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

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

pprint.pprint(texts)

# Dictionnary of ID :

dictionary = gm.corpora.Dictionary(texts)

# Save dic in folder : 
#dictionnary.save('.')

# Print tokens for word 
#print(dictionnary.token2id)
# To create a modele need to bow a document => 
# Count each relevant word, uses his ID and give a vector of the couple : (wordID, count in doc)

#### --- Corpus Streaming --- ####

# allow to use enormous corpus without overloading RAM 

from smart_open import open  # for transparently opening remote files


class MyCorpus:
    def __iter__(self):
        for line in open('https://radimrehurek.com/mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
print(corpus_memory_friendly)

for vector in corpus_memory_friendly:  # load one vector into memory at a time
    print(vector)

# collect statistics about all tokens
dictionary = gm.corpora.Dictionary(line.lower().split() for line in open('https://radimrehurek.com/mycorpus.txt'))
# remove stop words and words that appear only once
stop_ids = [
    dictionary.token2id[stopword]
    for stopword in stoplist
    if stopword in dictionary.token2id
]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)

# Market Matrix Formatting a corpus

corpus = [[(1, 0.5)], []]  # make one document empty, for the heck of it

# SAVE CORPUS
gm.corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus_memory_friendly)

# Load corpus 

corpus2 = gm.corpora.MmCorpus('/tmp/corpus.mm')

# Corpus is a stream to avoid overloading RAM :

# 1st Method : load everything in RAM ...
print(list(corpus2))

# 2nd Method : Use stream interface 
for doc in corpus2:
    print(doc)

