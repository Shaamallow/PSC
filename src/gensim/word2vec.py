# Pour debug

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Pour load le modèle
# Using CBOW (continuous bag of words) model

import re
import numpy as np
import nltk as sl
import os
from matplotlib import pyplot as plt
import gensim.downloader as api
import time
import pandas as pd
import gensim.models
import sys

from termcolor import colored

### GLOBAL VARIABLES ###

path = os.getcwd()
corpus_path = '/data/corpus/corpus4'


files = os.listdir(path+corpus_path)

### IMPORTATION ###

# Define Methods to load files and clean text

def get_txt_from_folder(files):
    # Get all .txt files in list of files
    document_list = []
    for k in range(len(files)):
        if files[k][-4:] == ".txt":
            document_list.append(files[k])
    return document_list


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
            
            yield text
            


#### REDUCE THE DIMENSION

# reduce dimensions for plotting

from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
#from sklearn.decomposition import TruncatedSVD
import random

def reduce_dimensions(wv, n=2):
    
    num_dimensions = n  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(wv.vectors)
    labels = np.asarray(wv.index_to_key)  # fixed-width numpy strings
    #print(vectors)
    #print(labels)

    print("1st method starting...")
    t1 = time.time()
    # reduce using t-SNE
    vectors = TSNE(n_components=num_dimensions).fit_transform(vectors)
    t2 = time.time()

    print("Time to reduce dimension : ", end='')
    print(colored(t2-t1, 'red'))

    print("### END ###")
    # extract the coords of the vectors

    return ([[v[i] for v in vectors] for i in range(num_dimensions)], labels)

def reduce_dimensions2(vectors, labels, **kwargs):
    """
    Reduce the dimension for plotting

    ## Parameters : 
    - vectors : list of vectors
    - labels : list of labels
    - n : Desired dimension (default 2)
    - method : Method to reduce dimension (default tsne)

    ## Output : 

    - coords : matrix of size n x len(vectors) with the coordinates of the vectors
    - labels : list of labels 
    
    """

    if 'n' in kwargs:
        n = kwargs['n']
    else:
        n = 2
    
    if 'method' in kwargs:
        if kwargs['method'] not in ['tsne', 'pca', 'svd']:
            raise ValueError('method must be one of "tsne", "pca", "svd"')
        method = kwargs['method']
    else:
        method = 'tsne'
    
    num_dimensions = n  # final num dimensions (2D, 3D, etc)

    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # Benchmark the differetns process :

    print("Starting the dimension reduction process...")
    t1 = time.time()
    # reduce using t-SNE
    if method == 'tsne':
        vectors = TSNE(n_components=num_dimensions).fit_transform(vectors)
    elif method == 'pca':
        vectors = IncrementalPCA(n_components=num_dimensions).fit_transform(vectors)

    t2 = time.time()
    print("Time to reduce dimension : ", end='')
    print(colored(t2-t1, 'red'))
    # extract the coords of the vectors

    return ([[v[i] for v in vectors] for i in range(num_dimensions)], labels)

#### MAIN ####

if __name__ == '__main__':

    # Get args from command line

    dimension = 2
    if len(sys.argv) > 1:
        dimension = int(sys.argv[1])

    ### PROCESSING

    files = get_txt_from_folder(files)

    corpus = MyCorpus()

    # Shit ton of parameters to play with
    # min_count, vector_size ... (https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)
    #model = gensim.models.Word2Vec(sentences=corpus)

    # LOAD THE MODEL 

    #model = gensim.models.Word2Vec.load(path+'/src/gensim/models/w2v')

    # Version with glove but only words in corpus 

    print("\nLoading model...")
    glove = api.load("glove-twitter-25")
    print("Done\n")

    # Construct manualy the array of words in the corpus using the glove model 

    vectors = []
    noms = []

    for doc in corpus:
        for word in doc:
            if word in glove and not word in noms:
                vectors.append(glove[word])
                noms.append(word)

    coords, labels = reduce_dimensions2(vectors, noms, method='pca', n=dimension)


    print("\nSaving the data...")
    df = pd.DataFrame(coords)
    df = df.transpose()
    if dimension == 2:
        name = 'coords2D.csv'
        df.columns = ['x', 'y']
    elif dimension == 3:
        name = 'coords3D.csv'
        df.columns = ['x', 'y', 'z']

    df['label'] = labels

    #df.to_csv(path+'/src/gensim/models/corpus4/coords2D.csv', index=False)
    df.to_csv(path+'/src/gensim/models/pca/'+name, index=False)
    print("Done")