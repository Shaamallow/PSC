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

path = os.getcwd()
corpus_path = '/data/corpus/corpus4'


files = os.listdir(path+corpus_path)


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
            
# Load corpus

files = get_txt_from_folder(files)

corpus = MyCorpus()



# Create and Train Model
import gensim.models
# Shit ton of parameters to play with
# min_count, vector_size ... (https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)
#model = gensim.models.Word2Vec(sentences=corpus)

# LOAD THE MODEL 

model = gensim.models.Word2Vec.load(path+'/src/gensim/models/w2v')

# Version with glove but only words in corpus 

print("Loading model...")
glove = api.load("glove-twitter-25")
print("Done")

# Construct manualy the array of words in the corpus using the glove model 

vectors = []
noms = []

for doc in corpus:
    for word in doc:
        if word in glove and not word in noms:
            vectors.append(glove[word])
            noms.append(word)


#### PCA

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
    print(vectors)
    print(labels)

    # reduce using t-SNE
    vectors = TSNE(n_components=num_dimensions).fit_transform(vectors)

    # extract the coords of the vectors

    return ([[v[i] for v in vectors] for i in range(num_dimensions)], labels)

def reduce_dimensions2(vectors, labels, n=2):
    
    num_dimensions = n  # final num dimensions (2D, 3D, etc)

    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    print(vectors)
    print(labels)

    # reduce using t-SNE
    vectors = TSNE(n_components=num_dimensions).fit_transform(vectors)

    # extract the coords of the vectors

    return ([[v[i] for v in vectors] for i in range(num_dimensions)], labels)

# plot the data

def plot_with_matplotlib(x_vals, y_vals, labels):
    

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))


coords, labels = reduce_dimensions(model.wv, 3)
coords, labels = reduce_dimensions2(vectors, noms, 3)

x_vals = coords[0]
y_vals = coords[1]
# Don't forget to comment to 2D/3D plot
z_vals = coords[2]

print(len(labels))
print(len(coords[0]))
print(len(vectors))
print(len(noms))
# 3D plot

from mpl_toolkits.mplot3d import Axes3D

def plot_with_matplotlib_3d(x_vals, y_vals, z_vals, labels):
        
    
        random.seed(0)
    
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_vals, y_vals, z_vals)
    
        #
        # Label randomly subsampled 25 data points
        #
        indices = list(range(len(labels)))
        selected_indices = random.sample(indices, 25)
        for i in selected_indices:
            ax.text(x_vals[i], y_vals[i], z_vals[i], labels[i])
                    
        plt.show()


### 2D plot

#plot_with_matplotlib(x_vals, y_vals, labels)


### 3D plot

#plot_with_matplotlib_3d(x_vals, y_vals, z_vals, labels)

# Save coords and label 

import pandas as pd

df = pd.DataFrame(coords)
df = df.transpose()
#df.columns = ['x', 'y']
df.columns = ['x', 'y', 'z']
df['label'] = labels

#df.to_csv(path+'/src/gensim/models/corpus4/coords2D.csv', index=False)
df.to_csv(path+'/src/gensim/models/glove/coords3D.csv', index=False)