# load from models with pandas

import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np

path = os.getcwd()
path_model = path+"/src/gensim/models"

df = pd.read_csv(path_model+"/coords.csv")

x_vals = df['x']
y_vals = df['y']
z_vals = df['z']

labels = df['label']

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



def plot_embeddings_cluster(x_vals, y_vals, z_vals, words, method = 'kmeans', n=6):
    '''
    Plot in a 2D-graph the words vectors, highlighting a number n of clusters. We are using the method k-means.

    ## Params:
        M_reduced (numpy matrix of shape (num_corpus_words, dim=2)): co-occurence matrix reduced by reduce_to_k_dim
        word2ind: dictionary mapping each word to its row number in M_reduced 
        words: array of the words vectors we are plotting
    '''
    if method == 'kmeans':
        clf = KMeans(n_clusters= n)
    if method == 'dbscan':
        clf = DBSCAN(eps=0.3, min_samples=10)
    #clf = KMeans(n_clusters= n) # Modify here the number of clusters
    X = x_vals
    Y = y_vals
    Z = z_vals
    
    d= {'X' : X, 'Y' : Y, 'Z' : Z}
    df = pd.DataFrame(data = d)
    clf.fit(df)
    labels = clf.labels_
    print(labels)
    # Generate n random colors

    #colors = [np.random.rand(3,) for i in range(n)]
    #colors = ['g.', 'r.', 'b.', 'c.', 'y.', 'm.'] # Modify this array if you use more than 6 clusters (add more colors)
    # Plot the points using plt.scatter with the colors according to the cluster they belong to
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_vals, y_vals, z_vals, cmap='rainbow', c=labels)

    # Label randomly subsampled 100 data points

    indices = list(range(len(words)))
    selected_indices = random.sample(indices, 100)
    for i in selected_indices:
        ax.text(x_vals[i], y_vals[i], z_vals[i], words[i])
    

    plt.show()

def plot_embeddings_cluster2(x_vals, y_vals, words, n = 6):
    '''
    Plot in a 2D-graph the words vectors, highlighting a number n of clusters. We are using the method k-means.

    ## Params:
        M_reduced (numpy matrix of shape (num_corpus_words, dim=2)): co-occurence matrix reduced by reduce_to_k_dim
        word2ind: dictionary mapping each word to its row number in M_reduced 
        words: array of the words vectors we are plotting
    '''
    clf = KMeans(n_clusters= n) # Modify here the number of clusters
    X = x_vals
    Y = y_vals
    
    d= {'X' : X, 'Y' : Y}
    df = pd.DataFrame(data = d)
    clf.fit(df)
    labels = clf.labels_
    # Generate n random colors

    #colors = [np.random.rand(3,) for i in range(n)]
    #colors = ['g.', 'r.', 'b.', 'c.', 'y.', 'm.'] # Modify this array if you use more than 6 clusters (add more colors)
    # Plot the points using plt.scatter with the colors according to the cluster they belong to
    plt.figure(figsize = (20,16))
    plt.scatter(X, Y, c=labels, cmap='rainbow')
    plt.show()

def elbow_method(x_vals, y_vals, z_vals, words, max_n = 10):
    '''
    Plot the elbow method to find the optimal number of clusters

    ## Params:
        
    '''

    X = x_vals
    Y = y_vals
    Z = z_vals
    
    d= {'X' : X, 'Y' : Y, 'Z' : Z}
    df = pd.DataFrame(data = d)
    sse = []
    list_k = list(range(1, max_n))
    for k in list_k:
        km = KMeans(n_clusters=k)
        km.fit(df)
        sse.append(km.inertia_)
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, '-o')
    plt.xlabel('Number of clusters *k*')
    plt.ylabel('Sum of squared distance')

    plt.show()

def elbow_method2(x_vals, y_vals, words, max_n = 10):
    '''
    Plot the elbow method to find the optimal number of clusters

    ## Params:
        
    '''

    X = x_vals
    Y = y_vals
    
    d= {'X' : X, 'Y' : Y}
    df = pd.DataFrame(data = d)
    sse = []
    list_k = list(range(1, max_n))
    for k in list_k:
        km = KMeans(n_clusters=k)
        km.fit(df)
        sse.append(km.inertia_)
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, '-o')
    plt.xlabel('Number of clusters *k*')
    plt.ylabel('Sum of squared distance')

    plt.show()

#plot_embeddings_cluster2(x_vals, y_vals, labels, n = 4)
#elbow_method2(x_vals, y_vals, labels, max_n = 15)
#elbow_method(x_vals, y_vals, z_vals, labels, max_n = 15)
#plot_embeddings_cluster(x_vals, y_vals, z_vals, labels, n = 7)

### Print the words of each cluster

def print_cluster_words(x_vals, y_vals, z_vals, words, n = 6):
    '''
    Print the words of each cluster

    ## Params:
        
    '''

    clf = KMeans(n_clusters= n) # Modify here the number of clusters
    X = x_vals
    Y = y_vals
    Z = z_vals
    
    d= {'X' : X, 'Y' : Y, 'Z' : Z}
    df = pd.DataFrame(data = d)
    clf.fit(df)
    labels = clf.labels_

    # Print 15 random words of each cluster
    for i in range(n):
        indices = np.where(labels == i)[0]
        selected_indices = random.sample(list(indices), 15)
        print('Cluster', i)
        for j in selected_indices:
            print(words[j])
        print('')
    
print_cluster_words(x_vals, y_vals, z_vals, labels, n = 7)
plot_embeddings_cluster(x_vals, y_vals, z_vals, labels,method = 'dbscan')