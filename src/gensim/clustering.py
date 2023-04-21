# load from models with pandas

import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import numpy as np
import sys

### LOAD VALUES ###



from mpl_toolkits.mplot3d import Axes3D

### TOOLS ###

def plot_embeddings_cluster(coords : np.ndarray, words : pd.DataFrame, method = 'kmeans', **param) -> None:
    '''
    Plot a 2D or 3D graph of the words vectors, highlighting a number n of clusters. 
    Using the given method

    ## Params:
        coords ndarray of the words vectors
        words: array of the words vectors we are plotting
        method: the method used to cluster the words vectors
        n_cluster (int): number of clusters to plot (default = 10)
        eps (float): the maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): the number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.

    ## Ouput: 
        Show the Graph
        
    '''

    if method == 'kmeans':

        if 'n_cluster' in param:
            n = param['n']
        else:
            n = 10

        clf = KMeans(n_clusters= n, n_init='auto')
    if method == 'dbscan':

        if 'eps' in param:
            eps = param['eps']
        else:
            eps = 0.3
        
        if 'min_samples' in param:
            min_samples = param['min_samples']
        else:
            min_samples = 10
        
        
        clf = DBSCAN(eps=eps, min_samples=min_samples)
    if method == "optics":
        
        if 'eps' in param:
            max_eps = param['eps']
        else:
            max_eps = np.inf

        if 'min_samples' in param:
            min_samples = param['min_samples']
        else:
            min_samples = 10

        clf = OPTICS(min_samples=min_samples, max_eps=max_eps)
    
    df = pd.DataFrame(data = coords)
    clf.fit(df)
    labels = clf.labels_


    #print(labels)
    # Plot the points using plt.scatter with the colors according to the cluster they belong to
    fig = plt.figure(figsize=(12, 12))
    if coords.shape[1] == 2:
        ax = fig.add_subplot(111)
        ax.scatter(*coords.transpose(), cmap='rainbow', c=labels)
    if coords.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*coords.transpose(), cmap='rainbow', c=labels)

    # Label randomly subsampled 100 data points

    indices = list(range(len(words)))
    selected_indices = random.sample(indices, 100)
    for i in selected_indices:
        ax.text(*coords[i], words[i])
    

    plt.show()

def elbow_method(coords, method = 'kmeans',  n = 10) -> None:
    '''
    Plot the elbow method to find the optimal number of clusters

    ## Params:
        max_n (int): maximum number of clusters to test
        coords : coordinates of the points we are plotting
    '''

    df = pd.DataFrame(data = coords)
    sse = []


    if method == 'kmeans':

        for k in range(1,n+1):
            print("Computing Elbow method for KMeans with k = ", k, " ...", end="\r", flush=True)
            kmeans = KMeans(n_clusters=k, n_init='auto')
            kmeans.fit(df)
            sse.append(kmeans.inertia_)

        plt.figure(figsize=(6, 6))
        plt.title('Elbow Method with ' + str(len(coords[0])) + ' dimension')
        plt.plot(range(1,n+1), sse, '-o')
        # add name to figure
    
        plt.xlabel('Number of clusters *k*')
        plt.ylabel('Sum of squared distance')

    plt.show()

    if method == 'dbscan':

        print("Not available YET")

### DEBUG FUNCTIONS ###
def cluster_themes(coords : np.ndarray, words : pd.DataFrame, path : str,  method = 'kmeans', **param) -> np.ndarray:
    '''
    Save the clusters in a csv file

    ## Params:
        coords ndarray of the words vectors
        words: array of the words vectors we are plotting
        method: the method used to cluster the words vectors
        n_cluster (int): number of clusters to plot (default = 10)
        eps (float): the maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): the number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.

    ## Ouput: 
        Updated coords with labels
    '''

    if method == 'kmeans':

        if 'n_cluster' in param:
            n = param['n']
        else:
            n = 10

        clf = KMeans(n_clusters= n, n_init='auto')
    if method == 'dbscan':

        if 'eps' in param:
            eps = param['eps']
        else:
            eps = 0.3
        
        if 'min_samples' in param:
            min_samples = param['min_samples']
        else:
            min_samples = 10
        
        
        clf = DBSCAN(eps=eps, min_samples=min_samples)
    if method == "optics":
        
        if 'eps' in param:
            max_eps = param['eps']
        else:
            max_eps = np.inf

        if 'min_samples' in param:
            min_samples = param['min_samples']
        else:
            min_samples = 10

        clf = OPTICS(min_samples=min_samples, max_eps=max_eps)
    
    df = pd.DataFrame(data = coords)
    clf.fit(df)
    labels = clf.labels_

    df = pd.DataFrame(data = words)
    df['labels'] = labels
    df = df.sort_values(by=['labels'])
    # format name with : dimension, method, parameters
    name = "clusters" + "_" + str(len(coords[0])) + "_" + method + "_" + str(param) + ".csv"
    print("Saving the dataframe at ", path+name, " ...", end="\n")
    #print(path)
    df.to_csv(path+name, index=False)


    

def avg_distance(coords):
    # random sample of 1000 points
    indices = list(range(len(coords)))
    selected_indices = random.sample(indices, 1000)
    coords = coords[selected_indices]
    # compute the average distance between each point and its 10 nearest neighbors
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    avg_distances = np.mean(distances, axis=1)
    return avg_distances



if __name__ == "__main__":

    if len(sys.argv) > 1:
        """
        argv[1] = path to the file containing the vectors to use 
        argv[2] = method to use
        argv[3] = parameters        
        """
        name_path = sys.argv[1]
        if len(sys.argv) > 2:
            method = sys.argv[2]
        else :
            method = 'kmeans'
    else :
        method = 'kmeans'
        name_path = "coords2D.csv"    

    print("Running main")

    
    path = os.getcwd()
    path_model = path+"/src/gensim/models/"

    print("Loading data from ", path_model+name_path, " ...")

    df = pd.read_csv(path_model+name_path)

    # get the columns
    keys = df.keys()
    # get all keys except label
    keys = keys[:-1]
    coords = df[keys].values

    labels = df['label']
    
    if method == "kmeans":
        param = {'n': 10}
    if method == "dbscan":
        param = {'eps': 0.3, 'min_samples': 10}
    if method == "optics":
        param = {'eps': 0.3, 'min_samples': 10}

    cluster = cluster_themes(coords, labels, path_model, method = method, **param)
    #print("Plotting the clusters ...")
    #plot_themes(labels, cluster, path_model+"themes.csv")

    #plot_embeddings_cluster(coords, labels, method = method, **param)
    #plot_embeddings_cluster(coords, labels, method = 'dbscan', eps = 0.3, min_samples = 10)
    #plot_embeddings_cluster(coords, labels, method = 'optics', eps = 0.3, min_samples = 10)
    #elbow_method(coords, method = 'kmeans', n = 10)