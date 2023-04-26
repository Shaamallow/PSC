import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import numpy as np
import gensim as gm
import sys

from termcolor import colored

### GLOBAL VARIABLES ###

path = os.getcwd()

### IMPORTATION ###

"""
Script uses a model/corpus/cluster to extract the most representative words of each cluster
"""

# Load dataframe
ID = 7
name_of_doc = 'clusters_3D_kmeans_4.csv'
df = pd.read_csv(path+'/src/gensim/models/corpus'+str(ID)+'/'+name_of_doc)

# Group by cluster

df_grouped = df.groupby('cluster')

# Split the df into a list of df

df_list = [df_grouped.get_group(x) for x in df_grouped.groups]

# Sort by tfidf

for i in range(len(df_list)):
    df_list[i] = df_list[i].sort_values(by=['tfidf'], ascending=False)

# Print the top 10 words of each cluster

for i in range(len(df_list)):
    print('Cluster '+str(i))
    print(df_list[i]['label'].head(15))
    print('')

# create a dataframe with the top 10 words of each cluster

df_top_words = pd.DataFrame()

# each column of the dataframe is a cluster
# the rows are the top 15 words of each cluster

for i in range(len(df_list)):
        print(i)
        print(df_list[i]['label'].head(15))
        df_top_words['cluster '+str(i)] = df_list[i]['label'].head(15).values

# save the dataframe

df_top_words.to_csv(path+'/src/gensim/models/corpus'+str(ID)+'/top_words.csv')
