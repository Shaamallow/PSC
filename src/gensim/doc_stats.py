# Script to display stats in a corpus of doc using the csv associated

import pandas as pd
import os
import matplotlib.pyplot as plt
import sys

from termcolor import colored

# Path variable

# EXECUTE FROM THE ROOT OF THE PROJECT 
path = os.getcwd()
ID = 4 # change this to the ID of the corpus you want to clean
path = path+"/data/"+'corpus/corpus'+str(ID)

df = pd.read_csv(path+'/stats.csv', index_col=0)

# Display stats

# Order by number of words
df = df.sort_values(by=['size'], ascending=False)
print(df)

print('\n')

# Order by size_removed_percentage
df = df.sort_values(by=['size_removed_percent'], ascending=False)
print(df)