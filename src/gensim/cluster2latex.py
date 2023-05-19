import pandas as pd
import os
import sys
from tabulate import tabulate

path = os.getcwd()
print(path)

total_path = path + '/src/gensim/models/corpus7/top_words.csv'

df = pd.read_csv(total_path)

latex_table = tabulate(df, headers='keys', tablefmt='latex', showindex=False)

# Save the table in a .tex file

with open(path+"/src/gensim/models/corpus7/top_words.tex", 'w') as f:
    f.write(latex_table)