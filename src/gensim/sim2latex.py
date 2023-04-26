import pandas as pd
import os
import sys
from tabulate import tabulate

### PATH :

path = os.getcwd()
print(path)

ID = 6
sim_path = "data/corpus/corpus"+str(ID)+"/removed"+"/similarities.csv"

# Check if the ID is correct
if not os.path.exists(path+"/"+sim_path):
    print("The ID is not correct, please check the ID of the corpus you want to clean")
    sys.exit()

# Read the stats.csv file
df = pd.read_csv(path+"/"+sim_path)

# Order by 1st column
df = df.sort_values(by=[df.columns[0]], ascending=False)

latex_table = tabulate(df, headers='keys', tablefmt='latex', showindex=False)

# Save the table in a .tex file

with open(path+"/data/corpus/corpus"+str(ID)+"/removed"+"/similarity.tex", 'w') as f:
    f.write(latex_table)

print("The table has been saved in a .tex file in the folder : "+path+"/data/corpus/corpus"+str(ID))