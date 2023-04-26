# Turn the stats.csv file of each corpus into a table using latex and save the result in a .tex file

import pandas as pd
import os
import sys
from tabulate import tabulate

### PATH :

path = os.getcwd()


### Start the script

if __name__ == "__main__":

    if len(sys.argv) > 1:
        ID = int(sys.argv[1])
    else:
        print("Please enter the ID of the corpus you want to clean as an argument for the script")

    #ID = 7 # change this to the ID of the corpus you want the results

    path = path+"/data/"+'corpus/corpus'+str(ID)+'/removed'

    # Check if the ID is correct
    if not os.path.exists(path):
        print("The ID is not correct, please check the ID of the corpus you want to clean")
        sys.exit()

    # Read the stats.csv file
    df = pd.read_csv(path+'/stats.csv', index_col=0)

    # Order by size_removed_percentage
    df = df.sort_values(by=['size_removed_percent'], ascending=False)

    # Shorten all names of the columns
    df = df.rename(columns={'name': 'Name', 'nb_pages': 'Pages','size':'Size', 'size_removed':'Removed', 'size_removed_percent':'% Removed'})

    # Shorten all values in the column name to the first 10 letters
    df['Name'] = df['Name'].str[:10]
    
    # Convert the dataframe into a table

    latex_table = tabulate(df, headers='keys', tablefmt='latex', showindex=False)

    # Save the table in a .tex file

    with open(path+'/stats.tex', 'w') as f:
        f.write(latex_table)

    print("The table has been saved in a .tex file in the folder : "+path)

    