# Script to clean the pdf and convert it to txt

# Use df.apply to clean the text 
# Use pdfplumber to extract text from pdf
# Save text in .txt file

# Remove all non alphabetical characters
# Remove words not in nltk english dictionary

# Get all .txt files in list of files

#### SCRIPT ####

import os 
import re
import nltk
import pdfplumber
import pandas as pd
import time
import sys

from termcolor import colored

import gensim.downloader as api

# Path variable 

# EXECUTE FROM THE ROOT OF THE PROJECT 
path = os.getcwd()
ID = 3 # change this to the ID of the corpus you want to clean
path = path+"/data/"+'corpus/corpus'+str(ID)

# Get all the name of the files in the corpus with .pdf extension

doc_names = os.listdir(path)
doc_names = [doc for doc in doc_names if doc[-4:] == '.pdf']

# Glove model

print("Loading Glove model...")
t1 = time.time()
dic = api.load('glove-twitter-25')
t2 = time.time()
print("Done.", len(dic), " words loaded!")
print("Time : ", t2-t1, "s")


dic = dic.index_to_key


# Function to clean the text of a pdf file

def clean_text(text : str):
    """
    Clean the text of a pdf file, function is supposed to be used with df.apply

    # Input :
    - text : string of the text of the pdf file

    # Ouput :
    - text : string of the cleaned text of the pdf file
    - removed : list of the words removed from the text
    """

    removed = []
    
    # Remove all non alphabetical characters

    text = re.sub('[^a-zA-Z]+', ' ', text)
    text = text.lower()
    text = text.split(' ')

    # Remove words not in nltk english dictionary

    words = []
    i = 0
    for word in text:
        print('Processing ', str(i+1),'/', len(text), ' : ', word, end=' ')
        if word not in dic:
            removed.append(word)
            print(colored('removed', 'red'), end='\r')
        else : 
            words.append(word)
            print(colored('kept', 'green'), end='\r')
        i += 1 
        sys.stdout.write("\033[K")

    return words, removed

def pdf_import(document_path :str):
    """print('', end='\r')
    Open a pdf with pdf plumber and return the text of the pdf

    # Input :
    - path to the document

    # Ouput : 
    - text : list of words in the pdf
    - size : number of pages in the pdf
    """

    print("Loading pdf...", end='\r')
    with pdfplumber.open(document_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    sys.stdout.write("\033[K")

    return text, len(pdf.pages)

def clean_folder(path : str):
    """
    Clean all documents in a folder and save them as txt files

    # Input :
    - path : path to the folder containing the pdf files

    # Output :
    - None

    ## Feedback :

    - The function will print the number of words removed from each document
    - The function will print the number of documents cleaned
    - The function will print the status of the cleaning process
    """

    pass

def save_txt(text : str, path : str, name : str = None):
    """
    Save a text as a txt file

    # Input :
    - text : string of the text to save
    - path : path to the folder where the text will be saved
    - name (Optional) : name of the file to save

    # Output :
    - None
    """

    with open(path+'/'+name+'.txt', 'w') as f:
        f.write(text)

#### TEST ####

# Construct some stats for the docs

n = len(doc_names)

stats = pd.DataFrame(columns=['name','nb_pages', 'size', 'size_removed', 'size_removed_percent'])
path_save = path[:-1]+'4'


# Apply to corpus 2

for i in range(n):
    print('Document ', i+1, '/', n, " : ", doc_names[i])
    text,npages = pdf_import(path+'/'+doc_names[i])
    text, removed = clean_text(text)
    stats.loc[i] = [doc_names[i], npages, len(text), len(removed), round(100*len(removed)/(len(text)+len(removed)),2)]
    print("Done : Saving...")
    text = ' '.join(text)
    removed = '\n'.join(removed)
    save_txt(text, path_save, doc_names[i][:-4])
    save_txt(removed, path_save, doc_names[i][:-4]+'_removed')
    sys.stdout.write("\033[F") # Cursor up one line
    sys.stdout.write("\033[K") # Clear to the end of line
    sys.stdout.write("\033[F") # Cursor up one line
    sys.stdout.write("\033[K") # Clear to the end of line

# Save stats in csv file

stats.to_csv(path_save+'/stats.csv')
