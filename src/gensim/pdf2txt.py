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

from termcolor import colored

# Path variable 

# EXECUTE FROM THE ROOT OF THE PROJECT 
path = os.getcwd()
ID = 2 # change this to the ID of the corpus you want to clean
path = path+"/data/"+'corpus/corpus'+str(ID)

# Get all the name of the files in the corpus with .pdf extension

doc_names = os.listdir(path)
doc_names = [doc for doc in doc_names if doc[-4:] == '.pdf']

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
        print(str(i+1),'/', len(text), word, end=' ')
        if word not in nltk.corpus.words.words('en'):
            removed.append(word)
            print(colored('removed', 'red'))
        else : 
            words.append(word)
            print(colored('kept', 'green'))
        i += 1 

    print("Removed words : ", len(removed))
    return words, removed

def pdf_import(document_path :str):
    """
    Open a pdf with pdf plumber and return the text of the pdf

    # Input :
    - path to the document

    # Ouput : 
    - text : list of words in the pdf
    """

    with pdfplumber.open(document_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    return text

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

    pass

#### TEST ####

# Apply to corpus 2

text = pdf_import(path+'/'+doc_names[2])
print(doc_names[2])
#text = text[:1000]
#print(text)
#print(type(text))
text, removed = clean_text(text)

print(text)

print()

print(removed)