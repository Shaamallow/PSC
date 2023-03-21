# Script to turn pdf in list of words 
# Save them in .txt format

import os
import pdfplumber
import re
import nltk

# Get all pdf in folder
global path
path = os.getcwd()
global corpus_path
corpus_path = '/data/corpus/corpus3'

# get list of file in folder :
files = os.listdir(path+corpus_path)


def pdf2txt(files):
    # Use PDF Plumber to extract text from pdf
    i = 1
    for doc_path in files:
        print('Doc : ', i, '/', len(files))
        with pdfplumber.open(path+corpus_path+"/"+doc_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        # Save text in .txt file
        with open(path+corpus_path+"/"+doc_path[:-4]+".txt", "w") as f:
            f.write(text)
        i += 1

def cleantxt(files):
    # Remove all non alphabetical characters
    i = 1
    # List of removed words
    l = []
    for doc_path in files:
        print('Doc : ', i, '/', len(files))
        with open(path+corpus_path+"/"+doc_path, "r") as f:
            text = f.read()
        text = re.sub('[^a-zA-Z]+', ' ', text)

        # Remove words not in nltk english dictionary
        words = []
        for word in text:
            if word not in nltk.corpus.words.words('en'):
                l.append(word)
                print(len(l))
            else : 
                words.append(word)
        

        #print(text)
        with open(path+corpus_path+"/"+doc_path, "w") as f:
            f.write(words)
        i += 1

    return l

def get_txt_from_folder(files):
    # Get all .txt files in list of files
    document_list = []
    for k in range(len(files)):
        if files[k][-4:] == ".txt":
            document_list.append(files[k])
    return document_list

# Uncomment to clean txt files

#txt_files_corpus = get_txt_from_folder(files)
#cleantxt(txt_files_corpus)

# Uncomment to convert pdf to txt
l = pdf2txt(files)

# Add a stats.txt file with stats of the corpus and list of removed words

with open(path+corpus_path+"/stats.txt", "w") as f:
    f.write("Number of documents : "+str(len(files)))
    f.write("Number of removed words : "+str(len(l)))
    f.write("List of removed words : "+str(l))

