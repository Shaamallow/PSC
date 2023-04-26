### IMPORT LIBS 

import os
import pdfplumber
import re
import gensim as gm
import numpy as np
import nltk as sl
import pandas as pd

from termcolor import colored

### DEFINE PATH and folder

path = os.getcwd()
corpus_path = '/data/corpus/corpus'
ID = 6
corpus_path = corpus_path+str(ID)


files = os.listdir(path+corpus_path)

### EXTRACT .txt files from folder

def get_txt_from_folder(files):
    # Get all .txt files in list of files
    document_list = []
    for k in range(len(files)):
        if files[k][-4:] == ".txt":
            document_list.append(files[k])
    return document_list

### IMPORT CORPUS

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __iter__(self):
        for doc_path in files:
            print('Process doc : ' + doc_path)
            with open(path+corpus_path+"/"+doc_path, "r") as f:
                text = f.read()
            text = re.sub('[^a-zA-Z]+', ' ', text)
            # remove double space
            text = re.sub(' +', ' ', text)

            # lower case
            text = text.lower()

            # remove stop words
            stop_words = sl.corpus.stopwords.words('english')
            text = [w for w in text.split() if not w in stop_words]

            # stem words
            stemmer = sl.stem.PorterStemmer()
            result = []
            for w in text:
                if w=='cid' or stemmer.stem(w)=='cid':
                    pass
                result.append(stemmer.stem(w))
            yield text

files = get_txt_from_folder(files)

corpus = MyCorpus()

### BUILD DICTIONARY

dictionary = gm.corpora.Dictionary(corpus)

### BUILD BAG OF WORDS

bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]

### BUILD TFIDF MODEL

tfidf = gm.models.TfidfModel(bow_corpus)

### Save the model 

dictionary.save(path+'/src/gensim/models/corpus7'+'/dictionary.model')
#bow_corpus.save(path+'/src/gensim/models/corpus7'+'/bow_corpus.model')
tfidf.save(path+'/src/gensim/models/corpus7'+'/tfidf.model')

### ORDER list of tuples by tfidf value

def order_tuples_by_tfidf(tuples):
    # Order tuples by tfidf value
    return sorted(tuples, key=lambda x: x[1], reverse=True)

### EXTRACT WORDS
for doc in tfidf[bow_corpus]:
    l = order_tuples_by_tfidf(doc)
    for k in range(5):
        print(dictionary[l[k][0]], l[k][1])
    print()

print("Display of the most similar documents : ")

### SIM 

index = gm.similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=len(dictionary))

### Display and save a DataFrame of the 5 most similar documents with the values
### Write all names of the documents using only the first 2 words (separated by a _) 

def display_similarities(index, tfidf, bow_corpus, dictionary, files, n=5):
    # Display similarities
    df = pd.DataFrame(columns=['Name'])
    # Add as many columns as n values 
    for i in range(len(files)):
        #print('Document : ' + files[i])
        sims = index[tfidf[bow_corpus[i]]]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        
        df.loc[i, 'Name'] = " ".join(files[i].split("_")[:2])
        # same thing with "-" instead of "_"
        df.loc[i, 'Name'] = " ".join(df.loc[i, 'Name'].split("-")[:2])
        df.loc[i, 'Name'] = " ".join(df.loc[i, 'Name'].split(" ")[:2])
        for j in range(n):
            doc_name = " ".join(str(files[sims[j+1][0]]).split('_')[:2])
            doc_name = " ".join(doc_name.split('-')[:2])
            doc_name = " ".join(doc_name.split(' ')[:2])
            df.loc[i, 'Doc'+str(j+1) + ' - Values'] = doc_name + ' - ' + str(round(sims[j+1][1]*100)) + '%'
    return df

#df = display_similarities(index, tfidf, bow_corpus, dictionary, files, n=2)

# Save the DataFrame in a csv file
#df.to_csv(path+corpus_path+'/removed'+'/similarities.csv', index=False)