# Class object to represent a document
# document refer to a document in the corpus
# document is define 

# import standard libraries
import re
import os
import nltk
import pandas as pd

# import corpus file 
from src.Interface.ClassObjects.CorpusClass import Corpus 

# Comment/Uncomment to test the file alone
#from CorpusClass import Corpus

class Document(object):

    def __init__(self, name, corpus: Corpus) -> None:

        self.corpus = corpus
        self.name = name
        self.path = corpus.path
        self.docs = corpus.docs
        self.WF = corpus.WF
        self.words = self.get_words()

    # Method to get the words of the document and order them by frequency

    def get_words(self):
        """
        return : dataframe of words of the document ordered by frequency
        """
        is_in_doc = self.WF[self.name]>0
        words = self.WF[self.name][is_in_doc]
        
        return words
    
    def get_top_words(self, n=10):
        """
        return : dataframe of words of the document ordered by frequency
        """
        return self.words.sort_values(ascending=False).head(n)

    # TODO : add method to order words by TF-IDF
    
if __name__=="__main__":
    # create corpus object
    corpus0 = Corpus("./data/corpus", "corpus2")
    print(corpus0.WF.head())
    # create document object
    doc1 = Document("docA.pdf", corpus0)
    # print the words of the document
    print(doc1.get_top_words())
