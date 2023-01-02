import os
import pandas as pd
import pdfplumber
import re
import nltk

class TFIDF(object):
    """TF-IDF Class
    
    PARAMETERS :
    - corpus : corpus object
    - docA : document object
    - docB : document object
    
    METHODS :
    - get_words : list of top of words of docA & docB
    - get_sim : similarity between docA & docB
    """

    def __init__(self, corpus, docA, docB) -> None:
        self.corpus = corpus
        self.docA = docA
        self.docB = docB

    # Method to get the words of the document and order them by frequency

    def get_words(self, n=10):
        """
        return : list of words of the document ordered by frequency
        """

        wordsA = self.docA.get_top_words(n)
        wordsB = self.docB.get_top_words(n)

        return wordsA, wordsB

    # Method to get the similarity between the two documents

    def get_sim(self):
        """
        return : similarity between the two documents
        """

    

