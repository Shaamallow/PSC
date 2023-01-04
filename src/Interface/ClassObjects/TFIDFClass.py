import os

from src.Interface.ClassObjects.CorpusClass import Corpus
from src.Interface.ClassObjects.DocumentClass import Document
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
        SIM = self.corpus.get_sim(self.docA.name, self.docB.name)
        SIM = int(SIM*100)
        return SIM

    

if __name__ == "__main__":
    # create corpus object
    corpus0 = Corpus("./data/corpus", "corpus2")
    print(corpus0.WF.head())
    # create document object
    doc1 = Document("docA.pdf", corpus0)
    doc2 = Document("docB.pdf", corpus0)
    # print the words of the document
    print(doc1.get_top_words())
    # create TFIDF object
    tfidf = TFIDF(corpus0, doc1, doc2)
    # print the words of the document
    print(tfidf.get_words())
    # print the similarity between the two documents
    print(tfidf.get_sim())