import os
import pandas as pd
import pdfplumber
import re
import nltk
import numpy as np

from math import log


class Corpus(object):


    def __init__(self, path, corpusID) -> None:

        self.path = path+"/"+corpusID
        self.corpusID = corpusID

        # Description of the corpus TODO : Proper Handling (generation + load)
        self.description = False
        documents = os.listdir(self.path)
        if "description.txt" in documents:
            documents.remove("description.txt")
            self.description=True

        self.docs = documents
        self.WF = self.load()

    # Method to get the size of a corpus

    def get_size(self):
        # check if the corpus has a description file
        return(len(self.docs))

    # Method to get the description of a corpus

    def get_description(self):
        # if the corpus has a description file

        if self.description:
            with open(self.path + "/description.txt", "r") as f:
                return f.read()
        else:
            return "No description available"

    # Method to get the text of a pdf

    def get_text_pdf(self, doc_path):
        """
        Input : path to the document
        return : text of the document
        """
        with pdfplumber.open(doc_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    
    # Method to get the words of a text

    # TODO : get all get_words function => Remove all non alphabetical characters

    def get_words(self, text):
        """
        Input : text of a document
        return : list of words of the document
        """

        # remove all non alphabetical characters and space (keep only words)
        text = re.sub(r'[^a-zA-Z]', ' ', text)

        # remove double spaces
        text = re.sub(' +', ' ', text)

        # remove stopwords
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = nltk.word_tokenize(text)
        words = [w for w in words if not w in stop_words]

        # stem words
        stemmer = nltk.stem.SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]

        return words

    # Method to update the WF matrix

    def generate_WF(self):

        # add progress bar for the loop
        i = 0

        for doc in self.docs:
            text = self.get_text_pdf(self.path + "/" + doc)
            words = self.get_words(text)
            j = 0
            for word in words:

                print("Processing document " + str(i) + "/" + str(len(self.docs)) + " | word " + str(j) + "/" + str(len(words)), end="\r")

                if word in self.WF.index:
                    self.WF.loc[word][doc] += 1
                else:
                    self.WF.loc[word] = [0] * self.WF.shape[1]
                    self.WF.loc[word][doc] += 1
                j += 1
            i += 1


    # Method to add document to corpus 

    # TODO FIX ADD DOC

    def add_doc(self, doc_path):
        doc = doc_path.split("/")[-1]
        self.docs.append(doc)
        self.WF[doc] = 0
        text = self.get_text_pdf(doc_path)
        words = self.get_words(text)

        for word in words:

            if word in self.WF["word"].values:
                self.WF.loc[self.WF["word"] == word, doc] += 1
            else:
                self.WF.loc[self.WF.shape[0]] = [word] + [0] * (self.WF.shape[1] - 1)
                self.WF.loc[self.WF["word"] == word, doc] += 1
            

    # Method to remove document from corpus

    def remove_doc(self, doc_name):
        self.docs.remove(doc_name)
        self.WF = self.WF.drop(doc_name, axis=1)
    
    # Method to save the corpus

    def save(self, path=None):
        if path==None:
            path = "./data/results/" + self.corpusID + "/WF.csv"
        self.WF.to_csv(path, index=True)

    # Method to load a corpus

    def load(self, new_corpus = None):
        """
        Input : ID of an existing corpus such as "corpus1"
        Change properties of the corpus object accordingly : 
            - self.path = /data/corpus/corpusID
            - self.docs = os.listdir(self.path)
            - self.WF = pd.read_csv(/data/results/corpusID/WF.csv)
        """
        
        # if no corpus is specified, load the WF matrix
        if new_corpus != None:
            self.path = self.path + "/" + new_corpus
            self.corpusID = new_corpus
            self.docs = os.listdir(self.path)
        
        # check if corpusID folder exists in results
        if self.corpusID not in os.listdir("./data/results"):
            os.mkdir("./data/results/" + self.corpusID)
        # check if WF matrix exists in /results/coprpusID
        if "WF.csv" not in os.listdir("./data/results/" + self.corpusID):
            self.WF = pd.DataFrame(columns=self.docs)
            self.save("./data/results/" + self.corpusID + "/WF.csv")

        # Import WF matrix wiwth word as index
        return pd.read_csv("data/results/" + self.corpusID + "/WF.csv", index_col=0)

        
    # TODO : Method to clean WF matrix (remove words too long or that appear in only one document only one time -most likely a typo/failure at import-)
    # TODO : Method to clean WF matrix by check in a dictionnary if the word is a real word

    # TODO : Method to generate TF-IDF matrix

    # Sim Matrix generator :

    def generate_cosine_matrix(self):

        # check if sim already exists
        if "cosine_matrix" in os.listdir("./data/results/" + self.path.split("/")[-1]):
            return

        M = np.zeros((len(self.docs),len(self.docs)))
        M = pd.DataFrame(M, columns=self.docs, index=self.docs)

        # Generate the cosine matrix
        for i in range(len(self.docs)): 
            docA = self.docs[i]
            
            # Sim matrix is symetric, so we only need to compute the upper triangle
            for j in range(i+1,len(self.docs)): 
                docB = self.docs[j]
                print(docA)
                print(docB)
                print("")

                TF_IDF_A = self.get_TF_IDF(docA)
                TF_IDF_B = self.get_TF_IDF(docB)

                # Dot product of TF-IDF vectors

                vecA = {word: TF_IDF_A[word][0]*TF_IDF_A[word][1] for word in TF_IDF_A}
                vecB = {word: TF_IDF_B[word][0]*TF_IDF_B[word][1] for word in TF_IDF_B}

                # List of Words
                words = list(set(vecA.keys()).union(set(vecB.keys())))

                # Dot product of TF-IDF vectors
                dot_product = sum([vecA.get(word, 0) * vecB.get(word, 0) for word in words])

                # Norm of TF-IDF vectors
                normA = sum([vecA[word]**2 for word in vecA])**0.5
                normB = sum([vecB[word]**2 for word in vecA])**0.5

                # Cosine similarity
                cosine_similarity = dot_product / (normA * normB)

                M.loc[docA, docB] = cosine_similarity
        return M

        

    def get_TF_IDF(self, doc):
        """
        Input : document name
        return : TF-IDF vector of the document
        """

        # TODO : add check (WF matrix exists, no empty docs)

        # Dictionnary, key = word, value = [TF, IDF]

        # N = Number of word in doc
        # l = Number of docs in corpus
        # n = number of doc where the word appears
        # IDF = 1+log(l/n+1)
        # TF = occurence of the word in the doc/total number of words in the doc
        # TF-IDF = TF*IDF

        def IDF(l,n):
            return 1+log(l/n+1)

        TF_IDF_doc = {}

        # number of words in document
        N = self.WF[doc].sum(axis=0)

        for word in self.WF.index:
            # count nb of doc where the word appears
            n = (self.WF.loc[word]>0).sum(axis=0)
            
            # add key to dictionnary
            TF_IDF_doc[word] = [self.WF.loc[word][doc]/N,IDF(len(self.docs), n)]

        return(TF_IDF_doc)
                   
        
# Test manipulation Dataframe

def test0():
    # generate corpus2 WF matrix

    corpus = Corpus("./data/corpus", "corpus2")
    print(corpus.WF.head())
    #corpus.generate_WF()
    #corpus.save()
    print("------")
    print(corpus.generate_cosine_matrix())

def test1():
    WF = pd.DataFrame(columns=["doc1", "doc2"])
    print(WF)
    WF.to_csv("test.csv", index=True)

    WF.loc["word1"] = [1,1]
    WF.loc["word2"] = [1,1]

    print(WF)

    print("-----")

    if 'word3' in WF.index:
        WF.loc['word3']["doc2"] = 2
    else:
        WF.loc['word3'] = [1,1]
    
    print(WF)

    if 'word2' in WF.index:
        WF.loc['word2']["doc2"] = 20

    print(WF)
        
# - Create a corpus object

if __name__ == "__main__":

    test0()
