import os
import pandas as pd
import pdfplumber
import re
import nltk


class Corpus(object):


    def __init__(self, path) -> None:

        self.path = path
        self.docs = os.listdir(path)
        self.WF = pd.DataFrame(columns=["word"] + self.docs)

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

                if word in self.WF["word"].values:
                    self.WF.loc[self.WF["word"] == word, doc] += 1
                else:
                    self.WF.loc[self.WF.shape[0]] = [word] + [0] * (self.WF.shape[1] - 1)
                    self.WF.loc[self.WF["word"] == word, doc] += 1
                j += 1
            i += 1


    # Method to add document to corpus 

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

    def save(self, path):
        self.WF.to_csv(path, index=False)

    # Method to load a corpus

    def load(self, corpusID):
        """
        Input : ID of an existing corpus such as "corpus1"
        Change properties of the corpus object accordingly : 
            - self.path = /data/corpus/corpusID
            - self.docs = os.listdir(self.path)
            - self.WF = pd.read_csv(/data/results/corpusID/WF.csv)
        """
        self.path = "./data/corpus/" + corpusID
        self.docs = os.listdir(self.path)
        # import without index
        self.WF = pd.read_csv("./data/results/" + corpusID + "/WF.csv", index_col=0 )

    # Method to build a document object from a corpus


        
    # TODO : Method to clean WF matrix (remove words too long or that appear in only one document only one time -most likely a typo/failure at import-)
    # TODO : Method to clean WF matrix by check in a dictionnary if the word is a real word

    
# - Create a corpus object

if __name__ == "__main__":

    # generate corpus1 WF matrix

    corpus = Corpus("./data/corpus/corpus2")
    corpus.load("corpus1")
    print(corpus.WF.head())
