# Module to run the TF-IDF algorithm
# Author : Eyal Benaroche

# take as input the path to the selected corpus

# Import libraries
import nltk
import os
import pandas as pd
import pdfplumber
import re
import progress.bar as pb

### --- FUNCTIONS --- ###

# - Get Words from a document

def get_text_pdf(doc_path):
    """
    path = path of a pdf

    return : string of all words in a document
    """

    document = ""
    with pdfplumber.open(doc_path) as pdf:
        for j in range(len(pdf.pages)):
            f_page = pdf.pages[j]
            texte = ""
            for i in range(len(f_page.chars)):
                texte= texte + f_page.chars[i]['text']

            # print text from evey page
            document = document + texte

    return document

# - Tokenize, Stem and remove stopwords

def get_words(text):
    """"
    path = string of words in a document
    return : list of stemmed words (stopwords get removed) from the pdf
    """

    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # remove numbers
    text = re.sub(r'[0-9]', '', text)

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

# - Update WF matrix with the words of a document

def add_words_to_WF(words, document, WF):
    """
    Input :
        document = documents in the corpus
        words = list of words in the document
        WF = WF dataframe

    return : WF dataframe with the words of the document added
    """
    
    # Add progress bar for words in each docs
    if progress:
        words_bar = pb.IncrementalBar('Processing Words', max=len(words))

    for word in words:

        if progress:
            words_bar.next()

        if word in WF["word"].values:
            WF.loc[WF["word"] == word, document] += 1
        else:
            WF.loc[WF.shape[0]] = [word] + [0] * (WF.shape[1] - 1)
            WF.loc[WF["word"] == word, document] += 1
    words_bar.finish()
    return WF

def doc_to_WF(doc_path, WF):
    """
    Input :
        doc_path = path to the document
        WF = WF dataframe

    return : WF dataframe with the words of the document added
    """

    # get document name 
    doc_name = doc_path.split("/")[-1]

    # get the text of the document
    text = get_text_pdf(doc_path)

    # get the words of the document
    words = get_words(text)

    # add the words to the WF matrix
    WF = add_words_to_WF(words, doc_path, WF)

    return WF

if __name__ == "__main__":
    # Run a test on the corpus 1
    path = "./data/corpus/corpus1"
    docs = os.listdir(path)
    results_path = "./results/corpus1"

    # Global variable to decide if progress bar should appear
    global progress 
    progress = True

    # Global bar for docs progress in corpus
    global bar
    bar = pb.IncrementalBar('Processing Docs', max=len(docs))
    
    # Create the WF matrix 
    WF = pd.DataFrame(columns=["word"] + docs)
    for doc in docs:
        WF = doc_to_WF(path + "/" + doc, WF)
        bar.next()
    bar.finish()
    
    # Save the WF matrix
    WF.to_csv(results_path + "/WF.csv", index=False)
