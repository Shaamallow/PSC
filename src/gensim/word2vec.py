# Quick implementation and testing of Word2vec Capabilities of gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Using CBOW (continuous bag of words) model

import gensim.downloader as api
wv = api.load('word2vec-google-news-300', return_path=True)
