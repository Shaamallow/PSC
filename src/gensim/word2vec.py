# Quick implementation and testing of Word2vec Capabilities of gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Using CBOW (continuous bag of words) model

import gensim as gm

wv = gm.models.KeyedVectors.load_word2vec_format("./data/pre-trained-models/word2vec-google-news-300/GoogleNews-vectors-negative300.bin", binary=True)


# Retrieve Voc
for index, word in enumerate(wv.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(wv.index_to_key)} is {word}")