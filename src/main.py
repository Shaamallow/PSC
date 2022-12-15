# Script to run the main program
# Suppose to handle every pages and the main gradio interface

import gradio as gr
import os
from Interface.Corpus.Corpus import CorpusInterface
from TF.corpus import Corpus

global path_to_corpus
path_to_corpus = "./data/corpus"

# list of Corpus and their path
corpus_list_path = os.listdir(path_to_corpus)


# Create a gradio app
with gr.Blocks() as demo:
    gr.Markdown("Demo")

    with gr.Tab("TEST"):
        gr.Markdown("OK")

    with gr.Tab("Corpus"):
        corpusInterface = CorpusInterface(corpus_list_path)
        corpusInterface.Interface()
    
demo.launch()