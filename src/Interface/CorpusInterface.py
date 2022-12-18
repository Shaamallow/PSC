# Script to create gradio component named Corpus
# Goto Readme Roadmap for more information

import gradio as gr
import os

# import Corpus Class from Corpus.py
from Interface.ClassObjects.CorpusClass import Corpus

class CorpusInterface(object):

    def __init__(self, path, corpus_list):
        self.path = path
        self.corpus_list = corpus_list # corpus object from Corpus.py
        self.corpus = None # current selected corpus object
        self.corpus_path = None # path to the current selected corpus
    
    # use block to create a gradio component

    def Interface(self):
        # create a gradio block for corpus 
        # return the corpus block 
        with gr.Blocks() as corpus:
            gr.Markdown("## Corpus Loader")

            # 2 rows
            with gr.Row():

                with gr.Column():
                    gr.Markdown("### Choix Corpus")
                    current_corpus = gr.Dropdown(label="Pick corpus", choices=self.corpus_list)
                    load_corpus = gr.Button(label="Load Corpus")

                    load_corpus.click(self.LoadCorpus, inputs=[current_corpus], outputs=[])

                with gr.Column():
                    gr.Markdown("### Size of Corpus")
                    corpus_size = gr.Number(label="Size Corpus")
            
            corpus_description = gr.Textbox(label="Description", lines=5)

        return corpus

    # Manualy modify corpus, add an other function to modify the selected corpus from the gradio app
    def Manual_LoadCorpus(self, corpus):
        # Modify the current corpus object
        self.corpus = corpus

    def LoadCorpus(self, corpus_name):
        # Load the corpus object
        self.corpus = Corpus(self.path+'/'+corpus_name)

        # TODO : add a function to update the size of the corpus


if __name__ == "__main__":

    global path_to_corpus
    path_to_corpus = "./data/corpus"

    # READ possible corpus adress
    corpus_list_path = os.listdir(path_to_corpus)

    Test = CorpusInterface(corpus_list_path)
    demo = Test.Interface()
    demo.launch()