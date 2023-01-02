# Script to create gradio component named Corpus
# Goto Readme Roadmap for more information

import gradio as gr
import os

# import Corpus Class from Corpus.py
from src.Interface.ClassObjects.CorpusClass import Corpus

class CorpusInterface(object):
    """Corpus Interface 

    PARAMETERS : 
    - path : path to the corpus folder
    - corpus_list : list of all available corpus
    - corpus : current selected corpus object
    - corpus_path : path to the current selected corpus

    METHODS : 
    - Interface : create and return a gradio component for the corpus TAB
    """

    def __init__(self, path, corpus_list):
        self.path = path
        self.corpus_list = corpus_list # all available corpus
        self.corpus = None # current selected corpus object
        self.corpus_path = None # path to the current selected corpus
    
    # use block to create a gradio component

    def Interface(self):
        """
        Return a gradio component for the corpus"""
        with gr.Blocks() as corpus:
            gr.Markdown("## Corpus Loader")

            # 2 rows
            with gr.Row():

                with gr.Column():
                    gr.Markdown("### Choix Corpus")
                    current_corpus = gr.Dropdown(label="Pick corpus", choices=self.corpus_list)
                    load_corpus = gr.Button("Load Corpus")

                    
                with gr.Column():
                    gr.Markdown("### Size of Corpus")
                    corpus_size = gr.Number(label="Size Corpus")
                    head_corpus = gr.Button("Head")
            
            corpus_description = gr.Textbox(label="Description", lines=5)

            # Define button behavior after all the components are created
            load_corpus.click(self.__LoadCorpus, inputs=[current_corpus], outputs=[corpus_size,corpus_description])
            head_corpus.click(self.__HeadCorpus, inputs=[], outputs=[corpus_description])

        return corpus

    # Manualy modify corpus, add an other function to modify the selected corpus from the gradio app
    def __Manual_LoadCorpus(self, corpus):
        # Modify the current corpus object
        self.corpus = corpus

    # Modify the function with the elemlents to display in the gradio app
    def __LoadCorpus(self, corpus_name):
        """
        Call by the gradio button, MODIFY the current corpus object
        Return the corresponding useful elements to display in the gradio app

        PARAMETERS :
        - corpus_name : name of the corpus to load
        
        RETURN : 
        - size of the corpus
        - description of the corpus"""

        self.corpus = Corpus(self.path,corpus_name)

        return [self.corpus.get_size(), self.corpus.get_description()]

    def __HeadCorpus(self):
        """
        RETURN : 
        - Head of the WF matrix"""

        return(self.corpus.WF.head())


if __name__ == "__main__":

    global path_to_corpus
    path_to_corpus = "./data/corpus"

    # READ possible corpus adress
    corpus_list_path = os.listdir(path_to_corpus)

    Test = CorpusInterface(corpus_list_path)
    demo = Test.Interface()
    demo.launch()