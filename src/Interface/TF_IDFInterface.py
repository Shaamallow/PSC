# SCRIPT TO Display the TF-IDF results
# Create a gradio TAB named "TF-IDF" in the main gradio app
# Goto README.md Roadmap for more information


import gradio as gr
import os

from src.Interface.ClassObjects.TFIDFClass import TFIDF
from src.Interface.ClassObjects.DocumentClass import Document

class TFInterface(object):
    """TF-IDF Interface

    PARAMETERS : 
    - path : path to the corpus folder
    - corpus : current selected corpus object

    METHODS : 
    - Interface : create and return a gradio component for the TF-IDF TAB
    """

    def __init__(self, path, corpus):
        self.path = path
        self.corpus = corpus
        self.docA = None
        self.docB = None
        self.TFIDF = None

    # use block to create a gradio component
    def Interface(self):
        """
        Return a gradio component for the corpus"""

        with gr.Blocks() as TFIDF:
            gr.Markdown("## TF-IDF")

            # 3 Row

            with gr.Row():

                with gr.Column():

                    gr.Markdown("### DocA")
                    gr_docA = gr.Dropdown(label="Pick DocA", choices=self.corpus.docs)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("")
                        gr_sim_button = gr.Button("Similarity")
                        gr_similarity = gr.Number(label="Similarity")

                with gr.Column():
                    gr.Markdown("### DocB")
                    gr_docB = gr.Dropdown(label="Pick DocB", choices=self.corpus.docs)

            # Define button behavior after all the components are created
            gr_sim_button.click(self.getSimilarity, inputs=[gr_docA,gr_docB], outputs=[gr_similarity])

        
        return TFIDF

    def getSimilarity(self, docA, docB):
        self.docA = Document(docA, self.corpus)
        self.docB = Document(docB, self.corpus)
        self.TFIDF = TFIDF(self.corpus, self.docA, self.docB)
        return self.TFIDF.get_sim()