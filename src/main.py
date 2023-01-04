# Script to run the main program
# Suppose to handle every pages and the main gradio interface

import gradio as gr
import os
from src.Interface.CorpusInterface import CorpusInterface
from src.Interface.TF_IDFInterface import TFInterface

# VARIABLES FOR THE MAIN PROGRAM
# Update thoses to make your own architecture for project

global PATH_TO_CORPUS # path to the corpus folder
global ALL_CORPUS # list of all available corpus
PATH_TO_CORPUS = "./data/corpus"
ALL_CORPUS = os.listdir(PATH_TO_CORPUS)



# Create a gradio app
with gr.Blocks() as demo:
    gr.Markdown("# Demo")

    with gr.Tab("Corpus"):
        corpusInterface = CorpusInterface(PATH_TO_CORPUS, ALL_CORPUS)
        corpusDisplay = corpusInterface.Interface()
        corpusInterface.LoadCorpus("corpus2")

    with gr.Tab("PDF"):
        gr.Markdown("## PDF")
    
    # TODO : Add loader for the Corpus object (create a method in the CorpusInterface class)
    with gr.Tab("TF-IDF"):
        tfInterface = TFInterface(PATH_TO_CORPUS, corpusInterface.corpus)
        tfInterface.Interface()


if __name__ == "__main__": 
    demo.launch()


#### LAUNCH INSTRUCTION ####

# 1. Open a terminal
# 2. Go to the root of the project
# 3. activate the virtual environment
# 4. run : gradio src/main.py 
# 5. simply save the file to update the gradio app

