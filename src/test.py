from src.Interface.ClassObjects.DocumentClass import Document 

if __name__=="__main__":
    # create corpus object
    corpus0 = Corpus("./data/corpus", "corpus2")
    print(corpus0.WF.head())
    # create document object
    doc1 = Document("docA.pdf", corpus0)
    # print the words of the document
    print(doc1.get_top_words())
