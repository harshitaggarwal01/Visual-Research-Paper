from summarizer import Summarizer

def bert_summarize(document):
    model=Summarizer()

    doc=document.read()

    summary=model(doc,ratio=0.5)

    return summary

# doc1=open("C:/Users/Harshit/Desktop/gensimsample.txt","r")
# bert_summarize(doc1)