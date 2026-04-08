import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from state import GraphState

def retrieve(state: GraphState):
    print("---NODE: RETRIEVING CONTEXT---")
    question = state["question"]
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    
    context = [doc.page_content for doc in docs]
    
    return {"context": context}