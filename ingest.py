import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()

def ingest_golden_data():

    file_path = "data/fundamental_rights.pdf" 
    
    print(f"Loading {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print("Chunking the document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split the PDF into {len(chunks)} chunks.")

    print("Generating OpenAI Embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Saving to local Chroma database...")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db" 
    )

    print("RAG database is locked and loaded in ./chroma_db")

if __name__ == "__main__":
    ingest_golden_data()