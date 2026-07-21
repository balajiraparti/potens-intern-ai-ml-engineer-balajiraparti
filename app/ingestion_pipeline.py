from pathlib import Path
from dotenv import load_dotenv
from langchain_docling.loader import DoclingLoader
from langchain_community.document_loaders import PyPDFLoader

# from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
import tempfile
import os

import streamlit as st
load_dotenv()

pdf_dir = Path(__file__).parent / "data"

#retieving embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# storing chunks into vector database
def store_data(chunks,embedding_model):
    return Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    persist_directory="chroma_db_hugging_face",
                )
# def assign_chunk_id_and_source(chunks,pdf_file):
#     for i,chunk in enumerate(chunks):
#                     chunk.metadata["chunk_id"] = f"{pdf_file.name}_chunk_{i}"
#                     chunk.metadata["source"] = pdf_file.name
def ingestion(pdf_files):
    # storing file on disk so that other libraries can access it
    docs=[]
    for file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            temp_path = tmp.name
            loader = PyPDFLoader(temp_path)
            docs.extend(loader.load())
    # Split text into smaller chunks like paragraphs/sentences
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=300,
    #     chunk_overlap=50
    # )


    text_splitter = SemanticChunker(embeddings=get_embedding_model())
    chunks = text_splitter.split_documents(docs)

    #created custom function to update the source file name and assign unique chunk id to each chunk before storing it to vector store
    # assign_chunk_id_and_source(chunks,pdf_files)
    # Embeddings
    embedding_model=get_embedding_model()

    vector_store=store_data(chunks,embedding_model)
    retriever=vector_store.as_retriever()
    print("Ingestion is done")
    print(vector_store._collection.count())
    print(retriever.invoke("what is indexing"))


  