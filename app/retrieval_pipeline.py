from dotenv import load_dotenv

from openai import OpenAI
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
load_dotenv()
client=OpenAI()
def get_embedding_model():
    return OpenAIEmbeddings() 

def get_vector_store(embedding_model):
    return Chroma(
            persist_directory="chroma_db",
            embedding_function=embedding_model
        )
def retrive_content(user_query:str):
    embedding_model=get_embedding_model()
    vector_store=get_vector_store(embedding_model)
    search_result=vector_store.similarity_search(query=user_query)
    return search_result
