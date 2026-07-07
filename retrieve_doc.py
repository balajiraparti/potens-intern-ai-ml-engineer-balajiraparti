from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings
load_dotenv()


def get_embedding_model():
    return OpenAIEmbeddings() 

def get_vector_db(embeddings):
    return Chroma(
    persist_directory="chroma_db_updated",
    embedding_function=embeddings
)

# asked chatgpt to write function to retreive chunks from vectordb based on unquie chunk_id
def get_chunk_by_id(chunk_id: str):
    embedding=get_embedding_model()
    vector_store=get_vector_db(embedding)
    results = vector_store.get(
        where={"chunk_id": chunk_id},
        include=["documents", "metadatas"]
    )

    if not results["documents"]:
        return None

    return {
        "text": results["documents"][0],
        "metadata": results["metadatas"][0]
    }

print(get_chunk_by_id("HR Manual DFY 2025.pdf_chunk_3"))