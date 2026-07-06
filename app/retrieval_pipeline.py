from dotenv import load_dotenv

from openai import OpenAI
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
load_dotenv()
client=OpenAI()

# made a query rewriting function converts abstract questions into more specific searchable question
def query_rewrite(user_query):
    system_prompt=f"""You are a query rewriter. Your task is to transform the user's question into a more effective search query.
        
Original question: {user_query}

Rewrite this question to be more search-friendly by:
1. Using more precise terminology
2. Adding relevant keywords
3. Making it clearer and more specific

Return ONLY the rewritten question, nothing else."""
    response=client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_query}
        ])
    return response.choices[0].message.content

def get_embedding_model():
    return OpenAIEmbeddings() 

def get_vector_store(embedding_model):
    return Chroma(
            persist_directory="chroma_db_semantic",
            embedding_function=embedding_model
        )
def retrive_content(user_query:str):
    rewritten_query=query_rewrite(user_query)
    embedding_model=get_embedding_model()
    vector_store=get_vector_store(embedding_model)
    search_result=vector_store.similarity_search(query=rewritten_query)
    return search_result
