from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from collections import defaultdict
from collections import defaultdict


from langchain_chroma.vectorstores import Chroma
from pydantic import BaseModel,Field
from typing import List
from pathlib import Path
from langchain_openai import ChatOpenAI
import os
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
class ParallelQuerySchema(BaseModel):
    Queries: List[str]= Field(description="Parallel Query")

def get_llm():
     return  [

        ChatOpenAI(model="gpt-4o-mini"),
            ChatMistralAI(
                model="mistral-large-latest",
                temperature=0,
            ),
            ChatMistralAI(
                model="mistral-large-latest",
                temperature=0,api_key=os.getenv("MISTRAL_CUSTOM_EVALUATION")
            ),
            ChatMistralAI(
                model="mistral-large-latest",
                temperature=0,api_key=os.getenv("MISTRAL_API_KEY_BACKUP")
            ),


        ]
def get_embedding_model():
    return OpenAIEmbeddings() 
def get_vector_db():
    embedding=get_embedding_model()
    project_root = Path(__file__).resolve().parent.parent
    chroma_path = project_root / "chroma_db_updated"
    return Chroma(
            persist_directory=chroma_path,
            embedding_function=embedding
        )


def parallel_query(query: str,k:int):
        user_prompt_parallel_query=f"""
       generate alternate or {k} different  variations/views of {query} this query that would help us to retrieve relevant documents.
        Rules:
        - do not generate any other text
        - output should be list
        
        format:
        ['question1','question2','question3']
        """
        for llm in get_llm():
            try:
                model=llm.with_structured_output(ParallelQuerySchema)
                prompt=ChatPromptTemplate.from_messages([
                ("system",user_prompt_parallel_query),
                ("human","{text}")

            ])
                map_chain=prompt|model
                response= map_chain.invoke({"text":"Generate Parallel query"})
                return response.Queries
            except Exception as e:
                print("rate limit reached trying another llm...")
 

def search_chunks(questions: list):
    vector_db=get_vector_db()
    # print(questions)
    Docs=[]
    for query in questions:
       
        search_result=vector_db.similarity_search(query)
        Docs.append([doc for doc in search_result])
  
    return Docs
def reciprocal_rank_fusion(results_list, k: int = 60):
    rrf_scores = defaultdict(float)  
    all_unique_chunks = {}  
    
    chunk_id_map = {}
    chunk_counter = 1
    chunk_counter = 1
    scores = defaultdict(float)
    document_map={}
    for result in results_list:
        
        for rank, doc in enumerate(result):
            chunk_content=doc.page_content
            document_map[chunk_content] = doc
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            chunk_id = chunk_id_map[chunk_content]
            
       
            all_unique_chunks[chunk_content] = doc
            
            position_score = 1 / (k + rank)
            rrf_scores[chunk_content] += position_score
    
            
            scores[chunk_content] += 1 / (k + rank + 1)  
    ranked_chunks=sorted(
    scores.items(),
    key=lambda x: x[1],
    reverse=True
)
    return [
        {
            "document": document_map[chunk],
            "score": score
        }
        for chunk, score in ranked_chunks
    ]

def build_chunks(docs:dict):

     lines = []
     for doc_chunk in docs:
         lines.append(doc_chunk['document'].page_content)
        

     return "\n\n".join(lines)
 

def ranked_context(userquery:str,k:int):
    parallel_queries=parallel_query(userquery,k)
    parallel_chunks=search_chunks(parallel_queries)
    # print(parallel_chunks)
    ranked_docs=reciprocal_rank_fusion(parallel_chunks)
    # print(ranked_docs)
    combined_context=build_chunks(ranked_docs)
    # print(combined_context)
    return combined_context,ranked_docs

def build_ranked_context(userquery:str,k:int=5):
    context,docs=ranked_context(userquery,k)
    system_prompt=f"""You are helpful assistant with who answers user query based on available context retrieved from a pdf file along with page number and page_contents.
    you should only ans the user based on the following context and navigate the user to open the right page number to know more.if you don't find any context then humbly say no the question
    Context:
    {context}
    """
    for llm in get_llm():
        try:
            model=llm
            prompt=ChatPromptTemplate.from_messages([
                ("system",system_prompt),
                ("human","{text}")

            ])
            map_chain=prompt|model | StrOutputParser()
            response = map_chain.invoke({"text":userquery})
            return response,docs
        except Exception as e:
            print("rate limit reached trying another llm...")

if __name__ == "__main__":
    userquery=input("enter user query: ")
    k=int(input("enter number of queries: "))
    print(build_ranked_context(userquery,k))