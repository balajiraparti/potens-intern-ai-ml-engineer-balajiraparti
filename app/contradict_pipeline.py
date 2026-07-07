from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel,Field
from langchain_openai import OpenAIEmbeddings
load_dotenv()

def get_llm():
     return ChatOpenAI(model="gpt-4o-mini")
def get_embedding_model():
    return OpenAIEmbeddings() 

def get_vector_db(embeddings):
    return Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)
#Used to enforce structured output from llm
class ContradictSchema(BaseModel):
    is_contradict: str = Field(description="Extracted factual claim")
    reason: str = Field(description="")

def contradict_two_chunks(chunk_id_1:str,chunk_id_2:str):
     model=get_llm().with_structured_output(ContradictSchema)
     prompt=ChatPromptTemplate.from_messages([
        ("system","""you are expert document comparison analyzer for RAG pipeline. Compare two documents to check whether they have contradictory information about the requested concept.
         Rules:
         - Use only information provided in context
         - Do not make any assumptions
         - If the documents discuss different concpets then respond with a "No conflict"
         - If both docs contradict then explain reasoning step by step
         - Contradict exist when facts or action that is opposite or different from each other
         - If you are uncertain, state that the documents do not provide enough information.
         
         strictly follow the ContradictSchema output format
         """),
        ("human","{text}")

    ])
     map_chain=prompt|model
     response= map_chain.invoke({"text":f"chunk_id_1:{chunk_id_1}\n chunk_id_2:{chunk_id_2}"})
     return response
     