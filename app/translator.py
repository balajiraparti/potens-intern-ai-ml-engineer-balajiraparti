
from typing import TypedDict
from pydantic import BaseModel,Field
from openai import OpenAI
from dotenv import load_dotenv
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
from langgraph.graph import START,StateGraph,END
from langchain_mistralai import ChatMistralAI

class TranslationSchema(BaseModel):
     is_translation_needed:bool = Field(description="Bool variable to check translation is needed or not")
class AIResponse(BaseModel):
    translated_message:str = Field(description="Translated Message")
class State(TypedDict):
    user_message: str
    is_translation_needed: bool
    translated_message: str
    
def detect_query(state:State):
    user_message=state.get("user_message")
   
    llm=ChatMistralAI(model="mistral-small-latest",temperature=.3)
    messages =ChatPromptTemplate.from_messages( [
    (
        "system",
        """You are Language Detection Agent.Your job is to check whether input user query is in english language.
        Rules:
        - If user query is in english language then return "False"
        - If user query is any other langauge then return "True"
        Return the output in JSON format like:
{{
  "is_translation_needed": True
}}

Only return valid JSON.
        """,
    ),
    ("human", "query: {user_message}"),
])
    llm_with_structured_output=messages | llm.with_structured_output(TranslationSchema,method="json_mode")
    ai_msg = llm_with_structured_output.invoke({"user_message":user_message})
    state['is_translation_needed']=ai_msg.is_translation_needed
    return state




def route_node(state:State)-> Literal["translate_query",END]:
    is_translation_needed=state.get('is_translation_needed')
    if is_translation_needed:
        return "translate_query"
    else:
        return END
        


def translate_query(state:State):
        user_message=state.get('user_message')
        llm=ChatMistralAI(model="mistral-small-latest",temperature=.3)
        messages = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are expert AI Language Translator Agent.Your job is to transform message from source language message to native english query.
        Return the output in JSON format like:
{{
  "translated_message": str
}}

Only return valid JSON.
        """,
    ),
    ("human", "query: {user_message}"),
])
        llm_with_structured_output=messages | llm.with_structured_output(AIResponse,method="json_mode")
        ai_msg = llm_with_structured_output.invoke({"user_message":user_message})
        state['translated_message']=ai_msg.translated_message
        return state

graph_builder=StateGraph(State)
graph_builder.add_node("detect_query",detect_query)
  
graph_builder.add_node("translate_query",translate_query)
graph_builder.add_node("route_node",route_node)
graph_builder.add_edge(START,"detect_query")
graph_builder.add_conditional_edges("detect_query",route_node)
graph_builder.add_edge("translate_query",END)
graph= graph_builder.compile()

def get_translation_graph():
    return graph_builder.compile()

def call_graph(query):
            graph=get_translation_graph()
            state={
                "user_message":query,
                "is_translation_needed": False,
            "translated_message":""
            }
            result=graph.invoke(state)
            if result['is_translation_needed']:
                return result['translated_message']
            return result['user_message']
