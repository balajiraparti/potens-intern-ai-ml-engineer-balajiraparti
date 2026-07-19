
from typing import TypedDict
from pydantic import BaseModel,Field
from openai import OpenAI
from dotenv import load_dotenv
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
from langgraph.graph import START,StateGraph,END
from langchain_mistralai import ChatMistralAI
import streamlit as st
#enforcing structured format
class TranslationSchema(BaseModel):
     is_translation_needed:bool = Field(description="Bool variable to check translation is needed or not")
     source_language:str =Field(description="language to be Translated")

class AIResponse(BaseModel):
    translated_message:str = Field(description="Translated Message")
    
# state information maintained byeach node
class State(TypedDict):
    user_message: str
    is_translation_needed: bool
    translated_message: str
    source_language:str
    generated_query:str

  
class SourceTOEng:    
    def detect_query(self,state:State):
        user_message=state.get("user_message")
    
        llm=ChatMistralAI(model="mistral-small-latest",temperature=.3,api_key=st.session_state.mistral_api_key)
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
    "source_language":str
    
    }}

    Only return valid JSON.
            """,
        ),
        ("human", "query: {user_message}"),
    ])
        llm_with_structured_output=messages | llm.with_structured_output(TranslationSchema,method="json_mode")
        ai_msg = llm_with_structured_output.invoke({"user_message":user_message})
        state['is_translation_needed']=ai_msg.is_translation_needed
        state['source_language']=ai_msg.source_language
        return state





    def route_node(self,state:State)-> Literal["translate_query",END]:
        is_translation_needed=state.get('is_translation_needed')
        if is_translation_needed:
            return "translate_query"
        else:
            return END
            


    def translate_query(self,state:State):
            user_message=state.get('user_message')
            llm=ChatMistralAI(model="mistral-small-latest",temperature=.3,api_key=st.session_state.mistral_api_key_evaluation)
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
            print(ai_msg.translated_message)
            state['translated_message']=ai_msg.translated_message
            return state



    

    def get_translation_graph(self):
        graph_builder=StateGraph(State)
        graph_builder.add_node("detect_query",self.detect_query)
        graph_builder.add_node("translate_query",self.translate_query)
        graph_builder.add_node("route_node",self.route_node)
        graph_builder.add_edge(START,"detect_query")
        graph_builder.add_conditional_edges("detect_query",self.route_node)
        graph_builder.add_edge("translate_query",END)
        return graph_builder.compile()
    
class EngToSource:    
    def detect_query(self,state:State):
        user_message=state.get("generated_query")
    
        llm=ChatMistralAI(model="mistral-small-latest",temperature=.3,api_key=st.session_state.mistral_api_key)
        messages =ChatPromptTemplate.from_messages( [
        (
            "system",
            """You are Language Detection Agent.Your job is to check whether input user query is in english language.

            Rules:
            - If user query is in english language then return "True"
            - If user query is any other langauge then return "False"
            Return the output in JSON format like:
    {{
    "is_translation_needed": True
    "source_language":str
    
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





    def route_node(self,state:State)-> Literal["translate_query",END]:
        is_translation_needed=state.get('is_translation_needed')
        source_language=state.get('source_language')
        if is_translation_needed==True and source_language!="English":
            return "translate_query"
        else:
            return END
            


    def translate_query(self,state:State):
            user_message=state.get('generated_query')
            llm=ChatMistralAI(model="mistral-small-latest",temperature=.3,api_key=st.session_state.mistral_api_key_backup)
            source_language=state.get('source_language')
            messages = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are expert AI Language Translator Agent.Your job is to transform message from English language message to source language.
            Return the output in JSON format like:
            source language:{source_language}
    {{
    "translated_message": str
    }}

    Only return valid JSON.
            """,
        ),
        ("human", "query: {user_message}"),
    ])
            llm_with_structured_output=messages | llm.with_structured_output(AIResponse,method="json_mode")
            ai_msg = llm_with_structured_output.invoke({"user_message":user_message,"source_language":source_language})
            state['translated_message']=ai_msg.translated_message
            return state



    

    def get_translation_graph(self):
        graph_builder=StateGraph(State)
        graph_builder.add_node("detect_query",self.detect_query)
        graph_builder.add_node("translate_query",self.translate_query)
        graph_builder.add_node("route_node",self.route_node)
        graph_builder.add_edge(START,"detect_query")
        graph_builder.add_conditional_edges("detect_query",self.route_node)
        graph_builder.add_edge("translate_query",END)
        return graph_builder.compile()


     
def call_source_to_eng_graph(query):
            source_to_eng=SourceTOEng()
            graph=source_to_eng.get_translation_graph()
            state={
                "user_message":query,
                "is_translation_needed": False,
                "translated_message":"",
                "source_language":"",
                "generated_query":""
            }
            result=graph.invoke(state)
            if result['is_translation_needed']:
                result['generated_query']=result['translated_message']
            result['generated_query']=result['user_message']
            return result
def call_eng_to_source_graph(state):
            source_to_eng=EngToSource()
            graph=source_to_eng.get_translation_graph()
            state['is_translation_needed']=False
            result=graph.invoke(state)
          
            if result['is_translation_needed']==True and result['source_language']!="English":
                return result['translated_message']
            return result['generated_query']


     
