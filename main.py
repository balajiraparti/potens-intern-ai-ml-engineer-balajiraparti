import streamlit as st
import os
from app.ingestion_pipeline import ingestion
# from app.retrieval_pipeline import retrive_content
from app.rrf_reranker import build_ranked_context
# from app.build_context import build_context_for_llm
from controller.retrieve_doc import get_chunk_by_id
from app.contradict_pipeline import contradict_two_chunks
from app.translator import call_source_to_eng_graph,call_eng_to_source_graph
from app.custom_evaluation import get_score
# Boilerplate code to accept pdf from user generated from chatgpt

st.set_page_config(page_title="RAG Document Upload", page_icon="📄")


def sidebar_api_keys():
    st.sidebar.header("API Keys")
    st.sidebar.caption("Keys are stored in session state for the current browser session.")

    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "chatgroq_api_key" not in st.session_state:
        st.session_state.chatgroq_api_key = ""
    if "mistral_api_key" not in st.session_state:
        st.session_state.mistral_api_key = ""
    if "mistral_api_key_backup" not in st.session_state:
        st.session_state.mistral_api_key_backup = ""
    if "mistral_api_key_evaluation" not in st.session_state:
        st.session_state.mistral_api_key_evaluation = ""

    st.session_state.openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=st.session_state.openai_api_key,
        type="password",
        placeholder="sk-...",
    )
    st.session_state.chatgroq_api_key = st.sidebar.text_input(
        "ChatGroq API Key",
        value=st.session_state.chatgroq_api_key,
        type="password",
        placeholder="gsk_...",
    )
    st.session_state.mistral_api_key = st.sidebar.text_input(
        "Mistral API Key",
        value=st.session_state.mistral_api_key,
        type="password",
        placeholder="mis-...",
    )
    st.session_state.mistral_api_key_backup = st.sidebar.text_input(
        "Mistral API Key backup",
        value=st.session_state.mistral_api_key_backup,
        type="password",
        placeholder="mis-...",
    )
    st.session_state.mistral_api_key_evaluation = st.sidebar.text_input(
        "Mistral API Key evaluation",
        value=st.session_state.mistral_api_key_evaluation,
        type="password",
        placeholder="mis-...",
    )
    if not st.session_state.chatgroq_api_key or not st.session_state.mistral_api_key or not st.session_state.mistral_api_key_backup or not st.session_state.mistral_api_key_evaluation:
        st.warning("Please enter api key details")
        return
    if st.session_state.openai_api_key:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    if st.session_state.chatgroq_api_key:
        os.environ["GROQ_API_KEY"] = st.session_state.chatgroq_api_key
    if st.session_state.mistral_api_key:
        os.environ["MISTRAL_API_KEY"] = st.session_state.mistral_api_key


sidebar_api_keys()


def ask_question(pdf_file):
         if not st.session_state.chatgroq_api_key or not st.session_state.mistral_api_key or not st.session_state.mistral_api_key_backup or not st.session_state.mistral_api_key_evaluation:
            st.warning("Please enter api key details")
            return
         else:
            st.subheader("Ask Document")
            query=st.text_input("Enter your query",placeholder="User Query")
            if st.button("Ask 🤔:") and query:
                    with st.status("Processing PDF..."):
                        st.session_state.query=query
                        st.write("Ingesting...")
                        if st.session_state.is_ingestion==False:
                            ingestion(pdf_file)
                            st.session_state.is_ingestion=True
                        result=call_source_to_eng_graph(query)
                        query=result['generated_query']
                        st.write("retrieved chunks...")
                        # chunks=retrive_content(query)
                        st.write("Generating response....") #updating file name with actual pdf name
                        # response=build_context_for_llm(chunks,query)
                        response,chunks=build_ranked_context(query)
                        if chunks:
                            st.session_state.chunks=chunks
                        result['generated_query']=response
                        response=call_eng_to_source_graph(result)
                        st.write(response)
                        st.session_state.response=response
                        if chunks:
                            with st.expander("See citations"):
                                #st.write(f"source{chunks.metadata['source']} \n page_label: {chunks.metadata['page_label']}\n page_content: {chunks.page_content} \n type:{chunks.type}")
                                st.write(chunks)
  


# comparing two chunks based unique chunk id
def contradict():
     st.subheader("Compare two chunks")
     doc_id_1=st.text_input("Enter chunk id 1:",placeholder="HR Manual DFY 2025.pdf_chunk_2")
     doc_id_2=st.text_input("Enter chunk id 2:",placeholder="HR Manual DFY 2025.pdf_chunk_2")
     if st.button("Compare"):
        if doc_id_1 and doc_id_2:
            chunk_1=get_chunk_by_id(doc_id_1)
            chunk_2=get_chunk_by_id(doc_id_2)
            if chunk_1 and chunk_2:
                response=contradict_two_chunks(chunk_1['text'],chunk_2['text'])
                st.write(f"is_contradict:{response.is_contradict}\n Reason:{response.reason}")
                with st.expander("See citations"):
                    st.write(f"Chunk 1:\n page_content: {chunk_1['text']} \n\n Metadata:{chunk_1['metadata']}")
                    st.write(f"Chunk 2:\n page_content: {chunk_2['text']} \n\n Metadata:{chunk_2['metadata']}")
            else:
                st.write("Did not find valid chunks from vector db please enter correct chunk ids")
        else:
            st.write("Please input doc id first!")
          
          


     



def accept_pdf():
    st.title("📄 Upload a PDF")
    st.write("Upload a single PDF document to begin processing.")
    if "chunks" not in st.session_state:
         st.session_state.chunks=None
    if "query" not in st.session_state:
         st.session_state.query=None
    if "response" not in st.session_state:
         st.session_state.response=None
    uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    accept_multiple_files=False
        )
    if "is_ingestion" not in st.session_state:
         st.session_state.is_ingestion=False
    if "is_file_uploaded" not in st.session_state:
        st.session_state.is_file_uploaded=False
   
    if not st.session_state.is_file_uploaded and uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

        st.write("**File Details**")
        st.write(f"- File Name: {uploaded_file.name}")
        st.write(f"- File Size: {uploaded_file.size / 1024:.2f} KB")

        # Read file bytes (for processing with PyMuPDF, pdfplumber, etc.)
        pdf_bytes = uploaded_file.read()

        st.info("PDF is ready for processing.")
        st.session_state.is_file_uploaded=True
        st.rerun()
    if st.session_state.is_file_uploaded:
        ask_question(uploaded_file)
        if st.session_state.is_ingestion:
            contradict()
        if st.session_state.query and st.session_state.response and st.session_state.chunks:
             get_score(st.session_state.query,st.session_state.response,st.session_state.chunks)
        return 
accept_pdf()

            