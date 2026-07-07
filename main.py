import streamlit as st
from app.ingestion_pipeline import ingestion
from app.retrieval_pipeline import retrive_content
from app.build_context import build_context_for_llm
"""
Boilerplate code to accept pdf from user generated from chatgpt
"""
st.set_page_config(page_title="RAG Document Upload", page_icon="📄")


def ask_question(pdf_file):
        query=st.text_input("Enter your query",placeholder="User Query")
        if st.button("Ask 🤔:") and query:
                with st.status("Processing PDF..."):
                    st.write("Ingesting...")
                    # ingestion(pdf_file)
                    st.write("retrieved chunks...")
                    chunks=retrive_content(query)
                    if chunks:
                         st.session_state.chunks
                    for i,chunk in enumerate(chunks):
                        chunk.metadata["chunk_id"] = f"{pdf_file.name}_chunk_{i}"
                        chunk.metadata["source"] = pdf_file.name
                    st.write("Generating response....") #updating file name with actual pdf name
                    response=build_context_for_llm(chunks,query)
                    st.write(response)
                    if chunks:
                        with st.expander("See citations"):
                            #st.write(f"source{chunks.metadata['source']} \n page_label: {chunks.metadata['page_label']}\n page_content: {chunks.page_content} \n type:{chunks.type}")
                            st.write(chunks)


def accept_pdf():
    st.title("📄 Upload a PDF")
    st.write("Upload a single PDF document to begin processing.")
    if "chunks" not in st.session_state:
         st.session_state.chunks=None
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

        return 
accept_pdf()

            