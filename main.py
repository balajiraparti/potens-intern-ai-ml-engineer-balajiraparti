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
                with st.spinner("Processing PDF..."):
                    # ingestion(pdf_file)
                    chunks=retrive_content(query)
                    for chunk in chunks:
                        chunk.metadata["source"] = pdf_file.name #updating file name with actual pdf name
                    response=build_context_for_llm(chunks,query)
                    st.write(response)
                    with st.expander("See citations"):
                         st.write(f"ID: {chunk.id}\n source{chunk.metadata['source']} \n ")



def accept_pdf():
    st.title("📄 Upload a PDF")
    st.write("Upload a single PDF document to begin processing.")

    uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    accept_multiple_files=False
        )
 
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

            