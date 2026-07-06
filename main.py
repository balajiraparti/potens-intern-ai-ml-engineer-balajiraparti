import streamlit as st
"""
Boilerplate code to accept pdf from user generated from chatgpt
"""
st.set_page_config(page_title="RAG Document Upload", page_icon="📄")

st.title("📄 Upload a PDF")
st.write("Upload a single PDF document to begin processing.")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    st.write("**File Details**")
    st.write(f"- File Name: {uploaded_file.name}")
    st.write(f"- File Size: {uploaded_file.size / 1024:.2f} KB")

    # Read file bytes (for processing with PyMuPDF, pdfplumber, etc.)
    pdf_bytes = uploaded_file.read()

    st.info("PDF is ready for processing.")

    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            st.success("PDF processed successfully!")