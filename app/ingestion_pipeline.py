from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.vectorstores import Chroma
# load_dotenv()
# pdf_path=Path(__file__).parent/ "2507.14306v1.pdf"
# loader=PyPDFLoader(file_path=pdf_path)
# docs=loader.load()
# print(docs[0])
# text_spiltter=rs(chunk_size=1000,chunk_overlap=400)
# chunks=text_spiltter.split_documents(documents=docs)
# print(chunks)
# embedding_model=OpenAIEmbeddings(model="text-embedding-3-large")
# vector_store=QdrantVectorStore.from_documents(
#     documents=chunks,
#     embedding=embedding_model,
#     url="http://localhost:6333",
#     collection_name="research_collection"
    
# )


load_dotenv()

pdf_dir = Path(__file__).parent / "data"

docs = []
for pdf_file in pdf_dir.glob("*.pdf"):
    loader = PyPDFLoader(pdf_file)
    docs.extend(loader.load())

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
chunks = text_splitter.split_documents(docs)

# Embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Store in Qdrant
vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )