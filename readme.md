# POTENS-RAG

A production-inspired Retrieval-Augmented Generation (RAG) system that answers questions strictly from uploaded PDF documents while providing transparent citations, multilingual support, and document contradiction detection.

---

# Features

- Upload and index PDF documents
- Semantic chunking
- Vector search using ChromaDB
- LLM-powered question answering
- Source-aware citations
- Document contradiction detection
- Multilingual query support
- Hallucination prevention
- Streamlit UI
- Custom Evaluation using LLM-as-Judge
- RAGAS based evaluation
---



# Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Backend | Python |
| LLM | Mistral,OpenAI|
| Embeddings | OpenAI  |
| Vector Database | ChromaDB |
| Chunking | SemanticChunker,RecursiveCharacterSplitter|
| Framework | LangChain |
| Workflow | LangGraph |





# Ingestion Pipeline

The ingestion pipeline converts uploaded PDF documents into searchable vector embeddings.

### Steps

1. Upload PDF

2. Load document using PyPDFLoader

3. Perform Semantic Chunking,RecursiveCharacterSplitter

4. Generate embeddings

5. Store embeddings inside ChromaDB

---
# Design Decisions

### Chunking Strategy

The system uses **SemanticChunker** instead of fixed-size chunking.

### Why Semantic Chunking?

Other chunking stratigies deals with content and structure of documents and necessitate maintaining constant value of chunk size. This chunking method aims to extract semantic meaning from embeddings and then assess the semantic relationship between these chunks. The core idea is to keep together chunks that are semantic similar.
Benefits:

- Better context preservation
- Higher retrieval accuracy
- Reduced context fragmentation
- Better LLM answers

Each chunk stores metadata including:

```

{
"id":""
"metadata":{
"producer":""
"creator":""
"creationdate":""
"moddate":""
"chunk_id":""
"total_pages":
"page_label":""
"source":""
"page":
}
"page_content":""
"type":""
}
```
### Langchain_community deprecation issue
downgraded version of langchain_community(0.4.0) as it was deprecated in May

### Query Rewriting
If the user query is vague then it converted into more specific searchable query(less abstract)

### Vector Database
The system ChromaDB to store vector embeddings of chunks.Tried both RecursiveCharacterSplitter and SemanticChunker for document chunking.Selected Semantic Chunker because it preserves contextual meaning, creates more coherent chunks, and improves retrieval accuracy for RAG applications.



# Embedding Strategy

Each chunk is converted into an embedding vector.

```
Chunk--->Embedding Model--->Dense Vector--->Stored inside ChromaDB
```

---

# Retrieval Pipeline

When a user asks a question:

1. Query is rewritten

2. Query embedding is generated

3. Similar chunks are retrieved

4. Context is built

5. Context is sent to the LLM

6. Final answer is generated

---

# Hallucination Prevention

The model is instructed to answer **only** from the retrieved document context.

If the retrieved context does not contain sufficient information, the system responds:

> I could not find the answer in the provided document.

The model never invents facts outside the uploaded documents.

---

# Citation Support

Every answer includes citations.

Each citation contains:

- Source file
- Page number
- Chunk ID
- Supporting snippet






# Contradiction Detection

The application can compare two retrieved document sections.

Workflow

```
User-->Select Chunk A-->Select Chunk B-->LLM Comparison-->Conflict / No Conflict with reasoning
```

The output includes:

- Conflict status
- Reasoning
- Supporting evidence



---

# Multilingual Support

The application supports multilingual queries.

Workflow

```
User Query-->Detect Language-->Translate to English-->Retrieve Chunks-->Generate Answer
```

Example

User

```
कर्मचारी को कितनी छुट्टियां मिलती हैं?
```



is Translated to

```
How many annual leaves does an employee receive?
```


The retrieval pipeline always works on English while the user receives the response in the English.

---

# API Endpoints

## Ask

Returns answers with citations.

```
POST /ask
```

Input

```json
{
    "question":"How many annual leaves are provided?"
}

```


## Contradict

Compares two retrieved chunks.

```
POST /contradict
```
---

# Running the Project

Clone repository

```
git clone https://github.com/balajiraparti/potens-intern-ai-ml-engineer-balajiraparti.git
```

Install dependencies

```
pip install -r requirements.txt
```
Start Virtual Environment
```
venv/scripts/activate
```
Configure environment

```
OPENAI_API_KEY=xxxx

MISTRAL_API_KEY=xxxx
```

Run

```
streamlit run main.py
```

---

# Evaluation

A golden dataset is included for evaluation(Generated from Claude).

```
examples/

golden_dataset.csv
```

The dataset contains:

- User Question

- Expected Answer

The system can be evaluated using:

- LLM-as-a-Judge(Custom Evaluation)

- RAGAS Based Evaluation

---

# Future Improvements

- Hybrid Search (BM25 + Dense Retrieval)

- RRF(reciprocal rank fusion) based reranker for chunks

- Cloude based Qdrant support


---

# Project Highlights

- End-to-end RAG pipeline

- Semantic chunking

- Vector similarity search

- Transparent citations

- Hallucination prevention

- Multilingual support

- Document contradiction detection

- Streamlit user interface

- Production-inspired architecture

# Evaluation 
```

Context Precision: 0.800

Answer Relevance: 0.770

Faithfulness: 0.800

with semantic chunking
```

# AI USE LOG

Honestly, Used chatgpt to generate boilerplate code for streamlit,solving issue of loading PyPDFLoader which was deprecated in May 2027 and controller function to get chunk data from vector database using chunk id  
