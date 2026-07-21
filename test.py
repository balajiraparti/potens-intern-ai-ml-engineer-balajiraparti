from langchain_huggingface import HuggingFaceEmbeddings
# from huggingface_hub import login
# import os

# login(token=os.getenv("HF_TOKEN"))
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

query = "What is the coverage for accidental damage?"

embedding = embedding_model.embed_query(query)

print(len(embedding))
print(embedding[:5])