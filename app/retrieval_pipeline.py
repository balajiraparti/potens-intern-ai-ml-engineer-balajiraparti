from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_community.vectorstores import Chroma
load_dotenv()
client=OpenAI()
def retrive_content(user_query:str):
    embedding_model=HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embedding_model
        )
    userquery=input("Ask something:")
    search_result=vectorstore.similarity_search(query=userquery)
# context="\n\n\n".join([f"Page Content:{result.page_content}\nPage Number:{result.metadata['page_label']}\nFile Location:{result.metadata['source']}" for result in search_result])
# system_prompt=f"""You are helpful assistant with who answers user query based on available context retrieved from a pdf file along with page number and page_contents.
# you should only ans the user based on the following context and navigate the user to open the right page number to know more.
# Context:
# {context}
# """
# response=client.chat.completions.create(
#     model="gpt-4-0",
#     messages=[
#         {"role":"system","content":system_prompt},
#         {"role":"user","content":userquery}
#     ]
# )
# print("bot: ",response.choices[0].message.content)