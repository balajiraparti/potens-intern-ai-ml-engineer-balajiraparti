from dotenv import load_dotenv

from openai import OpenAI
from langchain_community.vectorstores import Chroma
load_dotenv()
client=OpenAI()
def build_context_for_llm(search_result,user_query):
    context="\n\n\n".join([f"Page Content:{result.page_content}\nPage Number:{result.metadata['page_label']}\nFile Location:{result.metadata['source']}" for result in search_result])
    system_prompt=f"""You are helpful assistant with who answers user query based on available context retrieved from a pdf file along with page number and page_contents.
    you should only ans the user based on the following context and navigate the user to open the right page number to know more.
    Context:
    {context}
    """
    response=client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_query}
        ]
    )
    return response.choices[0].message.content