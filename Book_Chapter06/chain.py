
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from databricks.vector_search.client import VectorSearchClient
import mlflow

catalog = "porya_catalog"
database_name = "default"

embedding_model = DatabricksEmbeddings(endpoint="databricks-gte-large-en")
vsc_endpoint_name = "one-env-shared-endpoint-1"
index_name = f"{catalog}.{database_name}.docs_vsc_idx_cont"

def get_retriever(persist_dir=None):
    vsc = VectorSearchClient(
        workspace_url=os.environ.get("DATABRICKS_HOST", ""),
        personal_access_token=os.environ.get("DATABRICKS_TOKEN", "")
    )
    vs_index = vsc.get_index(endpoint_name=vsc_endpoint_name, index_name=index_name)
    vectorstore = DatabricksVectorSearch(vs_index, text_column="content", embedding=embedding_model)
    return vectorstore.as_retriever()

chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens=200)

TEMPLATE = """
You are an assistant for the AI Swat Team. You are answering questions related to the GenerativeAI and LLM's and how they impact humans life, labour, economic and financial impact. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = get_retriever()

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

mlflow.models.set_model(chain)
