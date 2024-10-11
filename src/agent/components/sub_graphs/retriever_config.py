import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def setup_chroma_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    chroma_directory = os.getenv("CHROMADB_DIRECTORY")

    vector_store = Chroma(
        collection_name="obp_endpoints",
        embedding_function=embeddings,
        persist_directory=chroma_directory
    )

    return vector_store

def setup_retriever(k: int, vector_store: Chroma):
    endpoint_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return endpoint_retriever
    
