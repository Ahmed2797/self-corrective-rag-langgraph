import os
from pinecone import Pinecone, ServerlessSpec
from src.chat_model import get_embeddings


def create_pincone_database(index_name:str):
    """
    Create a Pinecone database if it doesn't exist and connect to it.

    Returns:
        pc.Index: Connected Pinecone index.
    """
    # 1. Check if the index exists; if not, create it
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is required to use Pinecone.")
    pc = Pinecone(api_key=api_key)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=1536,  # 1536 matches OpenAI text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # 2. Connect to the index
    cache_index = pc.Index(index_name)

    return cache_index



def create_embedding(text: str) -> list[float]:
    # embed_query takes a string and returns a list of floats
    return get_embeddings().embed_query(text)
