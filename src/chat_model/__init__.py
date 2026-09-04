from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
    """
    Initialize and return the OpenAI chat language model.

    Returns:
        ChatOpenAI: Configured GPT-4o-mini chat model with deterministic
        responses using temperature=0.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    return llm


def get_embeddings():
    """
    Initialize and return the OpenAI embedding model.

    Returns:
        OpenAIEmbeddings: Configured text-embedding-3-small model
        used to convert text into vector embeddings for semantic search
        and retrieval.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    return embeddings

