import sys
import sqlite3
from src.retrive import get_project_retriever
from src.graph import create_graph
from src.exception import CustomException
from src.logger import logging

from langgraph.checkpoint.sqlite import SqliteSaver

# Initialize SQLite database connection for checkpointing
db_conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(db_conn)


def pipeline(project_id: str):
    """
    Create the retriever and compile the LangGraph RAG pipeline.

    Returns:
        CompiledStateGraph: Compiled LangGraph application.

    Raises:
        CustomException: If retriever or graph creation fails.
    """
    try:
        logging.info("Starting the RAG pipeline.")

        # Ensure project_id matches Pinecone index naming rules (lowercase, hyphens only)
        formatted_project_id = str(project_id).lower().replace("_", "-")

        logging.info(f"Creating document retriever for project: {formatted_project_id}")
        retriever = get_project_retriever(formatted_project_id)

        logging.info("Retriever created successfully.")

        logging.info("Creating LangGraph application.")
        app = create_graph(retriever=retriever, checkpointer=memory)

        logging.info("LangGraph application created successfully.")

        return app

    except Exception as e:
        logging.error(f"Error in RAG pipeline: {str(e)}")
        raise CustomException(e, sys)
    