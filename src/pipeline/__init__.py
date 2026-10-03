from src.retrive import get_project_retriever
# from src.constants import CHUNK_SIZE, CHUNK_OVERLAP, K, DOCUMENTS
from src.graph import create_graph
from src.exception import CustomException
from src.logger import logging
import sys

from langgraph.checkpoint.sqlite import SqliteSaver
# MemorySaver with:
memory = SqliteSaver.from_conn_string("file:///path/to/db.sqlite")


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

        logging.info("Creating document retriever.")
        retriever = get_project_retriever(project_id)

        logging.info("Retriever created successfully.")

        logging.info("Creating LangGraph application.")
        app = create_graph(retriever=retriever)
        # app = create_graph(retriever=retriever,checkpointer=memory)

        logging.info("LangGraph application created successfully.")

        return app

    except Exception as e:
        logging.error(f"Error in RAG pipeline: {str(e)}")
        raise CustomException(e, sys)
