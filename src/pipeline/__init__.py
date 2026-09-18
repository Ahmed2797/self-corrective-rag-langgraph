from src.retrive import create_retriever
from src.constants import CHUNK_SIZE, CHUNK_OVERLAP, K, DOCUMENTS
from src.graph import create_graph
from src.exception import CustomException
from src.logger import logging


def pipeline():
    """
    Create the retriever and compile the LangGraph RAG pipeline.

    Returns:
        CompiledStateGraph: Compiled LangGraph application.

    Raises:
        CustomException: If retriever or graph creation fails.
    """
    try:
        logging.info("Starting the RAG pipeline.")

        documents = DOCUMENTS

        logging.info("Creating document retriever.")
        retriever = create_retriever(
            documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            k=K
        )

        logging.info("Retriever created successfully.")

        logging.info("Creating LangGraph application.")
        app = create_graph(retriever=retriever)

        logging.info("LangGraph application created successfully.")

        return app

    except Exception as e:
        logging.error(f"Error in RAG pipeline: {str(e)}")
        raise CustomException(e)