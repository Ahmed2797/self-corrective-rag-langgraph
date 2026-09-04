from src.retrive import create_retriever
from src.constants import CHUNK_SIZE, CHUNK_OVERLAP, K, DOCUMENTS
from src.exception import CustomException
from src.logger import logging

def pipeline():
    """
    Main pipeline function to create a retriever and perform retrieval.

    Returns:
        VectorStoreRetriever: Configured document retriever.
    """
    try:
        logging.info("Starting the retrieval pipeline.")
        print("Starting the retrieval pipeline.")
        # Example usage of the create_retriever function
        documents = DOCUMENTS  # Use the documents defined in constants
        retriever = create_retriever(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, k=K)
        logging.info("Retriever created successfully.")
        print("Retriever created successfully.")

        return None
    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}")
        raise CustomException(e)