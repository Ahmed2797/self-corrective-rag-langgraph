from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from src.logger import logging
from src.exception import CustomException
from src.chat_model import get_embeddings
from src.constants import K, IMAGE_DIR
from src.vector import create_pincone_database

import uuid
import fitz
import pdfplumber
import pandas as pd
import os


def create_retriever(documents, chunk_size=500, chunk_overlap=50, k=K, index_name=None):
    """
    Create a retriever using Pinecone or FAISS.

    Args:
        documents (str | list[str]): Path to a single PDF or a list of PDF paths.
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
        k (int): Number of documents to retrieve.
        index_name (str): Name of the Pinecone index to use.

    Returns:
        VectorStoreRetriever: Configured document retriever.

    Raises:
        CustomException: If retriever creation fails.
    """
    try:
        logging.info("Starting retriever creation.")

        if isinstance(documents, str):
            documents = [documents]

        if index_name:
            logging.info(f"Creating Pinecone retriever for index: {index_name}")
            retriever = create_retriever_pinecone(documents, index_name=index_name, k=k, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            logging.info("Creating FAISS retriever.")

            docs = []

            for document in documents:
                logging.info(f"Loading document: {document}")
                docs.extend(PyPDFLoader(document).load())

            logging.info(f"Loaded {len(docs)} document pages.")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_documents(docs)

            logging.info(f"Created {len(chunks)} text chunks.")

            embeddings = get_embeddings()
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": k})

            logging.info("FAISS retriever created successfully.")

        return retriever

    except Exception as e:
        logging.error(f"Error while creating retriever: {str(e)}")
        raise CustomException(e)


def create_retriever_pinecone(documents, index_name, k=K, chunk_size=500, chunk_overlap=50):
    """
    Process one or multiple PDF documents and create a Pinecone retriever.

    Args:
        documents (str | list[str]): Path to a single PDF or a list of PDF paths.
        index_name (str): Name of the Pinecone index.
        k (int): Number of documents to retrieve.
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        VectorStoreRetriever: Pinecone-based document retriever.

    Raises:
        CustomException: If Pinecone processing fails.
    """
    try:
        logging.info("Starting Pinecone retriever creation.")

        if isinstance(documents, str):
            documents = [documents]

        logging.info(f"Processing {len(documents)} document(s).")
        logging.info(f"Using Pinecone index: {index_name}")

        index = create_pincone_database(index_name=index_name)

        logging.info("Pinecone index initialized successfully.")

        image_files = []
        embeddings = get_embeddings()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for file_path in documents:
            logging.info(f"Processing document: {file_path}")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Document not found: {file_path}")

            doc = fitz.open(file_path)

            logging.info(f"PDF opened successfully: {file_path}")
            logging.info(f"Number of pages: {len(doc)}")

            try:
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text()

                        if text:
                            text_chunks = text_splitter.split_text(text)

                            logging.info(f"Page {i + 1}: created {len(text_chunks)} text chunks.")

                            for chunk_idx, chunk in enumerate(text_chunks):
                                vector_id = f"p{i + 1}_txt_c{chunk_idx}_{uuid.uuid4().hex[:6]}"
                                vector = embeddings.embed_query(chunk)

                                index.upsert([(vector_id, vector, {"text": chunk, "type": "text", "page": i + 1})])

                        tables = page.extract_tables()

                        if tables:
                            logging.info(f"Page {i + 1}: found {len(tables)} table(s).")

                        for j, table in enumerate(tables):
                            if table and len(table) > 1:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                table_md = df.to_markdown(index=False)
                                vector_id = f"p{i + 1}_tbl_{j}_{uuid.uuid4().hex[:6]}"
                                vector = embeddings.embed_query(table_md)

                                index.upsert([(vector_id, vector, {"text": table_md, "type": "table", "page": i + 1})])

                logging.info("Text and table extraction completed.")

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    images = page.get_images(full=True)

                    if images:
                        logging.info(f"Page {page_num + 1}: found {len(images)} image(s).")

                    for img_index, img in enumerate(images):
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)

                        if pix.n >= 5:
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        image_id = uuid.uuid4().hex[:6]
                        img_name = f"p{page_num + 1}_img_{img_index}_{image_id}.png"
                        img_path = os.path.join(IMAGE_DIR, img_name)

                        pix.save(img_path)
                        image_files.append(img_name)

                        context = page.get_text("text")[:500]
                        vector_id = f"p{page_num + 1}_img_{img_index}_{image_id}"
                        vector = embeddings.embed_query(context)

                        index.upsert([(vector_id, vector, {"text": context, "type": "image", "file": img_name, "page": page_num + 1})])

                        pix = None

                logging.info("Image extraction completed.")

            finally:
                doc.close()

            logging.info(f"Finished processing document: {file_path}")

        retriever = PineconeVectorStore(index=index, embedding=embeddings).as_retriever(search_kwargs={"k": k})

        logging.info("Pinecone retriever created successfully.")
        logging.info(f"Extracted {len(image_files)} image(s).")

        return retriever

    except Exception as e:
        logging.error(f"Error while creating Pinecone retriever: {str(e)}")
        raise CustomException(e)