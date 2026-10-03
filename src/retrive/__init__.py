import os
import sys
import uuid
import fitz
import shutil
import pdfplumber
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from src.logger import logging
from src.exception import CustomException
from src.chat_model import get_embeddings
from src.constants import K, IMAGE_DIR, FAISS_DB_PATH
from src.vector import create_pincone_database


PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME",None)
FAISS_PROJECTS_ROOT = f"{FAISS_DB_PATH}_projects"
EMBED_BATCH = 100


def _splitter(chunk_size=500, chunk_overlap=50):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# ---------------------------- extraction ---------------------------------------
def _extract_pdf(path: str, project_id: str, splitter) -> list[Document]:
    """Text chunks, tables (as markdown) and image records from one PDF."""
    source = os.path.basename(path)
    docs: list[Document] = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                for chunk in splitter.split_text(text):
                    docs.append(Document(
                        page_content=chunk,
                        metadata={"type": "text", "page": i, "source": source},
                    ))

            for table in page.extract_tables() or []:
                if table and len(table) > 1:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    docs.append(Document(
                        page_content=df.to_markdown(index=False),
                        metadata={"type": "table", "page": i, "source": source},
                    ))

    img_dir = os.path.join(IMAGE_DIR, project_id)
    os.makedirs(img_dir, exist_ok=True)

    with fitz.open(path) as pdf_doc:
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            context = page.get_text("text")[:500]
            for img_index, img in enumerate(page.get_images(full=True)):
                pix = fitz.Pixmap(pdf_doc, img[0])
                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_name = f"p{page_num + 1}_img_{img_index}_{uuid.uuid4().hex[:6]}.png"
                pix.save(os.path.join(img_dir, img_name))
                pix = None
                if context.strip():  # skip images with no surrounding text to embed
                    docs.append(Document(
                        page_content=context,
                        metadata={
                            "type": "image",
                            "page": page_num + 1,
                            "source": source,
                            "file": os.path.join(project_id, img_name),
                        },
                    ))
    return docs


def _extract_text_file(path: str, splitter) -> list[Document]:
    source = os.path.basename(path)
    docs = TextLoader(path, encoding="utf-8", autodetect_encoding=True).load()
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata.update({"type": "text", "source": source})
    return chunks


def _extract(path: str, project_id: str, splitter) -> list[Document]:
    if path.lower().endswith(".pdf"):
        return _extract_pdf(path, project_id, splitter)
    return _extract_text_file(path, splitter)


# ---------------------------- FAISS backend ------------------------------------
def _faiss_dir(project_id: str) -> str:
    return os.path.join(FAISS_PROJECTS_ROOT, project_id)


class _EmptyRetriever:
    """FAISS-compatible empty result for projects with no uploaded files."""

    @staticmethod
    def similarity_search_with_score(query: str, k: int = K):
        return []


def _index_faiss(project_id: str, docs: list[Document]) -> None:
    embeddings = get_embeddings()
    path = _faiss_dir(project_id)
    if os.path.exists(path):
        store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        store.add_documents(docs)  # embeds only the new chunks
    else:
        store = FAISS.from_documents(docs, embeddings)
    store.save_local(path)


# ---------------------------- Pinecone backend ---------------------------------
def _index_pinecone(project_id: str, docs: list[Document]) -> None:
    index = create_pincone_database(index_name=PINECONE_INDEX_NAME)
    embeddings = get_embeddings()

    for start in range(0, len(docs), EMBED_BATCH):
        batch = docs[start:start + EMBED_BATCH]
        vectors = embeddings.embed_documents([d.page_content for d in batch])
        index.upsert(
            vectors=[
                (
                    f"{d.metadata['type']}_{uuid.uuid4().hex[:12]}",
                    vec,
                    {**d.metadata, "text": d.page_content},
                )
                for d, vec in zip(batch, vectors)
            ],
            namespace=project_id,
        )


# ---------------------------- public API ---------------------------------------
def index_files(project_id: str, file_paths: list[str], chunk_size=500, chunk_overlap=50) -> int:
    """Extract, chunk, embed and store files for one project. Returns chunks added."""
    try:
        splitter = _splitter(chunk_size, chunk_overlap)
        docs: list[Document] = []
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Document not found: {path}")
            logging.info(f"Extracting: {path}")
            docs.extend(_extract(path, project_id, splitter))

        if not docs:
            logging.warning("No extractable text found (scanned PDF?).")
            return 0

        if PINECONE_INDEX_NAME:
            _index_pinecone(project_id, docs)
        else:
            _index_faiss(project_id, docs)

        logging.info(f"Indexed {len(docs)} chunks for project {project_id}.")
        return len(docs)
    except Exception as e:
        logging.error(f"Error indexing files: {e}")
        raise CustomException(e, sys)


def get_project_retriever(project_id: str, k: int = K):
    """Retriever limited to one project's documents."""
    embeddings = get_embeddings()

    if PINECONE_INDEX_NAME:
        index = create_pincone_database(index_name=PINECONE_INDEX_NAME)
        logging.info(f"PINECONE_INDEX_NAME {PINECONE_INDEX_NAME} Init.")

        store = PineconeVectorStore(
            index=index, embedding=embeddings, namespace=project_id
        )
        # The graph calls similarity_search_with_score(query, k=5) directly.
        return store

    path = _faiss_dir(project_id)
    logging.info(f"FAISS_Database {path} Init.")

    if not os.path.exists(path):
        return _EmptyRetriever()  # no uploads yet
    logging.info(f"FAISS_Database {path} Doesn't exits.")
    
    store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    logging.info(f"FAISS_Database {path} exits and load local.")

    # The graph needs scored results for its hybrid reranking step.
    return store


def delete_index(project_id: str) -> None:
    """Remove a project's vectors and extracted images."""
    shutil.rmtree(os.path.join(IMAGE_DIR, project_id), ignore_errors=True)
    if PINECONE_INDEX_NAME:
        index = create_pincone_database(index_name=PINECONE_INDEX_NAME)
        try:
            index.delete(delete_all=True, namespace=project_id)
        except Exception as e:  # namespace may not exist yet
            logging.warning(f"Pinecone namespace delete skipped: {e}")
    else:
        shutil.rmtree(_faiss_dir(project_id), ignore_errors=True)
