CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
K = 4
MAX_REWRITE_TRIES = 3
INDEX_NAME_CACHE_MEMORY = "crag-semantic-answer-cache"
INDEX_NAME = "self-corrective-rag-langgraph"
IMAGE_DIR = "src/data/images"
DOCUMENTS = ["src/data/AI_Engineer-page1.pdf", "src/data/Tanvir_Ahmed_Enterprise_Demo_RAG_Knowledge_Base.pdf", "src/data/Tanvir_Ahmed_RAG_Knowledge_Base.pdf"]
CACHE_THRESHOLD = 0.90
CACHE_VERSION = "self-rag-v1"
# FAISS returns squared L2 distances (lower is closer). A 0.30 cutoff was
# rejecting all retrieved chunks in normal document queries.
SIMILARITY_THRESHOLD_FIASS = 0.80
SIMILARITY_THRESHOLD_PINECONE = 0.70
FAISS_DB_PATH = "./faiss_index"
