import os

# OpenAI API kulcs
API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None:
    raise ValueError("API kulcs nincs beállítva!")

# Embedding modell
EMBEDDING_MODEL = "text-embedding-ada-002"

# ChromaDB adatbázis útvonala
CHROMA_DB_PATH = "C:/Users/device/Desktop/work/Projects/LLM_RAG/chunking/chromadb"

# Keresési találatok száma
TOP_K_RESULTS = 7