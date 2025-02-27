import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from chromadb.utils import embedding_functions
import os

api_key = os.getenv("OPENAI_API_KEY")  # Biztonságos elérés
if api_key is None:
    raise ValueError("API kulcs nincs beállítva!")
# OpenAI embedding függvény beállítása
openai_ef = OpenAIEmbeddings(
    api_key=api_key,
    model_name="text-embedding-ada-002"  # Ez 1536 dimenziós embeddinget ad
)

# ChromaDB kliens inicializálása (ugyanaz az elérési út, mint a tárolásnál)
chroma_client = chromadb.PersistentClient(path="C:/Users/device/Desktop/work/Projects/LLM_RAG/chunking/chromadb")

# Gyűjtemény megnyitása
collection = chroma_client.get_collection(name="faq_embeddings", embedding_function=openai_ef)

# Példa keresési lekérdezés
query = "Milyen hitelek érhetők el a banknál?"

# Keresési lekérdezés beágyazása az OpenAI embedding függvény segítségével
query_embedding = openai_ef.embed_query(query)  # Az embed_query metódust kell használni

# Legközelebbi találatok lekérése
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3  # Hány találatot adjon vissza
)

# Eredmények kiírása
for i, (metadata, score) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
    print(f"\nTalálat {i+1}:")
    print(f"Cím: {metadata['cím']}")
    print(f"Leírás: {metadata['leírás']}")
    print(f"Pontosság: {score:.4f}")
    print(f"Query embedding dimenziója: {len(query_embedding)}")


