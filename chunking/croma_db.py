from langchain.embeddings.openai import OpenAIEmbeddings
import chromadb
from chromadb.utils import embedding_functions
import json
import os

api_key = os.getenv("OPENAI_API_KEY")  # Biztonságos elérés
if api_key is None:
    raise ValueError("API kulcs nincs beállítva!")
# OpenAI embedding függvény beállítása
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-ada-002"  # Ez 1536 dimenziós embeddinget ad
)

# ChromaDB kliens inicializálása
chroma_client = chromadb.PersistentClient(path="C:/Users/device/Desktop/work/Projects/LLM_RAG/chunking/chromadb")

# Kollekció törlése, ha már létezik
try:
    chroma_client.delete_collection(name="faq_embeddings")
    print("A kollekció törölve.")
except Exception as e:
    print(f"A kollekció törlése nem sikerült: {e}")

# Új kollekció létrehozása OpenAI embedding funkcióval
collection = chroma_client.get_or_create_collection(name="faq_embeddings", embedding_function=openai_ef)

# Betöltjük az előzőleg generált JSON-t
with open("chunks_with_embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Adatok feltöltése a vektoradatbázisba
for chunk in chunks:
    collection.add(
        ids=[str(chunk["chunk_id"])],  # Azonosítóként az indexet használjuk
        embeddings=[chunk["embedding"]],  # Az előző lépésben generált embedding
        metadatas=[{"cím": chunk["cím"], "leírás": chunk["leírás"]}],  # Metaadatok
        documents=["Gránit_faq"]  # Feltételezve, hogy van "document" kulcs a chunk-ban
    )

print("Embedding-ek sikeresen betöltve a ChromaDB-be!")

# Snapshot a kollekcióról: Adatok lekérése és megjelenítése
def snapshot_collection(collection):
    snapshot = collection.get(include=["embeddings", "metadatas", "documents"])  # Lekéri a vektorokat, metaadatokat és a dokumentumokat
    print("\nSnapshot az adatbázisból:")
    print(f"Összes rekord száma: {len(snapshot['documents'])}")
    for i in range(min(5, len(snapshot["documents"]))):  # Az első 5 rekordot mutatjuk meg, ha van
        print(f"Dokumentum: {snapshot['documents'][i]}")
        print(f"Metaadatok: Cím: {snapshot['metadatas'][i]['cím']}, Leírás: {snapshot['metadatas'][i]['leírás']}")
        print("-" * 50)


first_chunk_embedding = chunks[0]["embedding"]

# Snapshot elkészítése és megjelenítése
snapshot_collection(collection)
embedding_dimension = len(first_chunk_embedding)
print(f"Az első embedding dimenziója: {embedding_dimension}")


