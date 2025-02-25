import json
import chromadb
import numpy as np

# ChromaDB kliens inicializálása
chroma_client = chromadb.PersistentClient(path="./chromadb")  # Tartós tárolás

# Új vektorgyűjtemény létrehozása
collection = chroma_client.get_or_create_collection(name="faq_embeddings")

# Betöltjük az előzőleg generált JSON-t
with open("chunks_with_embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Adatok feltöltése a vektoradatbázisba
for chunk in chunks:
    collection.add(
        ids=[str(chunk["chunk_id"])],  # Azonosítóként az indexet használjuk
        embeddings=[chunk["embedding"]],  # Az előző lépésben generált embedding
        metadatas=[{"cím": chunk["cím"], "leírás": chunk["leírás"]}]  # Metaadatok
    )

print("Embedding-ek sikeresen betöltve a ChromaDB-be!")
