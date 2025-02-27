from config import API_KEY, EMBEDDING_MODEL, CHROMA_DB_PATH, TOP_K_RESULTS
from chromadb.utils import embedding_functions
import chromadb
import json


openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name=EMBEDDING_MODEL  # Ez 1536 dimenziós embeddinget ad
)


# ChromaDB kliens inicializálása
def init_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_collection(client, name="faq_embeddings"):
    return client.get_or_create_collection(name=name, embedding_function=openai_ef)


def delete_collection(client, name="faq_embeddings"):
    try:
        client.delete_collection(name=name)
        print("A kollekció törölve.")
    except Exception as e:
        print(f"A kollekció törlése nem sikerült: {e}")


def load_chunks_to_chroma(collection, json_path="chunks_with_embeddings.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    for chunk in chunks:
        collection.add(
            ids=[str(chunk["chunk_id"])],
            embeddings=[chunk["embedding"]],
            metadatas=[{"cím": chunk["cím"], "leírás": chunk["leírás"]}],
            documents=["Gránit_faq"]
        )
    print("Embedding-ek sikeresen betöltve a ChromaDB-be!")


def snapshot_collection(collection, limit=5):
    snapshot = collection.get(include=["embeddings", "metadatas", "documents"])
    print("\nSnapshot az adatbázisból:")
    print(f"Összes rekord száma: {len(snapshot['documents'])}")
    for i in range(min(limit, len(snapshot["documents"]))):
        print(f"Dokumentum: {snapshot['documents'][i]}")
        print(f"Metaadatok: Cím: {snapshot['metadatas'][i]['cím']}, Leírás: {snapshot['metadatas'][i]['leírás']}")
        print("-" * 50)


if __name__ == "__main__":
    client = init_chroma_client()
    delete_collection(client)
    collection = get_collection(client)
    load_chunks_to_chroma(collection)
    snapshot_collection(collection)