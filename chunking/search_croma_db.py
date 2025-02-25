import chromadb
from sentence_transformers import SentenceTransformer

# ChromaDB kliens inicializálása (ugyanaz az elérési út, mint a tárolásnál)
chroma_client = chromadb.PersistentClient(path="./chromadb")

# Gyűjtemény megnyitása
collection = chroma_client.get_collection(name="faq_embeddings")

# Modell betöltése a kereséshez
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Példa keresési lekérdezés
query = "Milyen hitelek érhetők el a banknál?"
query_embedding = model.encode(query).tolist()

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
