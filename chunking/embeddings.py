import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Betöltjük a JSON fájlt
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Modell betöltése az embedding-ekhez
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Embedding-ek generálása (cím + leírás együtt)
for chunk in chunks:
    full_text = f"{chunk['cím']} {chunk['leírás']}"  # Egyesítjük a két mezőt
    chunk["embedding"] = model.encode(full_text).tolist()

# Embedding-ek mentése NumPy formátumban
embeddings = np.array([chunk["embedding"] for chunk in chunks])
np.save("embeddings.npy", embeddings)

# Frissített JSON mentése
with open("chunks_with_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=4)

print("Embedding-ek (cím + leírás) elkészültek és elmentve.")
