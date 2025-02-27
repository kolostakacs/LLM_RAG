import openai
import json
import numpy as np
import os
# OpenAI API beállítása
api_key = os.getenv("OPENAI_API_KEY")  # Biztonságos elérés
if api_key is None:
    raise ValueError("API kulcs nincs beállítva!")
# Betöltjük a JSON fájlt
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Embedding-ek generálása az OpenAI modellel
for chunk in chunks:
    full_text = f"{chunk['cím']} {chunk['leírás']}"  # Egyesítjük a két mezőt

    # OpenAI embedding generálás
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=full_text
    )

    chunk["embedding"] = response['data'][0]['embedding']

# Embedding-ek mentése NumPy formátumban
embeddings = np.array([chunk["embedding"] for chunk in chunks])
np.save("embeddings.npy", embeddings)

# Frissített JSON mentése
with open("chunks_with_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=4)

print("Embedding-ek (cím + leírás) elkészültek és elmentve.")