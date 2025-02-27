import json
import openai
import chromadb
import numpy as np
from chromadb.utils import embedding_functions
import os


# OpenAI API Key beállítása (Cseréld ki a saját kulcsodra!)
api_key = os.getenv("OPENAI_API_KEY")  # Biztonságos elérés
if api_key is None:
    raise ValueError("API kulcs nincs beállítva!")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-ada-002"  # Ez 1536 dimenziós embeddinget ad
)
# ChromaDB inicializálása és vektoros kereső beállítása
chroma_client = chromadb.PersistentClient(path="C:/Users/device/Desktop/work/Projects/LLM_RAG/chunking/chromadb")  # Ha már van adatbázisod
collection = chroma_client.get_collection(name="faq_embeddings", embedding_function=openai_ef)

def search_chroma(query, top_k=7):
    """Keresés a ChromaDB-ben és releváns dokumentumok visszaadása."""
    # A query embeddinget most már nem generáljuk, hanem közvetlenül keresünk a ChromaDB-ben tárolt embeddingek között.
    results = collection.query(
        query_texts=[query],  # Keresés szöveges lekérdezéssel (nem szükséges az embedding generálás)
        n_results=top_k  # Hány találatot szeretnénk visszakapni
    )

    # Ha nincs találat, akkor is adunk választ
    if results["documents"]:
        retrieved_docs = "\n\n".join(results["documents"][0])  # Az első találat összes dokumentuma
    else:
        retrieved_docs = "Nem találtunk releváns információt."

    return retrieved_docs, results


def ask_chatbot(user_query):
    """A chatbot megkeresi a releváns információkat, majd LLM segítségével választ generál."""
    retrieved_text, search_results = search_chroma(user_query, top_k=5)  # Növelt találatok száma

    print("Talált dokumentumok a ChromaDB-ben:")
    print(retrieved_text)
    print("Keresési eredmények:")
    content = [f"{item['cím']} - {item['leírás']}" for item in search_results["metadatas"][0]]
    print(content)


    prompt = f"""Használj releváns információkat az alábbi szövegből a válaszhoz:

    --- Források ---
    {content}
    ----------------

    Kérdés: {user_query}

    Adj pontos és érthető választ a fenti információk alapján."""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Te egy segítőkész AI vagy, amely információt keres és összegzi."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def run_tests(question_list, output_file="test_results.json"):
    results = []

    for idx, question in enumerate(question_list):
        response_text = ask_chatbot(question)  # A chatbot válasza
        retrieved_text, search_results = search_chroma(question, top_k=5)

        retrieved_context = []
        for i, doc in enumerate(search_results["metadatas"][0]):
            retrieved_context.append({
                "doc_id": f"{i:03}",
                "text": doc
            })

        results.append({
            "query_id": str(idx),
            "query": question,
            "gt_answer": "",  # Ground truth answer később tölthető
            "response": response_text,
            "retrieved_context": retrieved_context
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=4)

    print(f"Teszt eredmények elmentve: {output_file}")


# Példa kérdéslista
questions = [
    "Milyen hitelek érhetőek el a banknál",
    "Bar listások kaphatnak hitelt?",
    "van korhatári megkötés a nyugdíjasok hiteleinél?"
]

run_tests(questions)
