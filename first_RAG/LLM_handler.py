import openai
from chunking.Chroma_db_handler import init_chroma_client, get_collection


chroma_client = init_chroma_client()
collection = get_collection(chroma_client)

def search_chroma(query, top_k=7):
    """Keresés a ChromaDB-ben és releváns dokumentumok visszaadása."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    if results["documents"]:
        retrieved_docs = "\n\n".join(results["documents"][0])
    else:
        retrieved_docs = "Nem találtunk releváns információt."

    return retrieved_docs, results

def ask_chatbot(user_query):
    """A chatbot megkeresi a releváns információkat, majd LLM segítségével választ generál."""
    retrieved_text, search_results = search_chroma(user_query, top_k=5)

    print("Talált dokumentumok a ChromaDB-ben:")
    print(retrieved_text)

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