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
    retrieved_text, search_results = search_chroma(user_query, top_k=7)

    print(user_query)

    content = [f"{item['cím']} - {item['leírás']}" for item in search_results["metadatas"][0]]
    print(content)

    prompt = f"""Használj releváns információkat az alábbi szövegből a válaszhoz. A válasz legyen tömör és lényegre törő. Ne legyen több mint 3 mondat a felsorolásokat kivéve. 
    Abban az esetben ha valami egy mondattal is megválaszolható törekedj arra hogy úgy válaszold meg pl ha valami nem elérhető akkor csak azt add vissza hogy a termék nem elérhető
    Mindig csak a kérdésre válaszolj a kérdést nem kell absztraktan értelmezned és addícionális információt adnod. Figyelj az egyértelmű egyszerű információ átadására ami megválaszolja a kérdést
     

        --- Források ---  
        {content}  
        ----------------  

        Kérdés: {user_query}  

        Adj pontos választ rövid mondatokkal vagy bulletpointokkal. Ne adj extra magyarázatot, csak a lényeges információt.  
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Te a Gránit bank asszisztense vagy és segíted az ügyfeleket az ügyeik intézésével úgy hogy információt keresel és összegzve ezeket átadod."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content