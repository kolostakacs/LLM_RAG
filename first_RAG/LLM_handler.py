from query_translation.query_translation_methods import (
    generate_roadmap,
    stepwise_search_and_answer,
    final_decision_maker,
    generate_subqueries,
    multi_search_chroma,
)
import openai


'''
Ez a step-by step generáláshoz kapcsolódik
def ask_chatbot(user_query):
    """Az új roadmap-alapú chatbot kérdésfeldolgozása."""
    roadmap_steps = generate_roadmap(user_query)  # 🔹 1. Roadmap generálása
    accumulated_answers, context_history = stepwise_search_and_answer(roadmap_steps)  # 🔹 2. Lépésenkénti keresés
    content, final_response = final_decision_maker(user_query, roadmap_steps, context_history)  # 🔹 3. Döntéshozás

    return content, final_response
'''

def ask_chatbot(user_query):
    """A chatbot először három al-kérdésre bontja a kérést, majd az ezekre kapott keresési eredményekkel válaszol."""
    subqueries = generate_subqueries(user_query)
    retrieved_text = multi_search_chroma(subqueries)

    prompt = f"""Használj releváns információkat az alábbi szövegből a válaszhoz. A válasz legyen tömör és lényegre törő. Ne legyen több mint 3 mondat a felsorolásokat kivéve.
    Ha valami egy mondattal is megválaszolható, törekedj arra, hogy úgy válaszold meg. 
    Mindig csak a kérdésre válaszolj, ne adj addicionális információt. 

    --- Források ---  
    {retrieved_text}  
    ----------------  

    Kérdés: {user_query}  

    Adj pontos választ rövid mondatokkal vagy bulletpointokkal. Ne adj extra magyarázatot, csak a lényeges információt.  
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Te a Gránit bank asszisztense vagy, és segíted az ügyfeleket az ügyeik intézésében úgy, hogy információt keresel és összegzed."},
            {"role": "user", "content": prompt}
        ]
    )

    return retrieved_text, response.choices[0].message.content
    #return response.choices[0].message.content

