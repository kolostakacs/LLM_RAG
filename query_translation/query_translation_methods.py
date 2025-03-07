import openai
from chunking.Chroma_db_handler import search_chroma


def generate_subqueries(user_query):
    """Az LLM segítségével a felhasználói kérdést három különböző aspektusból szétbontja."""
    prompt = f"""
    A felhasználó az alábbi kérdést tette fel:
    "{user_query}"

    A feladatod, hogy ezt a kérdést három különböző aspektusból lebontsd három külön al-kérdésre.
    A kérdéseknek más-más nézőpontból kell megközelíteniük a témát.

    Adj vissza pontosan három kérdést az alábbi formátumban:
    1. [első al-kérdés]
    2. [második al-kérdés]
    3. [harmadik al-kérdés]
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    generated_text = response.choices[0].message.content
    subqueries = [line.split(". ", 1)[1] for line in generated_text.split("\n") if line.startswith(tuple("123"))]

    return subqueries if len(subqueries) == 3 else [user_query]


def multi_search_chroma(subqueries):
    """Mindhárom al-kérdésre végrehajt keresést a ChromaDB-ben, majd az eredményeket egyesíti, kiszűrve a duplikációkat."""
    all_content = []
    seen_ids = set()

    for subquery in subqueries:
        retrieved_text, search_results = search_chroma(subquery, top_k=4)

        if search_results["metadatas"]:
            for i, item in enumerate(search_results["metadatas"][0]):
                chunk_id = search_results["ids"][0][i]

                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    all_content.append(f"{item['cím']} - {item['leírás']}")

    return all_content


def generate_roadmap(user_query):
    """LLM segítségével létrehoz egy háromlépéses roadmapet a válaszhoz."""
    prompt = f"""
    A felhasználó az alábbi kérdést tette fel:
    "{user_query}"

    A feladatod, hogy bonts 3 lépéses folyamatra rész kérdésekre, amelyek segíthetnek a végső válasz meghatározásában.
    Adj vissza pontosan három lépést az alábbi formátumban:
    1. [első lépés]
    2. [második lépés]
    3. [harmadik lépés]
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    roadmap_text = response.choices[0].message.content
    steps = [line.split(". ", 1)[1] for line in roadmap_text.split("\n") if line.startswith(tuple("123"))]

    return steps if len(steps) == 3 else [user_query]


def stepwise_search_and_answer(roadmap_steps):
    """Lépésenként végrehajtja a keresést és válaszadást."""
    accumulated_answers = []
    seen_ids = set()
    context_history = ""

    for step_index, step_question in enumerate(roadmap_steps):
        retrieved_text, search_results = search_chroma(step_question, top_k=4)

        step_context = []
        for i, doc in enumerate(search_results["metadatas"][0]):
            chunk_id = search_results["ids"][0][i]

            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                step_context.append(f"{doc['cím']} - {doc['leírás']}")

        prompt = f"""
        Figyelembe véve az alábbi korábbi kontextust:
        {context_history}

        Most válaszolj a következő lépés kérdésére az alábbi információk alapján:

        --- Források ---  
        {"\n".join(step_context)}  
        ----------------  

        Kérdés: {step_question}

        Adj egy rövid és lényegre törő választ.
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        step_answer = response.choices[0].message.content
        accumulated_answers.append(f"Lépés {step_index + 1}: {step_answer}")

        context_history += f"\nLépés {step_index + 1} válasza: {step_answer}\n"

    return accumulated_answers, context_history


def final_decision_maker(user_query, roadmap_steps, context_history):
    """A chatbot a roadmap válaszai és az eredeti kérdés alapján hozza meg a végső döntést."""
    retrieved_text, search_results = search_chroma(user_query, top_k=5)

    additional_context = []
    for i, doc in enumerate(search_results["metadatas"][0]):
        additional_context.append(f"{doc['cím']} - {doc['leírás']}")

    prompt = f"""
    Az alábbi kérdést kaptuk a felhasználótól:
    "{user_query}"

    Egy háromlépéses roadmap segítségével már feltártuk a releváns információkat. 
    Ezek voltak az egyes lépések és a válaszaik:

    {context_history}

    Továbbá, az eredeti kérdéshez kapcsolódó legfontosabb információk az alábbiak voltak:

    {"\n".join(additional_context)}

    Most kérlek, hogy a fentiek alapján add meg a végső választ a felhasználónak!
    A válasz legyen tömör, lényegre törő, maximum 3 mondat.
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    final_response = response.choices[0].message.content
    context_list = additional_context + [context_history]

    return context_list, final_response
