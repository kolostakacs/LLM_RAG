import json

from LLM_handler import search_chroma, ask_chatbot

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
