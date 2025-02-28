import json
import time
import logging
import openpyxl
from first_RAG.LLM_handler import search_chroma, ask_chatbot

# Logolás beállítása
logging.basicConfig(
    filename="../first_RAG/test_run.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

MAX_RETRIES = 3  # Maximális próbálkozások száma
RETRY_DELAY = 5  # Másodpercek várakozás két próbálkozás között


# Excel fájl beolvasása és kérdések, válaszok kinyerése
def load_questions_and_answers(excel_file="C:/Users/device/Desktop/work/Projects/LLM_RAG/test/kerdesek_es_gtk.xlsx"):
    wb = openpyxl.load_workbook(excel_file)
    sheet = wb.active

    questions = []
    gt_answers = []

    # A 2. sortól kezdjük, mivel az első sor a fejléc
    for row in sheet.iter_rows(min_row=2, values_only=True):
        question = row[0]  # A 'Question' oszlop
        answer = row[1]  # A 'Fact' oszlop
        questions.append(question)
        gt_answers.append(answer)

    return questions, gt_answers


def run_tests(output_file="results_prompt_change3.json"):
    # Kérdések és válaszok betöltése
    questions, gt_answers = load_questions_and_answers()

    results = []

    for idx, (question, gt_answer) in enumerate(zip(questions, gt_answers)):
        logging.info(f"Kérdés feldolgozása: {question}")

        response_text = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response_text = ask_chatbot(question) # A chatbot válasza
                print(response_text)
                break  # Ha sikeres, kilépünk a retry ciklusból
            except Exception as e:
                logging.error(f"Hiba a kérdés feldolgozásakor (próbálkozás {attempt}/{MAX_RETRIES}): {str(e)}")
                time.sleep(RETRY_DELAY)  # Várakozás az újrapróbálás előtt


        if response_text is None:
            response_text = "Error: Nem sikerült választ generálni"

        retrieved_text, search_results = search_chroma(question, top_k=7)

        retrieved_context = []
        for i, doc in enumerate(search_results["metadatas"][0]):
            retrieved_context.append({
                "doc_id": f"{i:03}",
                "text": doc
            })

        results.append({
            "query_id": str(idx),
            "query": question,
            "gt_answer": gt_answer,  # A Ground truth válasz itt kerül be
            "response": response_text,
            "retrieved_context": retrieved_context
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=4)

    logging.info(f"Teszt eredmények elmentve: {output_file}")
    print(f"Teszt eredmények elmentve: {output_file}")


run_tests()

