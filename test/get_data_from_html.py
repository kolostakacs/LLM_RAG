import pandas as pd
from bs4 import BeautifulSoup

# HTML fájl betöltése
html_file = "C:/Users/device/Desktop/work/Projects/LLM_RAG/test/Loan_qa_test2.html"

with open(html_file, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# Az összes táblázatot megkeressük
tables = soup.find_all("table")

# Feltételezzük, hogy az első táblázat a megfelelő
if tables:
    table = tables[0]

    # Az első sorban keressük meg a fejléceket
    headers = [th.text.strip() for th in table.find_all("th")]

    # Oszlopok indexe
    fact_index = headers.index("fact") if "fact" in headers else None
    question_index = headers.index("question") if "question" in headers else None

    # Ha mindkét oszlop létezik
    if fact_index is not None and question_index is not None:
        data = []
        for row in table.find_all("tr")[1:]:  # Az első sor fejléc, ezért kihagyjuk
            cells = row.find_all("td")
            if len(cells) > max(fact_index, question_index):  # Ellenőrizzük, hogy van-e elég oszlop
                fact = cells[fact_index].text.strip()
                question = cells[question_index].text.strip()
                data.append({"Fact": fact, "Question": question})

        # Adatok mentése Excelbe
        df = pd.DataFrame(data)
        output_file = "loan_facts_questions.xlsx"
        df.to_excel(output_file, index=False)
        print(f"Az adatok sikeresen kimentve: {output_file}")
    else:
        print("Nem található 'Fact' vagy 'Question' oszlop!")
else:
    print("Nem található táblázat a HTML-fájlban.")
