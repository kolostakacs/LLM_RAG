import pandas as pd
import json

df = pd.read_excel("C:/Users/device/Desktop/work/Projects/Hitelek_master-2.xlsx", header=None)

chunks = [{"chunk_id": i, "cím": row[0], "leírás": row[1]} for i, row in df.iterrows()]

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=4)

print("hello")