import pandas as pd
import json

df = pd.read_csv("dataset_sintetico_ragas_seed13.csv")
with open("perguntas.jsonl", "w", encoding="utf-8") as f:
    for i, r in df.iterrows():
        f.write(json.dumps({"id": f"q{i+1:03d}", "pergunta": r.user_input, "reference": r.reference}, ensure_ascii=False) + "\n")