"""Extrai as médias (e estatísticas descritivas) de ragas_por_pergunta.csv.

Uso:
    python resumo_ragas.py                       # usa ragas_por_pergunta.csv
    python resumo_ragas.py caminho/arquivo.csv   # outro arquivo
"""

import sys
from pathlib import Path

import pandas as pd

ARQUIVO = Path(sys.argv[1] if len(sys.argv) > 1 else "ragas_por_pergunta.csv")

METRICAS = [
    "faithfulness",
    "answer_relevancy",
    "context_relevance",
    "context_utilization",
    "context_recall",
]

# O CSV tem cabeçalho em dois níveis (sistema, métrica); o pandas já consome
# a linha "id" vazia, então NÃO se deve descartar nenhuma linha de dados.
df = pd.read_csv(ARQUIVO, header=[0, 1], index_col=0)
df = df.apply(pd.to_numeric, errors="coerce")
sistemas = list(dict.fromkeys(df.columns.get_level_values(0)))
print(f"{len(df)} perguntas | sistemas: {sistemas}\n")

# 1) Estatísticas por sistema e métrica -------------------------------------
linhas = []
for s in sistemas:
    for m in METRICAS:
        x = df[(s, m)].dropna()
        linhas.append(
            {
                "sistema": s,
                "metrica": m,
                "n_validos": len(x),
                "media": x.mean(),
                "desvio": x.std(),
                "mediana": x.median(),
                "n_nota_1": int((x == 1.0).sum()),
                "pct_nota_1": (x == 1.0).mean() * 100,
                "n_nota_0": int((x == 0.0).sum()),
            }
        )
estat = pd.DataFrame(linhas).round(3)
print("Estatísticas por sistema e métrica:\n", estat.to_string(index=False))

# 2) Tabela de médias lado a lado + diferença ---------------------------------
medias = df.mean().unstack(0).loc[METRICAS, sistemas]
if len(sistemas) == 2:
    a, b = sistemas  # ex.: semantico, hibrido
    medias[f"dif_pontos ({b}-{a})"] = medias[b] - medias[a]
    medias[f"var_relativa_% ({b} vs {a})"] = (medias[b] - medias[a]) / medias[a] * 100
medias = medias.round(3)
print("\nMédias:\n", medias)

# 3) Pergunta a pergunta: quem ganha, perde ou empata --------------------------
if len(sistemas) == 2:
    placar = []
    for m in METRICAS:
        d = df[(b, m)] - df[(a, m)]
        placar.append(
            {
                "metrica": m,
                f"{b}_maior": int((d > 0).sum()),
                f"{b}_menor": int((d < 0).sum()),
                "empate": int((d == 0).sum()),
                f"pct_{b}_maior": round((d > 0).mean() * 100, 1),
                f"pct_{b}_menor": round((d < 0).mean() * 100, 1),
                f"pct_empate": round((d == 0).mean() * 100, 1),
            }
        )
    placar = pd.DataFrame(placar)
    print(f"\nPlacar por pergunta ({b} vs {a}):\n", placar.to_string(index=False))

# 4) Exportar -----------------------------------------------------------------

medias.to_csv("ragas_medias.csv", encoding="utf-8-sig")

print("\nArquivos gerados: ragas_estatisticas.csv, ragas_medias.csv, ragas_placar.csv")
