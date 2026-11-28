import pandas as pd

df = pd.read_csv("trulens_resultados.csv")

metricas = ["Answer Relevance", "Context Relevance", "Groundedness"]

# Tabela principal (NaN é ignorado automaticamente)
tabela = pd.DataFrame({
    "n_validos": df[metricas].count(),
    "media": df[metricas].mean(),
    "desvio": df[metricas].std(),
    "mediana": df[metricas].median(),
    "n_nota_1": (df[metricas] == 1.0).sum(),
}).round(3)
tabela["pct_nota_1"] = (tabela["n_nota_1"] / tabela["n_validos"] * 100).round(1)
print(tabela)

# Correlações entre as métricas
print("\nCorrelações:\n", df[metricas].corr().round(2))

# Perguntas com Answer Relevance zero
zeros = df[df["Answer Relevance"] == 0]
print(f"\nAnswer Relevance = 0: {len(zeros)} perguntas")

# Perguntas sem nota (falha da avaliação ou recusa)
print("\nSem nota:", df.loc[df[metricas].isna().any(axis=1), "input"].str[:80].tolist())

# 5 piores em Groundedness
piores = df.nsmallest(5, "Groundedness")[["input", *metricas]]
piores["input"] = piores["input"].str[:80]
print("\nPiores Groundedness:\n", piores.to_string())

# Exportar a tabela para usar no TCC
tabela.to_csv("trulens_resumo.csv")