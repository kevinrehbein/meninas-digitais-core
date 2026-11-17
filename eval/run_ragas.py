"""Avaliação RAGAS em duas fases, para caber em 2x T4 (30 GB) no Kaggle.

Fase 1 (gerar):   recupera + gera respostas com o GERADOR (Mistral) e salva em
                  respostas_<sistema>.csv. Ao final, descarrega o gerador da VRAM.
Fase 2 (avaliar): lê esses CSVs e roda as métricas RAGAS com o JUIZ (Qwen),
                  salvando ragas_<sistema>.csv e a tabela comparativa.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # raiz

import pandas as pd
import requests
from langchain.chat_models import init_chat_model
from openai import AsyncOpenAI
from ragas.embeddings import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextRelevance,
    ContextUtilization,
    Faithfulness,
)

from src.agents.utils import message_text
from src.config import validate_config
from src.retrieval.hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parent

OLLAMA_URL = "http://127.0.0.1:11434"
GEN_MODEL = "mistral-small3.2:24b"
GEN_TEMP = 0.0
JUDGE_MODEL = "qwen3.8:27b"
JUDGE_TEMP = 0.0
NUM_CTX = 24576  # igual a OLLAMA_CONTEXT_LENGTH do notebook (evita recarregar o modelo)

SISTEMAS = ["semantico", "hibrido"]
METRICAS = [
    "faithfulness",
    "answer_relevancy",
    "context_relevance",
    "context_utilization",
]

PROMPT = """Você é um assistente do Programa Meninas Digitais.
Responda usando APENAS o contexto abaixo, que tem duas fontes complementares: um grafo (ENTIDADES e RELAÇÕES, quando existir) e trechos de artigos.

Pergunta: {pergunta}

Regras:
- Leia o grafo e os trechos e responda com a UNIÃO dos itens relevantes das duas fontes. Um item pode aparecer só em uma delas ou em ambas: nesse caso trate como um só e combine as informações.
- Consolidação: agrupe todas as menções referentes ao mesmo projeto, iniciativa ou conceito em um único item. Não desmembre um mesmo projeto em vários itens separados.
- Filtragem de relevância: inclua APENAS itens que respondam diretamente ao que foi perguntado.
- Após cada item, cite [FONTE: Título exato do artigo], copiando o título que vem depois de "FONTE:" (cite todos se o item vem de mais de um artigo). Nunca use nome de entidade como fonte.
- Se o contexto cobrir só parte da pergunta, responda APENAS o que foi encontrado.

{relacoes}

Trechos dos artigos:
{contexto}

Resposta:"""


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def recuperadores(r):
    return {
        "semantico": lambda vec: (r._chunks_entrada(vec), ""),
        "hibrido": lambda vec: (r._chunks_entrada(vec), r.bloco_grafo(vec)),
    }


def descarregar(modelo: str):
    """Libera a VRAM do modelo no Ollama (keep_alive=0)."""
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": modelo, "keep_alive": 0},
            timeout=60,
        )
        print(f"Modelo {modelo} descarregado da VRAM.", flush=True)
    except requests.RequestException as e:
        print(f"Não foi possível descarregar {modelo}: {e!r}", flush=True)


def carregar_perguntas(n: int):
    todas = [
        json.loads(l)
        for l in (ROOT / "perguntas.jsonl").open(encoding="utf-8")
        if l.strip()
    ]
    return todas[:n] if n else todas


# ---------------------------------------------------------------------------
# Fase 1: gerar respostas
# ---------------------------------------------------------------------------
def fase_gerar(r, emb, perguntas):
    gen_llm = init_chat_model(
        model=GEN_MODEL,
        model_provider="ollama",
        base_url=OLLAMA_URL,
        temperature=GEN_TEMP,
        num_ctx=NUM_CTX,
        num_predict=2048,
        repeat_penalty=1.1,
    )
    fmt = HybridRetriever.formatar_chunk
    print(
        f"Dataset: {len(perguntas)} pergunta(s) | gerador={GEN_MODEL} | temp={GEN_TEMP}",
        flush=True,
    )

    for nome, fn in recuperadores(r).items():
        arquivo = ROOT / f"respostas_{nome}.csv"

        processados = set()
        if arquivo.exists():
            df = pd.read_csv(arquivo, dtype={"id": str})
            df = df.dropna(subset=["response"]).drop_duplicates("id", keep="last")
            df.to_csv(arquivo, index=False)
            processados = set(df["id"])

        print(
            f"\n=== GERAR | {nome.upper()} | Já geradas: {len(processados)}/{len(perguntas)} ===",
            flush=True,
        )

        for q in perguntas:
            q_id = str(q["id"])
            if q_id in processados:
                print(f"Pulando ID {q_id} (já gerado).")
                continue

            chunks, rels = fn(emb.embed_query(q["pergunta"]))
            print(
                f"ID {q_id} | Chunks: {len(chunks)} | Rels: {bool(rels)} -> Gerando...",
                flush=True,
            )

            prompt = PROMPT.format(
                relacoes=f"{rels}\n" if rels else "",
                contexto="\n\n".join(map(fmt, chunks)),
                pergunta=q["pergunta"],
            )
            resposta = message_text(gen_llm.invoke(prompt)).strip()
            print(f"Resposta gerada: {len(resposta)} chars.", flush=True)

            if not resposta:
                print(f"ID {q_id}: resposta vazia, não será salva.", flush=True)
                continue

            contextos = ([rels] if rels else []) + [fmt(c) for c in chunks]

            linha = pd.DataFrame(
                [
                    {
                        "id": q_id,
                        "user_input": q["pergunta"],
                        "contextos": json.dumps(contextos, ensure_ascii=False),
                        "response": resposta,
                        "gerador": GEN_MODEL,
                        "temperatura": GEN_TEMP,
                    }
                ]
            )
            linha.to_csv(arquivo, mode="a", index=False, header=not arquivo.exists())
            print(f"Linha salva em {arquivo.name}", flush=True)

    descarregar(GEN_MODEL)
    print("\nFase GERAR concluída. Agora rode --fase avaliar.", flush=True)


# ---------------------------------------------------------------------------
# Fase 2: avaliar com o juiz
# ---------------------------------------------------------------------------
class RagasEmbAdapter(BaseRagasEmbedding):
    def __init__(self, lc_emb):
        super().__init__()
        self.lc_emb = lc_emb

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self.lc_emb.embed_query(text)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await self.lc_emb.aembed_query(text)


def avaliar(loop, metricas, pergunta, resposta, contextos):
    """Roda cada métrica isoladamente; uma falha não derruba as demais."""
    f, ar, cr, cu = metricas
    chamadas = {
        "faithfulness": lambda: f.ascore(
            user_input=pergunta, response=resposta, retrieved_contexts=contextos
        ),
        "answer_relevancy": lambda: ar.ascore(user_input=pergunta, response=resposta),
        "context_relevance": lambda: cr.ascore(
            user_input=pergunta, retrieved_contexts=contextos
        ),
        "context_utilization": lambda: cu.ascore(
            user_input=pergunta, response=resposta, retrieved_contexts=contextos
        ),
    }
    out = {}
    for nome, fn in chamadas.items():
        try:
            out[nome] = loop.run_until_complete(fn()).value
        except Exception as e:
            print(f"  [{nome}] falhou: {e!r}", flush=True)
            out[nome] = float("nan")
    return out


def fase_avaliar(emb):
    # Um único event loop: o AsyncOpenAI mantém conexões atreladas ao loop.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    client = AsyncOpenAI(
        base_url=f"{OLLAMA_URL}/v1",
        api_key="ollama",
        max_retries=0,
    )

    judge_llm = llm_factory(
        model=JUDGE_MODEL,
        provider="openai",
        client=client,
        temperature=JUDGE_TEMP,
        max_tokens=32768,
        reasoning_effort="none",  # desliga o thinking do juiz (Ollama)
    )

    emb.embed_query("ping")
    metricas = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=RagasEmbAdapter(emb)),
        ContextRelevance(llm=judge_llm),
        ContextUtilization(llm=judge_llm),
    ]

    res_consolidados = {}

    for nome in SISTEMAS:
        arq_resp = ROOT / f"respostas_{nome}.csv"
        arq_ragas = ROOT / f"ragas_{nome}.csv"

        if not arq_resp.exists():
            print(f"{arq_resp.name} não existe. Rode --fase gerar primeiro.")
            sys.exit(1)

        respostas = pd.read_csv(arq_resp, dtype={"id": str}).drop_duplicates(
            "id", keep="last"
        )

        processados = set()
        if arq_ragas.exists():
            df = pd.read_csv(arq_ragas, dtype={"id": str})
            df = df.drop_duplicates("id", keep="last")
            completo = df[METRICAS].notna().all(axis=1)
            df[completo].to_csv(arq_ragas, index=False)
            processados = set(df.loc[completo, "id"])

        print(
            f"\n=== AVALIAR | {nome.upper()} | Já concluídas: {len(processados)}/{len(respostas)} ===",
            flush=True,
        )

        for _, row in respostas.iterrows():
            q_id = str(row["id"])
            if q_id in processados:
                print(f"Pulando ID {q_id} (já avaliado).")
                continue

            contextos = json.loads(row["contextos"])
            print(f"ID {q_id} -> avaliando no juiz...", flush=True)
            notas = avaliar(
                loop, metricas, row["user_input"], row["response"], contextos
            )

            linha = pd.DataFrame(
                [
                    {
                        "id": q_id,
                        "user_input": row["user_input"],
                        "contextos": row["contextos"],
                        "response": row["response"],
                        **notas,
                    }
                ]
            )
            linha.to_csv(
                arq_ragas, mode="a", index=False, header=not arq_ragas.exists()
            )
            processados.add(q_id)
            print(f"Linha salva em {arq_ragas.name}", flush=True)

        if not arq_ragas.exists():
            print(f"Nenhum resultado salvo para {nome}; abortando consolidação.")
            sys.exit(1)
        res_consolidados[nome] = pd.read_csv(arq_ragas, dtype={"id": str})

    cols = METRICAS
    tabela = pd.concat(
        {k: v.set_index("id")[cols] for k, v in res_consolidados.items()}, axis=1
    ).round(3)
    tabela.to_csv(ROOT / "ragas_por_pergunta.csv")
    print("Execução finalizada com sucesso!", flush=True)

    print("\nPor pergunta:\n", tabela)
    print(
        "\nMédia:\n",
        pd.DataFrame({k: v[cols].mean() for k, v in res_consolidados.items()}).round(3),
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", required=True, choices=["gerar", "avaliar"])
    ap.add_argument("--n", type=int, default=0, help="nº de perguntas (0 = todas)")
    args = ap.parse_args()

    validate_config()
    r = HybridRetriever.get_instance()
    emb = r.embedding_model

    if args.fase == "gerar":
        fase_gerar(r, emb, carregar_perguntas(args.n))
    else:
        fase_avaliar(emb)
