import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from chunking import chunks
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.rate_limiters import InMemoryRateLimiter
from langchain_neo4j import LLMGraphTransformer, Neo4jGraph
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)

WORKERS = 4

MODELO = "gemma-4-31b-it"
IGNORAR_FALHAS = False  # True = não tenta de novo os chunks que já falharam
REQUISICOES_POR_MINUTO = 10
ROOT = Path(__file__).resolve().parents[1]
FALHAS = ROOT / "ingestion" / "checkpoint_falhas.jsonl"
FEITOS = ROOT / "ingestion" / "checkpoint_feitos.jsonl"
VAZIOS = ROOT / "ingestion" / "checkpoint_vazios.jsonl"

rate_limiter = InMemoryRateLimiter(
    requests_per_second=REQUISICOES_POR_MINUTO / 60,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

load_dotenv()

llm = init_chat_model(
    "gemma-4-31b-it",
    model_provider="google_genai",
    temperature=0,
    timeout=240,
    max_retries=0,
    rate_limiter=rate_limiter,
)

# Grafo Neo4j hospedado no AuraDB
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)


def make_LLMGraphTransformer(llm):
    return LLMGraphTransformer(
        llm=llm,
        allowed_nodes=[
            "Projeto",
            "Acao",
            "Ferramenta",
            "Publico",
            "Conceito",
            "Resultado",
            "Instituicao",
            "Local",
        ],
        allowed_relationships=[
            ("Projeto", "REALIZA", "Acao"),
            ("Projeto", "DESENVOLVE", "Ferramenta"),
            ("Projeto", "UTILIZA", "Ferramenta"),
            ("Acao", "UTILIZA", "Ferramenta"),
            ("Projeto", "VINCULADO_A", "Projeto"),
            ("Projeto", "VINCULADO_A", "Instituicao"),
            ("Instituicao", "LOCALIZADO_EM", "Local"),
            ("Acao", "LOCALIZADO_EM", "Local"),
            ("Projeto", "LOCALIZADO_EM", "Local"),
            ("Local", "LOCALIZADO_EM", "Local"),
            ("Acao", "DIRIGIDA_A", "Publico"),
            ("Projeto", "DIRIGIDA_A", "Publico"),
            ("Projeto", "ABORDA", "Conceito"),
            ("Acao", "ABORDA", "Conceito"),
            ("Ferramenta", "ABORDA", "Conceito"),
            ("Projeto", "PRODUZ", "Resultado"),
            ("Acao", "PRODUZ", "Resultado"),
            ("Ferramenta", "PRODUZ", "Resultado"),
            ("Resultado", "IMPACTA", "Publico"),
        ],
        strict_mode=True,
        node_properties=["descricao"],
        additional_instructions=(
            "Escreva em português, com nomes naturais e espaços (nunca snake_case nem prefixos como 'Acao_', 'Conceito de'). "
            "Preencha 'descricao' em TODO nó com uma frase descritiva e objetiva, baseada no texto, que resume a ideia central desse nó.\n"
            "CRITÉRIO: só crie nó com nome próprio ou termo técnico específico, que serviria de título de verbete. "
            "Se precisar de contexto para fazer sentido ('oficinas', 'alunas', 'inclusão', 'escola'), não crie. "
            "Na dúvida, não extraia.\n"
            "Projeto: iniciativa contínua com nome próprio ('Meninas.Comp'). "
            "Acao: atividade com título ou tema próprio ('Oficina de Scratch'); 'realizou oficinas' não gera nó.\n"
            "Instituicao: só executora, proponente ou de fomento (universidades, IFs, CNPq), "
            "somente com nome próprio ('UFPel', 'UFSC').\n"
            "Local: só cidades, estados, regiões ou países; nunca espaços físicos.\n"
            "DESENVOLVE: só software criado pela equipe; ferramentas prontas (Scratch, Python, Arduino) usam UTILIZA.\n"
            "Conceito: tema técnico, metodologia ou campo do conhecimento específico "
            "('Computação Desplugada', 'Robótica Educacional', 'STEM'); "
            "temas sociais só qualificados por quem e onde ('Inclusão de Meninas na Computação'); "
            "nunca palavras vagas como 'Inclusão', 'Diversidade', 'Empoderamento', 'Metodologia'. "
            "Reuse o termo exato para o mesmo conceito.\n"
            "Publico: perfil padronizado ('Alunas do Ensino Médio', 'Graduandas em Computação'). "
            "Resultado: só efeito concreto que diga o que mudou e para quem ('Redução da evasão'); "
            "nunca 'Resultados positivos'.\n"
            "Não extraia citações e autores ('Muller et al., 2025'), seções, autorreferências ('este trabalho'), "
            "métodos de pesquisa ('pesquisa quantitativa') nem itens numerados ('Figura 1', 'QP1', 'Grupo 3').\n"
            "EXEMPLO: 'O Meninas.Comp, da UFSC, realizou oficinas com alunas, com resultados positivos e maior inclusão.' "
            "-> extrair Meninas.Comp VINCULADO_A UFSC; não extrair 'oficinas', 'alunas', 'resultados positivos', 'inclusão'."
        ),
        ignore_tool_usage=False,
    )


transformer = make_LLMGraphTransformer(llm)


def add_metadata(doc):
    m = doc.metadata
    graph.query(
        """
        MERGE (a:Artigo {id: $artigo_id})
        SET a.titulo = $titulo, a.url_artigo = $url_artigo, a.url_pdf = $url_pdf
        WITH a
        MATCH (d:Document {id: $chunk_id})
        MERGE (d)-[:PARTE_DE]->(a)
        WITH a
        MERGE (e:Edicao {nome: $edicao})
        MERGE (a)-[:PUBLICADO_EM]->(e)
        WITH a
        UNWIND $autores AS nome
        MERGE (au:Autor {nome: nome})
        MERGE (a)-[:ESCRITO_POR]->(au)
        """,
        {
            "artigo_id": m["artigo_id"],
            "titulo": m["titulo"],
            "url_artigo": m["url_artigo"],
            "url_pdf": m["url_pdf"],
            "chunk_id": m["id"],
            "edicao": m["edicao"].replace("\xa0", " ").strip(),
            "autores": [a.strip() for a in m["autores"].split(",")],
        },
    )


def chunk_id(chunk) -> str:
    return chunk.metadata["id"]


def carregar(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                ids.add(json.loads(linha)["id"])
            except (json.JSONDecodeError, KeyError):
                continue  # ignora linhas corrompidas
    return ids


def anexar_linha(path: Path, linha: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(linha + "\n")


@retry(
    wait=wait_exponential_jitter(initial=15, max=120),
    stop=stop_after_attempt(5),
    reraise=True,
)
def extrair_grafo(doc):
    return transformer.convert_to_graph_documents([doc])


def extrair(doc):
    """Roda na thread: só chama o modelo, não mexe em banco nem em arquivo."""
    try:
        graph_docs = extrair_grafo(doc)
        if not graph_docs:
            raise ValueError("Erro na extração.")
        return doc, graph_docs, None
    except Exception as e:  # noqa: BLE001
        return doc, None, e


def gravar(doc, graph_docs):
    """Roda na thread principal, um chunk por vez."""
    graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
    add_metadata(doc)

    if not graph_docs[0].nodes:
        anexar_linha(VAZIOS, json.dumps({"id": chunk_id(doc)}, ensure_ascii=False))

    anexar_linha(FEITOS, json.dumps({"id": chunk_id(doc)}, ensure_ascii=False))


def main():
    feitos = carregar(FEITOS)
    falhos = carregar(FALHAS) if IGNORAR_FALHAS else set()
    pendentes = [
        c for c in chunks if chunk_id(c) not in feitos and chunk_id(c) not in falhos
    ]
    print(f"Total: {len(chunks)} | Feitos: {len(feitos)} | Pendentes: {len(pendentes)}")

    ok = 0
    ex = ThreadPoolExecutor(max_workers=WORKERS)
    try:
        futuros = [ex.submit(extrair, d) for d in pendentes]
        for n, fut in enumerate(as_completed(futuros), start=1):
            doc, graph_docs, erro = fut.result()
            cid = chunk_id(doc)
            try:
                if erro:
                    raise erro
                gravar(doc, graph_docs)
                ok += 1
            except Exception as e:  # noqa: BLE001
                anexar_linha(
                    FALHAS, json.dumps({"id": cid, "erro": str(e)}, ensure_ascii=False)
                )
                print(f"Falha no chunk {cid}: {e}", flush=True)
            print(f"[{n}/{len(pendentes)}] ok={ok} falhas={n - ok}", flush=True)
    except KeyboardInterrupt:
        ex.shutdown(wait=False, cancel_futures=True)
        print("\nInterrompido. O progresso está salvo, rode de novo para retomar.")
    else:
        ex.shutdown()
    print("Fim da execução.")

if __name__ == "__main__":
    main()