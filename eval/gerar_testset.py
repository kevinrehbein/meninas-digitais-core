import asyncio
import logging
import random
from pathlib import Path

from langchain_ollama import ChatOllama
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import default_query_distribution

from ingestion.chunking import chunks
from retrieval.embedding_model import get_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.getLogger("ragas").setLevel(logging.INFO)

EVAL_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = EVAL_DIR / "ragas_prompts_pt"
CSV_PATH = EVAL_DIR / "dataset_sintetico_ragas.csv"

personas_pt = [
    Persona(
        name="Pesquisadora de Computação",
        role_description="Pesquisadora acadêmica interessada em diversidade de gênero e projetos de extensão na área de Computação e Tecnologia no Brasil.",
    ),
    Persona(
        name="Coordenadora de Projeto",
        role_description="Professora universitária que coordena iniciativas do Programa Meninas Digitais e busca métricas, metodologias e resultados de ações educativas.",
    ),
    Persona(
        name="Estudante de Graduação",
        role_description="Aluna de graduação em Tecnologia da Informação interessada em oportunidades, eventos como o WIT e programas de incentivo feminino na área.",
    ),
]


async def preparar_distribuicao_pt(llm):
    query_distribution = default_query_distribution(llm)

    for i, (query, _) in enumerate(query_distribution):
        pasta_sint = PROMPTS_DIR / f"sintetizador_{i}_{query.__class__.__name__}"

        if pasta_sint.exists() and any(pasta_sint.iterdir()):
            prompts = query.load_prompts(str(pasta_sint))
        else:
            pasta_sint.mkdir(parents=True, exist_ok=True)
            prompts = await query.adapt_prompts("portuguese", llm=llm)
            query.save_prompts(str(pasta_sint))

        query.set_prompts(**prompts)

    return query_distribution


def main():
    random.seed(13)
    chunks_validos = [c for c in chunks if len(c.page_content.strip()) > 300]
    docs_amostrados = random.sample(chunks_validos, 15)

    ollama_llm = ChatOllama(
        model="gpt-oss:20b",
        temperature=0.2,
        num_ctx=16384,
        num_predict=4096,
        reasoning="low",
        format="json",
    )
    llm = LangchainLLMWrapper(ollama_llm)
    emb = LangchainEmbeddingsWrapper(get_embeddings())
    emb.embed_query("ping")

    run_config = RunConfig(timeout=300, max_retries=5, max_wait=30, max_workers=4)
    query_distribution = asyncio.run(preparar_distribuicao_pt(llm))

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=emb,
        persona_list=personas_pt,
    )

    testset = generator.generate_with_chunks(
        chunks=docs_amostrados,
        testset_size=5,
        query_distribution=query_distribution,
        run_config=run_config,
        raise_exceptions=False,
    )

    df = testset.to_pandas()
    df.to_csv(CSV_PATH, index=False)
    print(f"Testset salvo em {CSV_PATH}")
    print(df[["user_input", "synthesizer_name"]])


if __name__ == "__main__":
    main()