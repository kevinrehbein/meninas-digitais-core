import asyncio
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

CSV_PATH = Path(__file__).resolve().parent / "dataset_sintetico_ragas.csv"

personas = [
    Persona(
        name="Mariana",
        role_description="Pesquisadora acadêmica interessada em diversidade de gênero e projetos de extensão em Computação no Brasil.",
    ),
    Persona(
        name="Beatriz",
        role_description="Professora universitária que coordena ações do Programa Meninas Digitais e busca metodologias e resultados.",
    ),
    Persona(
        name="Camila",
        role_description="Aluna de graduação em TI interessada em eventos como o WIT e programas de incentivo feminino.",
    ),
]


def make_llm():
    # sem format="json": o Ragas já cuida do parsing estruturado
    return LangchainLLMWrapper(
        ChatOllama(
            model="gpt-oss:20b",
            temperature=0,
            num_ctx=16384,
            num_predict=4096,
            reasoning="low",
        )
    )


async def adapt_to_portuguese(llm):
    distribution = default_query_distribution(llm)  # single-hop + multi-hop
    for synth, _ in distribution:
        prompts = await synth.adapt_prompts("portuguese", llm=llm)
        synth.set_prompts(**prompts)
    return distribution


def main():
    random.seed(13)
    emb = LangchainEmbeddingsWrapper(get_embeddings())

    # UM único asyncio.run, com um LLM só para a adaptação
    llm_adapt = make_llm()
    distribution = asyncio.run(adapt_to_portuguese(llm_adapt))

    # chunks vizinhos aumentam a chance de existirem relacionamentos (multi-hop)
    validos = [c for c in chunks if len(c.page_content.strip()) > 300]
    start = random.randint(0, len(validos) - 40)
    amostra = validos[start : start + 40]

    # LLM novo para o generator (o anterior já passou por um event loop fechado)
    llm = make_llm()
    generator = TestsetGenerator(llm=llm, embedding_model=emb, persona_list=personas)

    testset = generator.generate_with_chunks(
        chunks=amostra,
        testset_size=10,
        query_distribution=distribution,
        run_config=RunConfig(timeout=300, max_retries=3, max_workers=2),
    )

    df = testset.to_pandas()
    df.to_csv(CSV_PATH, index=False)
    print(df[["user_input", "synthesizer_name"]])


if __name__ == "__main__":
    main()