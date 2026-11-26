import asyncio
import random
from pathlib import Path

from langchain_ollama import ChatOllama
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.multi_hop.abstract import (
    MultiHopAbstractQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

from ingestion.chunking import chunks
from retrieval.embedding_model import get_embeddings

CSV_PATH = Path(__file__).resolve().parent / "dataset_sintetico_ragas.csv"

REGRA = " Sempre formula perguntas completas, em português do Brasil, com gramática correta, sem se referir a 'Tabela Y' ou 'figura X'."
personas = [
    Persona(
        name="Mariana",
        role_description="Pesquisadora acadêmica interessada em diversidade de gênero e projetos de extensão em Computação no Brasil."
        + REGRA,
    ),
    Persona(
        name="Beatriz",
        role_description="Professora universitária que coordena ações do Programa Meninas Digitais e busca metodologias e resultados."
        + REGRA,
    ),
    Persona(
        name="Camila",
        role_description="Aluna de graduação em TI interessada em eventos como o WIT e programas de incentivo feminino."
        + REGRA,
    ),
]


def make_llm():

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
    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=llm), 0.2),
        (MultiHopSpecificQuerySynthesizer(llm=llm), 0.3),
    ]
    for synth, _ in distribution:
        prompts = await synth.adapt_prompts("portuguese", llm=llm)
        synth.set_prompts(**prompts)
    return distribution


def main():
    random.seed(13)
    emb = LangchainEmbeddingsWrapper(get_embeddings())

    llm_adapt = make_llm()
    distribution = asyncio.run(adapt_to_portuguese(llm_adapt))

    validos = [c for c in chunks if len(c.page_content.strip()) > 300]

    # chunks aleatórios espalhados por todo o corpus
    N_CHUNKS = 500
    amostra = random.sample(validos, min(N_CHUNKS, len(validos)))
    print(f"Chunks na amostra: {len(amostra)}")

    # LLM novo para o generator
    llm = make_llm()
    generator = TestsetGenerator(llm=llm, embedding_model=emb, persona_list=personas)

    testset = generator.generate_with_chunks(
        chunks=amostra,
        testset_size=100,
        query_distribution=distribution,
        run_config=RunConfig(timeout=300, max_retries=3, max_workers=2),
    )

    df = testset.to_pandas()
    df.to_csv(CSV_PATH, index=False)
    print(df[["user_input", "synthesizer_name"]])


if __name__ == "__main__":
    main()
