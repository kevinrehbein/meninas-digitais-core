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
from ragas.testset.transforms.extractors import EmbeddingExtractor
from ragas.testset.transforms.extractors.llm_based import (
    NERExtractor,
    SummaryExtractor,
    ThemesExtractor,
)
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
)

from ingestion.chunking import chunks
from retrieval.embedding_model import get_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.getLogger("ragas").setLevel(logging.INFO)

EVAL_DIR = Path(__file__).resolve().parent
CSV_PATH = EVAL_DIR / "dataset_sintetico_ragas.csv"

personas_pt = [
    Persona(
        name="Mariana Silva",
        role_description="Pesquisadora acadêmica interessada em diversidade de gênero e projetos de extensão na área de Computação e Tecnologia no Brasil.",
    ),
    Persona(
        name="Beatriz Costa",
        role_description="Professora universitária que coordena iniciativas do Programa Meninas Digitais e busca métricas, metodologias e resultados de ações educativas.",
    ),
    Persona(
        name="Camila Oliveira",
        role_description="Aluna de graduação em Tecnologia da Informação interessada em oportunidades, eventos como o WIT e programas de incentivo feminino na área.",
    ),
]

async def build_transforms_pt(llm, emb):
    # 1. Extratores baseados em LLM (adaptados para português)
    summary_extractor = SummaryExtractor(llm=llm)
    themes_extractor = ThemesExtractor(llm=llm)
    ner_extractor = NERExtractor(llm=llm)

    for extractor in (summary_extractor, themes_extractor, ner_extractor):
        prompts = await extractor.adapt_prompts("portuguese", llm=llm)
        extractor.set_prompts(**prompts)

    # 2. Demais transforms com parâmetros padrão
    node_filter = CustomNodeFilter(llm=llm)
    embedding_extractor = EmbeddingExtractor(embedding_model=emb)
    cosine_builder = CosineSimilarityBuilder()
    overlap_builder = OverlapScoreBuilder(property_name="entities")

    return [
        summary_extractor,
        node_filter,
        embedding_extractor,
        themes_extractor,
        ner_extractor,
        cosine_builder,
        overlap_builder,
    ]

async def translate_distribuition(llm):
    distribution = default_query_distribution(llm)

    for query, _ in distribution:
        prompts = await query.adapt_prompts("portuguese", llm=llm)
        query.set_prompts(**prompts)

    return distribution

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
    query_distribution = asyncio.run(translate_distribuition(llm))
    transforms_pt = asyncio.run(build_transforms_pt(llm, emb))

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=emb,
        persona_list=personas_pt,
    )

    testset = generator.generate_with_chunks(
        chunks=docs_amostrados,
        testset_size=5,
        transforms=transforms_pt,
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