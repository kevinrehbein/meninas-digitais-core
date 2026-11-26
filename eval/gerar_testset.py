import asyncio
import random
from pathlib import Path

from langchain_ollama import ChatOllama
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.persona import Persona, PersonaList
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

SEED = 13
N_CHUNKS = 50
TESTSET_SIZE = 10

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / f"dataset_sintetico_ragas_seed{SEED}.csv"
KG_PATH = BASE_DIR / f"kg_{N_CHUNKS}_seed{SEED}.json"

RUN_CONFIG = RunConfig(timeout=300, max_retries=3, max_workers=2)

# se o LLM devolver um nome de persona que não existe (ex.: "Student")
_orig_getitem = PersonaList.__getitem__
_rng_persona = random.Random(SEED)


def _getitem_seguro(self, key):
    try:
        return _orig_getitem(self, key)
    except KeyError:
        return _rng_persona.choice(self.personas)


PersonaList.__getitem__ = _getitem_seguro

REGRA = " Sempre formula perguntas completas, em português do Brasil, com gramática correta, sem se referir a 'Tabela Y' ou 'figura X'."
personas = [
    Persona(
        name="Pesquisadora",
        role_description="Pesquisadora acadêmica interessada em diversidade de gênero e projetos de extensão em Computação no Brasil."
        + REGRA,
    ),
    Persona(
        name="Professora",
        role_description="Professora universitária que coordena ações do Programa Meninas Digitais e busca metodologias e resultados."
        + REGRA,
    ),
    Persona(
        name="Estudante",
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


def salvar_grafo(generator):
    """Salva o grafo se ele existir e tiver nós."""
    try:
        kg = generator.knowledge_graph
        if kg is not None and len(kg.nodes) > 0:
            kg.save(str(KG_PATH))
            print(f"Grafo salvo em {KG_PATH} ({len(kg.nodes)} nós)")
        else:
            print("Grafo vazio; nada para salvar.")
    except Exception as e:
        print(f"Não foi possível salvar o grafo: {e!r}")


def main():
    random.seed(SEED)
    emb = LangchainEmbeddingsWrapper(get_embeddings())

    llm_adapt = make_llm()
    distribution = asyncio.run(adapt_to_portuguese(llm_adapt))

    # LLM novo para o generator
    llm = make_llm()

    if KG_PATH.exists():
        # Retomada: reaproveita a extração já feita
        print(f"Carregando grafo existente: {KG_PATH}")
        kg = KnowledgeGraph.load(str(KG_PATH))
        generator = TestsetGenerator(
            llm=llm,
            embedding_model=emb,
            knowledge_graph=kg,
            persona_list=personas,
        )
        gerar = lambda: generator.generate(
            testset_size=TESTSET_SIZE,
            query_distribution=distribution,
            run_config=RUN_CONFIG,
        )
    else:
        validos = [c for c in chunks if len(c.page_content.strip()) > 300]

        # chunks aleatórios espalhados por todo o corpus
        amostra = random.sample(validos, min(N_CHUNKS, len(validos)))
        print(f"Chunks na amostra: {len(amostra)}")

        generator = TestsetGenerator(
            llm=llm, embedding_model=emb, persona_list=personas
        )
        gerar = lambda: generator.generate_with_chunks(
            chunks=amostra,
            testset_size=TESTSET_SIZE,
            query_distribution=distribution,
            run_config=RUN_CONFIG,
        )

    try:
        testset = gerar()
    except BaseException:
        # preserva a extração para retomar na próxima execução
        salvar_grafo(generator)
        raise

    # grafo salvo também em caso de sucesso
    salvar_grafo(generator)

    df = testset.to_pandas()
    df.to_csv(CSV_PATH, index=False)
    print(f"CSV salvo em {CSV_PATH} ({len(df)} perguntas)")
    print(df[["user_input", "synthesizer_name"]])


if __name__ == "__main__":
    main()