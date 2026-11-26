import asyncio
import random
from pathlib import Path

from langchain_ollama import ChatOllama
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
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
from ragas.testset.transforms import apply_transforms, default_transforms

from ingestion.chunking import chunks
from retrieval.embedding_model import get_embeddings

SEED = 13
N_CHUNKS = 50
TESTSET_SIZE = 10

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / f"dataset_sintetico_ragas_seed{SEED}.csv"
KG_PATH = BASE_DIR / f"kg_{N_CHUNKS}_seed{SEED}.json"
KG_PARCIAL_PATH = BASE_DIR / f"kg_{N_CHUNKS}_seed{SEED}_parcial.json"

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
            model="phi4",
            temperature=0,
            num_ctx=16384,
            num_predict=4096,
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
    random.seed(SEED)
    emb = LangchainEmbeddingsWrapper(get_embeddings())
    llm = make_llm()
    distribution = asyncio.run(adapt_to_portuguese(llm))

    if KG_PATH.exists():
        print(f"Carregando grafo existente: {KG_PATH}")
        kg = KnowledgeGraph.load(str(KG_PATH))
        generator = TestsetGenerator(
            llm=llm,
            embedding_model=emb,
            knowledge_graph=kg,
            persona_list=personas,
        )
        testset = generator.generate(
            testset_size=TESTSET_SIZE,
            query_distribution=distribution,
            run_config=RUN_CONFIG,
        )
    else:
        validos = [c for c in chunks if len(c.page_content.strip()) > 300]
        amostra = random.sample(validos, min(N_CHUNKS, len(validos)))

        generator = TestsetGenerator(
            llm=llm,
            embedding_model=emb,
            persona_list=personas,
        )

        testset = generator.generate_with_chunks(
            chunks=amostra,
            testset_size=TESTSET_SIZE,
            query_distribution=distribution,
            run_config=RUN_CONFIG,
        )
        # Salva o grafo que foi construído
        generator.knowledge_graph.save(str(KG_PATH))

    df = testset.to_pandas()
    df.to_csv(CSV_PATH, index=False)
    print(f"CSV salvo em {CSV_PATH} ({len(df)} perguntas)")
    print(df[["user_input", "synthesizer_name"]])


if __name__ == "__main__":
    main()