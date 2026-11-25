import asyncio
import logging
import random
from pathlib import Path

from langchain_ollama import ChatOllama
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.prompt import PromptMixin
from ragas.run_config import RunConfig
from ragas.testset import Testset, TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.synthesizers.multi_hop.abstract import (
    MultiHopAbstractQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms import (
    CosineSimilarityBuilder,
    EmbeddingExtractor,
    OverlapScoreBuilder,
    SummaryExtractor,
    apply_transforms,
)
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor

from ingestion.chunking import chunks
from retrieval.embedding_model import get_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.getLogger("ragas").setLevel(logging.INFO)

EVAL_DIR = Path(__file__).resolve().parent
GRAFO_PATH = EVAL_DIR / "grafo_ragas_500.json"
PROMPTS_DIR = EVAL_DIR / "ragas_prompts_pt"
CSV_PATH = EVAL_DIR / "dataset_sintetico_ragas.csv"

# Amostra exata de 500 chunks substanciais
random.seed(13)
chunks_validos = [c for c in chunks if len(c.page_content.strip()) > 300]
docs_amostrados = random.sample(chunks_validos, 500)

# Configuração Local (Ollama + Space BGE-M3)
ollama_llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0.3,
    num_ctx=16384,
    num_predict=4096,
    reasoning="low",
)

generator_llm = LangchainLLMWrapper(ollama_llm)
generator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

run_config = RunConfig(
    timeout=300,
    max_retries=5,
    max_wait=30,
    max_workers=2,
)


# Constrói e salva o KnowledgeGraph
def criar_grafo() -> KnowledgeGraph:
    if GRAFO_PATH.exists():
        print(f"Carregando grafo salvo de {GRAFO_PATH}...")
        return KnowledgeGraph.load(GRAFO_PATH)

    print("Acordando HF Space de embeddings...")
    generator_embeddings.embed_query("ping")

    print("Construindo grafo a partir dos 500 chunks...")
    kg = KnowledgeGraph(
        nodes=[
            Node(
                type=NodeType.CHUNK,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
            for doc in docs_amostrados
        ]
    )

    transforms = [
        SummaryExtractor(llm=generator_llm),
        NERExtractor(llm=generator_llm),
        ThemesExtractor(llm=generator_llm),
        EmbeddingExtractor(
            embedding_model=generator_embeddings,
            property_name="summary",
            embed_property_name="embedding",
        ),
        CosineSimilarityBuilder(
            property_name="embedding",
            new_property_name="summary_similarity",
            threshold=0.7,
        ),
        OverlapScoreBuilder(
            property_name="entities",
            threshold=0.01,
        ),
        OverlapScoreBuilder(
            property_name="themes",
            new_property_name="themes_overlap",
            threshold=0.01,
        ),
    ]
    asyncio.run(adaptar_transforms_pt(transforms, generator_llm))

    apply_transforms(kg, transforms, run_config=run_config)

    kg.save(GRAFO_PATH)
    print(
        f"Grafo salvo com sucesso em {GRAFO_PATH} ({len(kg.nodes)} nós, {len(kg.relationships)} arestas)."
    )
    return kg


async def adaptar_transforms_pt(transforms, llm):
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(transforms):
        if isinstance(t, PromptMixin):
            t_dir = PROMPTS_DIR / f"transform_{i}_{t.__class__.__name__}"
            if t_dir.exists() and any(t_dir.iterdir()):
                t.load_prompts(str(t_dir))
            else:
                t_dir.mkdir(parents=True, exist_ok=True)
                prompts = await t.adapt_prompts("portuguese", llm=llm)
                t.set_prompts(**prompts)
                t.save_prompts(str(t_dir))


async def obter_distribuicao_pt(llm):
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    synthesizers = [
        SingleHopSpecificQuerySynthesizer(llm=llm),
        MultiHopSpecificQuerySynthesizer(llm=llm),
        MultiHopAbstractQuerySynthesizer(llm=llm),
    ]
    nomes = ["single_specific", "multi_specific", "multi_abstract"]

    for synth, nome in zip(synthesizers, nomes):
        s_dir = PROMPTS_DIR / nome
        if s_dir.exists() and any(s_dir.iterdir()):
            synth.load_prompts(str(s_dir))
        else:
            s_dir.mkdir(parents=True, exist_ok=True)
            prompts = await synth.adapt_prompts("portuguese", llm=llm)
            synth.set_prompts(**prompts)
            synth.save_prompts(str(s_dir))

    return [
        (synthesizers[0], 0.50),
        (synthesizers[1], 0.25),
        (synthesizers[2], 0.25),
    ]


def main():
    kg = criar_grafo()
    distribuicao = asyncio.run(obter_distribuicao_pt(generator_llm))

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
    )

    dataset = generator.generate(
        testset_size=50,
        query_distribution=distribuicao,
        run_config=run_config,
        raise_exceptions=False,
    )

    assert isinstance(dataset, Testset)

    df = dataset.to_pandas()
    df.to_csv(CSV_PATH, index=False)
    print(f"Finalizado! {len(df)} perguntas salvas em {CSV_PATH}")


if __name__ == "__main__":
    main()
