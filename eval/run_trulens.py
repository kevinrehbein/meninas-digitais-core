import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

for path in (str(ROOT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import json

import numpy as np
from langchain_ollama import ChatOllama
from trulens.apps.app import TruApp
from trulens.core import Metric, Selector, TruSession
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.langchain import Langchain

from src.agents.graph import graph_builder
from src.agents.state import initial_state
from agents.tools import CONTEXTS, QUERY, RETRIEVAL, USER_QUERY
from src.agents.utils import format_final_response
from src.config import GRAPH_RECURSION_LIMIT, validate_config
from src.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

PERGUNTAS_PATH = Path(__file__).resolve().parent / "perguntas.jsonl"
RESULTADOS_PATH = Path(__file__).resolve().parent / "trulens_resultados.csv"
APP_VERSION = "v1"
NUM_PERGUNTAS = 3

RETRIEVAL = SpanAttributes.SpanType.RETRIEVAL
QUERY = SpanAttributes.RETRIEVAL.QUERY_TEXT
CONTEXTS = SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS


def carregar_perguntas(limite: int | None = None) -> list[str]:
    with PERGUNTAS_PATH.open(encoding="utf-8") as f:
        perguntas = [json.loads(l)["pergunta"] for l in f if l.strip()]
    return perguntas[:limite]


class MeninasDigitaisApp:
    """Adaptador: traduz o estado do grafo em (pergunta: str) -> (resposta: str)."""

    def __init__(self) -> None:
        self.graph = graph_builder()

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "pergunta",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def ask(self, pergunta: str) -> str:

        token = USER_QUERY.set(pergunta)
        try:
            state = self.graph.invoke(
                initial_state(pergunta),
                config={"recursion_limit": GRAPH_RECURSION_LIMIT},
            )
            return format_final_response(state.get("final_response"))
        finally:
            USER_QUERY.reset(token)


# -------------------------------------------------------------------- MÉTRICAS
def criar_judge() -> Langchain:
    return Langchain(
        chain=ChatOllama(
            model="qwen3.8:27b",
            reasoning=False,
            temperature=0,
            num_ctx=24576,
            base_url="http://localhost:11434",
        )
    )

def criar_metricas(judge: Langchain) -> list[Metric]:
    """Métricas sempre ancoradas na pergunta original do registro."""
    answer_relevance = Metric(
        implementation=judge.relevance_with_cot_reasons,
        name="Answer Relevance",
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )

    context_relevance = Metric(
        implementation=judge.context_relevance_with_cot_reasons,
        name="Context Relevance",
        selectors={
            # A pergunta do registro é a pergunta original do usuário.
            "question": Selector.select_record_input(),
            # cada span de ferramenta fornece seu retorno como contexto.
            "context": Selector(
                span_type=RETRIEVAL,
                span_attribute=CONTEXTS,
                collect_list=False,
            ),
        },
        agg=np.mean,
    )

    groundedness = Metric(
        implementation=judge.groundedness_measure_with_cot_reasons,
        name="Groundedness",
        selectors={
            "source": Selector(
                span_type=RETRIEVAL,
                span_attribute=CONTEXTS,
                collect_list=True,
            ),
            "statement": Selector.select_record_output(),
        },
    )

    return [answer_relevance, context_relevance, groundedness]


# ----------------------------------------------------------------------- MAIN
def main() -> None:

    validate_config()

    perguntas = carregar_perguntas(NUM_PERGUNTAS)
    if not perguntas:
        raise RuntimeError(f"Nenhuma pergunta encontrada em {PERGUNTAS_PATH}")

    # Acorda o Space de embeddings
    HybridRetriever.get_instance().embedding_model.embed_query("ping")

    session = TruSession()
    session.reset_database()  # não preserva histórico anterior

    app = MeninasDigitaisApp()

    tru = TruApp(
        app,
        app_name="meninas-digitais",
        app_version=APP_VERSION,
        main_method=app.ask,
        feedbacks=criar_metricas(criar_judge()),
    )

    with tru as recording:
        for indice, pergunta in enumerate(perguntas, start=1):
            logger.info("Executando pergunta %d/%d", indice, len(perguntas))
            app.ask(pergunta)

    record_ids = [record.record_id for record in recording.records]
    if not record_ids:
        raise RuntimeError("A avaliação terminou sem produzir registros no TruLens.")

    logger.info("Aguardando a conclusão das avaliações...")
    resultados = tru.retrieve_feedback_results(
        record_ids=record_ids,
        timeout=360,
    )

    RESULTADOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(RESULTADOS_PATH, index=False, encoding="utf-8-sig")
    logger.info("Resultados salvos em %s", RESULTADOS_PATH)
    print(resultados.to_string(index=False))


if __name__ == "__main__":
    main()
