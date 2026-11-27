import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # raiz

import json

import numpy as np
from langchain_ollama import ChatOllama
from trulens.apps.app import TruApp
from trulens.core import Metric, Selector, TruSession
from trulens.core.otel.instrument import instrument
from trulens.dashboard import run_dashboard
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.langchain import Langchain

from src.agents.graph import graph_builder
from src.agents.state import initial_state
from src.agents.utils import format_final_response
from src.config import GRAPH_RECURSION_LIMIT, validate_config
from src.retrieval.hybrid import HybridRetriever

PERGUNTAS_PATH = Path(__file__).resolve().parent / "perguntas.jsonl"
RESULTADOS_PATH = Path(__file__).resolve().parent / "trulens_resultados.csv"
APP_VERSION = "v1"

RETRIEVAL = SpanAttributes.SpanType.RETRIEVAL
QUERY = SpanAttributes.RETRIEVAL.QUERY_TEXT
CONTEXTS = SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS


def carregar_perguntas(limite: int | None = None) -> list[str]:
    with PERGUNTAS_PATH.open(encoding="utf-8") as f:
        perguntas = [json.loads(l)["pergunta"] for l in f if l.strip()]
    return perguntas[:limite]


# ------------------------------------------------------------------ RETRIEVAL
def _attrs_context(ret, exception, *args, **kwargs) -> dict:
    """Contexto completo da chamada (relações + chunks) como UM contexto."""
    # método de instância: args[0] pode ser `self`
    pergunta = kwargs.get("pergunta") or next(
        (a for a in args if isinstance(a, str)), ""
    )
    return {QUERY: pergunta, CONTEXTS: [ret] if ret else []}


def instrument_retriever() -> None:
    """Instrumenta o retriever por fora; src/ continua sem TruLens."""
    HybridRetriever.recuperar_contexto = instrument(
        span_type=RETRIEVAL, attributes=_attrs_context
    )(HybridRetriever.recuperar_contexto)


# ------------------------------------------------------------------------ APP
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
        state = self.graph.invoke(
            initial_state(pergunta),
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        )
        return format_final_response(state.get("final_response"))


# -------------------------------------------------------------------- MÉTRICAS
judge = Langchain(
    chain=ChatOllama(
        model="qwen3.8:27b",
        reasoning=False,
        temperature=0,
        num_ctx=24576,
        base_url="http://localhost:11434",
    )
)

f_answer = Metric(
    implementation=judge.relevance_with_cot_reasons,
    name="Answer Relevance",
    selectors={
        "prompt": Selector.select_record_input(),
        "response": Selector.select_record_output(),
    },
)

f_context = Metric(
    implementation=judge.context_relevance_with_cot_reasons,
    name="Context Relevance",
    selectors={
        "question": Selector.select_record_input(),
        "context": Selector(
            span_type=RETRIEVAL,
            span_attribute=CONTEXTS,
            collect_list=False,  # um score por contexto
        ),
    },
    agg=np.mean,
)

f_ground = Metric(
    implementation=judge.groundedness_measure_with_cot_reasons,
    name="Groundedness",
    selectors={
        "source": Selector(
            span_type=RETRIEVAL,
            span_attribute=CONTEXTS,
            collect_list=True,  # todos os contextos juntos
        ),
        "statement": Selector.select_record_output(),
    },
)


# ----------------------------------------------------------------------- MAIN
def main() -> None:
    validate_config()
    # Acorda o Space de embeddings fora dos registros (não distorce latência/notas).
    HybridRetriever.get_instance().embedding_model.embed_query("ping")

    session = TruSession()
    session.reset_database()  # não preserva histórico anterior
    instrument_retriever()

    app = MeninasDigitaisApp()
    tru = TruApp(
        app,
        app_name="meninas-digitais",
        app_version=APP_VERSION,
        main_method=app.ask,
        feedbacks=[f_answer, f_context, f_ground],
    )

    with tru as recording:
        for pergunta in carregar_perguntas(3):
            app.ask(pergunta)

    ids = [r.record_id for r in recording.records]
    print("Aguardando a conclusão das avaliações...")

    res = tru.retrieve_feedback_results(record_ids=ids, timeout=360)
    display(res)
    res.to_csv(RESULTADOS_PATH, index=False)



if __name__ == "__main__":
    main()
