import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_ollama import ChatOllama
from trulens.apps.app import TruApp
from trulens.core import Metric, Selector, TruSession
from trulens.core.otel.instrument import instrument
from trulens.dashboard import run_dashboard
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.langchain import Langchain

from agents.graph import graph_builder
from agents.state import initial_state
from agents.tools import CONTEXTS, RETRIEVAL, USER_QUERY
from agents.utils import format_final_response
from config import GRAPH_RECURSION_LIMIT, validate_config
from retrieval.hybrid import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PERGUNTAS_PATH = Path(__file__).resolve().parent / "perguntas.jsonl"
RESULTADOS_PATH = Path(__file__).resolve().parent / "trulens_resultados.csv"
FALHAS_PATH = RESULTADOS_PATH.with_name("trulens_falhas.csv")
APP_VERSION = "v1"
NUM_PERGUNTAS = 3
AVALIACAO_TIMEOUT_S = 3600


def carregar_perguntas(limite: int | None = None) -> list[str]:
    with PERGUNTAS_PATH.open(encoding="utf-8") as f:
        perguntas = [json.loads(l)["pergunta"] for l in f if l.strip()]
    return perguntas


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
            model="qwen3.8:27b",  # confirme com `ollama list`
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
            # Cada span de ferramenta fornece seu retorno como contexto.
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
    # session.reset_database()  # não preserva histórico anterior
    run_dashboard(session, port=8501)

    app = MeninasDigitaisApp()

    tru = TruApp(
        app,
        app_name="meninas-digitais",
        app_version=APP_VERSION,
        main_method=app.ask,
        feedbacks=criar_metricas(criar_judge()),
    )

    RESULTADOS_PATH.parent.mkdir(parents=True, exist_ok=True)

    falhas: list[dict] = []
    recording = None

    try:
        with tru as recording:
            for indice, pergunta in enumerate(perguntas, start=1):
                logger.info("Executando pergunta %d/%d", indice, len(perguntas))
                try:
                    app.ask(pergunta)
                except Exception as exc:
                    logger.exception("Falha na pergunta %d", indice)
                    falhas.append(
                        {"indice": indice, "pergunta": pergunta, "erro": repr(exc)}
                    )
    finally:
        record_ids = [r.record_id for r in getattr(recording, "records", [])]

        if record_ids:
            logger.info("Aguardando a conclusão das avaliações...")
            try:
                resultados = tru.retrieve_feedback_results(
                    record_ids=record_ids,
                    timeout=AVALIACAO_TIMEOUT_S,
                )
                resultados.to_csv(RESULTADOS_PATH, index=False, encoding="utf-8-sig")
                logger.info("Resultados salvos em %s", RESULTADOS_PATH)
                print(resultados.to_string(index=False))
            except Exception:
                logger.exception("Falha ao recuperar avaliações")
        else:
            logger.error("Nenhum registro foi produzido no TruLens.")

        if falhas:
            pd.DataFrame(falhas).to_csv(FALHAS_PATH, index=False, encoding="utf-8-sig")
            logger.warning("%d pergunta(s) falharam; ver %s", len(falhas), FALHAS_PATH)


if __name__ == "__main__":
    main()
