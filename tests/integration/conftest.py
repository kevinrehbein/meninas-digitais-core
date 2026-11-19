import pytest

from agents.graph import graph_builder
from agents.state import initial_state
from config import GRAPH_RECURSION_LIMIT


@pytest.fixture
def run_graph():
    """Compila o grafo real (LangGraph) e devolve uma função que o executa."""
    graph_builder.cache_clear()
    app = graph_builder()

    def _run(question: str) -> dict:
        return app.invoke(
            initial_state(question),
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        )

    yield _run
    graph_builder.cache_clear()
