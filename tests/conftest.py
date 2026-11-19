"""Fixtures compartilhadas: dublês de LLM e de retriever.

Nenhum teste chama LLM, Neo4j ou Hugging Face de verdade.
"""

import sys
from collections import defaultdict
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def pytest_collection_modifyitems(items):
    for item in items:
        if "integration" in Path(str(item.fspath)).parts:
            item.add_marker(pytest.mark.integration)


def _prompt_text(prompt_value) -> str:
    return prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)


class FakeLLMs:
    """Substitui `get_structured_llm`, `get_llm` e `get_llm_with_tools` dos nós.

    - `queue(Schema, *itens)`: respostas (instâncias do schema ou Exception) em ordem.
    - `writer_outputs`: conteúdos devolvidos pelo redator (str ou lista de blocos).
    - `tool_call_rounds`: uma lista de tool_calls por execução do pesquisador
      (fila vazia => sem tool_calls, o que aciona o fallback do nó).
    - `prompts[chave]`: prompts recebidos ("writer", "researcher" ou nome do schema).
    """

    def __init__(self) -> None:
        self._structured: dict[str, list] = defaultdict(list)
        self.writer_outputs: list = []
        self.tool_call_rounds: list[list[dict]] = []
        self.prompts: dict[str, list[str]] = defaultdict(list)
        self.writer_temperatures: list[float] = []
        self.bound_tools: list[list[str]] = []

    def queue(self, schema: type, *items) -> None:
        self._structured[schema.__name__].extend(items)

    # --- substitutos das fábricas de LLM ---
    def structured(self, schema, temperature: float = 0.0):
        name = schema.__name__

        def _run(prompt_value):
            self.prompts[name].append(_prompt_text(prompt_value))
            queue = self._structured[name]
            if not queue:
                raise AssertionError(f"Nenhuma resposta enfileirada para {name}")
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        return RunnableLambda(_run)

    def llm(self, temperature: float = 0.0):
        self.writer_temperatures.append(temperature)

        def _run(prompt_value):
            self.prompts["writer"].append(_prompt_text(prompt_value))
            if not self.writer_outputs:
                raise AssertionError("Nenhuma saída enfileirada para o redator")
            return AIMessage(content=self.writer_outputs.pop(0))

        return RunnableLambda(_run)

    def with_tools(self, tools, temperature: float = 0.0):
        self.bound_tools.append([t.name for t in tools])

        def _run(prompt_value):
            self.prompts["researcher"].append(_prompt_text(prompt_value))
            calls = self.tool_call_rounds.pop(0) if self.tool_call_rounds else []
            return AIMessage(content="", tool_calls=calls)

        return RunnableLambda(_run)

    def assert_exhausted(self) -> None:
        """Falha se sobrou resposta enfileirada (o fluxo fez menos chamadas que o esperado)."""
        leftovers = {k: v for k, v in self._structured.items() if v}
        assert not leftovers, f"Respostas estruturadas não consumidas: {leftovers}"
        assert not self.writer_outputs, "Saídas do redator não consumidas"
        assert not self.tool_call_rounds, "tool_calls do pesquisador não consumidos"


@pytest.fixture
def fake_llms(monkeypatch) -> FakeLLMs:
    fake = FakeLLMs()
    monkeypatch.setattr("agents.supervisor.get_structured_llm", fake.structured)
    monkeypatch.setattr("agents.decomposer.get_structured_llm", fake.structured)
    monkeypatch.setattr("agents.planner.get_structured_llm", fake.structured)
    monkeypatch.setattr("agents.verifier.get_structured_llm", fake.structured)
    monkeypatch.setattr("agents.writer.get_llm", fake.llm)
    monkeypatch.setattr("agents.researcher.get_llm_with_tools", fake.with_tools)
    return fake


class FakeRetriever:
    """Substitui o HybridRetriever usado pela ferramenta `pesquisar_contexto_wit`."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.fail = False

    def recuperar_contexto(self, pergunta: str, um_por_artigo: bool = False) -> str:
        self.calls.append((pergunta, um_por_artigo))
        if self.fail:
            raise RuntimeError("neo4j fora do ar")
        slug = pergunta.replace(" ", "-")
        return f"[FONTE: Artigo {pergunta} | https://wit.example/{slug}]\nCorpo sobre {pergunta}."


@pytest.fixture
def fake_retriever(monkeypatch) -> FakeRetriever:
    retriever = FakeRetriever()

    class _Stub:
        @staticmethod
        def get_instance() -> FakeRetriever:
            return retriever

    monkeypatch.setattr("agents.tools.HybridRetriever", _Stub)
    return retriever
