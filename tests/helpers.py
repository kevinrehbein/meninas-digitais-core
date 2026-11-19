"""Construtores e dublês reutilizáveis pelos testes (sem rede, sem LLM, sem Neo4j)."""

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from agents.schemas import PlanStep, ProjectPlan
from agents.state import AgentState, initial_state

ARTICLE_URL = "https://wit.example/artigo-a"
ARTICLE_LINK = f"[Artigo A]({ARTICLE_URL})"


def make_plan(
    n_steps: int = 4,
    desc_len: int = 100,
    justification_len: int = 320,
    **overrides: Any,
) -> ProjectPlan:
    """ProjectPlan válido; por padrão passa nas regras de `quality_issues`."""
    data: dict[str, Any] = {
        "title": "Oficinas de programação para meninas",
        "target_audience": "Meninas do ensino médio",
        "justification": f"{ARTICLE_LINK} " + "j" * justification_len,
        "methodology": [
            PlanStep(
                etapa=f"Etapa {i}",
                descricao="d" * desc_len,
                fonte=ARTICLE_LINK if i % 2 else None,
            )
            for i in range(1, n_steps + 1)
        ],
        "challenges": ["Evasão"],
        "indicators": [f"Participação mínima de 70% ({ARTICLE_LINK})"],
        "expected_results": ["Meninas capacitadas"],
        "references": [ARTICLE_LINK],
    }
    data.update(overrides)
    return ProjectPlan(**data)


def make_plan_dict(**kwargs: Any) -> dict[str, Any]:
    return make_plan(**kwargs).model_dump()


def tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def tool_round(call_id: str, name: str, args: dict, content: str) -> list:
    """Um AIMessage com tool_call + o ToolMessage correspondente."""
    return [
        AIMessage(content="", tool_calls=[tool_call(name, args, call_id)]),
        ToolMessage(content=content, name=name, tool_call_id=call_id),
    ]


def make_state(question: str = "Pergunta?", **extra: Any) -> AgentState:
    state = initial_state(question)
    state.update(extra)  # type: ignore[typeddict-item]
    return state


class FakeGraph:
    """Neo4jGraph falso: devolve sempre as mesmas linhas e registra as chamadas."""

    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = rows or []
        self.calls: list[tuple[str, dict | None]] = []

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        self.calls.append((cypher, params))
        return self.rows


class ScriptedGraph:
    """Neo4jGraph falso que responde conforme um trecho presente no Cypher.

    `routes` mapeia "trecho do Cypher" -> linhas (ou uma Exception para simular falha).
    Consulta sem rota é um erro de teste: garante que o código não faz buscas inesperadas.
    """

    def __init__(self, routes: dict[str, Any] | None = None) -> None:
        self.routes: dict[str, Any] = routes or {}
        self.calls: list[tuple[str, dict | None]] = []

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        for key, rows in self.routes.items():
            if key in cypher:
                self.calls.append((key, params))
                if isinstance(rows, Exception):
                    raise rows
                return rows
        raise AssertionError(f"Consulta Cypher inesperada:\n{cypher}")

    def called(self, key: str) -> list[dict | None]:
        return [p for k, p in self.calls if k == key]