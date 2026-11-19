import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from agents.schemas import PlanStep, Review, SupervisorDecision, TaskDecomposition
from agents.state import initial_state
from tests.helpers import make_plan


def test_initial_state():
    state = initial_state("Quem escreveu X?")
    assert state["messages"] == [HumanMessage(content="Quem escreveu X?")]
    assert state["plan_steps"] == []
    assert state["step_index"] == 0
    assert state["iteration_count"] == 0
    assert state["verdict"] is None
    assert state["feedback"] is None
    assert state["warning"] is None


class TestSupervisorDecision:
    def test_complexidade_padrao_simples(self):
        d = SupervisorDecision(intent="qa", research_directive="buscar")
        assert d.complexity == "simples"

    def test_intencao_invalida(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(intent="outra", research_directive="x")

    def test_diretiva_vazia_invalida(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(intent="qa", research_directive="")


class TestTaskDecomposition:
    def test_exige_ao_menos_um_passo(self):
        with pytest.raises(ValidationError):
            TaskDecomposition(steps=[])

    def test_valido(self):
        assert TaskDecomposition(steps=["a"]).steps == ["a"]


class TestProjectPlan:
    def test_fonte_opcional(self):
        assert PlanStep(etapa="e", descricao="d").fonte is None

    def test_exige_metodologia(self):
        with pytest.raises(ValidationError):
            make_plan(methodology=[])

    def test_model_dump_gera_dicts_aninhados(self):
        dumped = make_plan().model_dump()
        assert isinstance(dumped["methodology"][0], dict)
        assert {"etapa", "descricao", "fonte"} <= dumped["methodology"][0].keys()


class TestReview:
    def test_defaults(self):
        r = Review(verdict="aprovado")
        assert r.feedback is None
        assert r.missing_steps == []

    def test_veredito_invalido(self):
        with pytest.raises(ValidationError):
            Review(verdict="talvez")
