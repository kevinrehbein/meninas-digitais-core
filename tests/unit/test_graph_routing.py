import logging

import pytest

from agents import graph


class TestRouteFromSupervisor:
    def test_fora_dominio(self):
        assert graph.route_from_supervisor({"intent": "fora_dominio"}) == "fora_dominio"

    def test_complexa_vai_para_decompositor(self):
        state = {"intent": "plano", "complexity": "complexa"}
        assert graph.route_from_supervisor(state) == "decompositor"

    @pytest.mark.parametrize("complexity", ["simples", None])
    def test_simples_vai_direto_ao_pesquisador(self, complexity):
        state = {"intent": "qa", "complexity": complexity}
        assert graph.route_from_supervisor(state) == "pesquisador"


class TestRouteAfterTools:
    def test_faltam_passos(self):
        state = {"step_index": 1, "plan_steps": ["a", "b"], "intent": "qa"}
        assert graph.route_after_tools(state) == "pesquisador"

    def test_passos_concluidos_plano_vai_ao_planejador(self):
        state = {"step_index": 2, "plan_steps": ["a", "b"], "intent": "plano"}
        assert graph.route_after_tools(state) == "planejador"

    @pytest.mark.parametrize("intent", ["qa", "sintese"])
    def test_passos_concluidos_demais_vao_ao_redator(self, intent):
        state = {"step_index": 1, "plan_steps": ["a"], "intent": intent}
        assert graph.route_after_tools(state) == "redator"


class TestRouteFromVerifier:
    @pytest.fixture(autouse=True)
    def _limite(self, monkeypatch):
        monkeypatch.setattr(graph, "MAX_ITERATIONS", 3)

    @pytest.mark.parametrize(
        ("verdict", "intent", "esperado"),
        [
            ("aprovado", "qa", "finalizar"),
            ("aprovado", "plano", "finalizar"),
            ("reclassificar", "qa", "supervisor"),
            ("pesquisar_mais", "qa", "pesquisador"),
            ("pesquisar_mais", "plano", "pesquisador"),
            ("reescrever", "qa", "redator"),
            ("reescrever", "sintese", "redator"),
            ("reescrever", "plano", "planejador"),
            (None, "qa", "redator"),
        ],
    )
    def test_decisoes(self, verdict, intent, esperado):
        state = {"verdict": verdict, "intent": intent, "iteration_count": 1}
        assert graph.route_from_verifier(state) == esperado

    @pytest.mark.parametrize("verdict", ["reescrever", "pesquisar_mais", "reclassificar"])
    def test_limite_de_iteracoes_finaliza_com_aviso(self, verdict, caplog):
        state = {"verdict": verdict, "intent": "qa", "iteration_count": 3}
        with caplog.at_level(logging.WARNING, logger="agents.graph"):
            assert graph.route_from_verifier(state) == "finalizar"
        assert "Limite de iterações" in caplog.text

    def test_aprovado_no_limite_nao_gera_aviso(self, caplog):
        state = {"verdict": "aprovado", "iteration_count": 3}
        with caplog.at_level(logging.WARNING, logger="agents.graph"):
            assert graph.route_from_verifier(state) == "finalizar"
        assert "Limite de iterações" not in caplog.text


def test_graph_builder_compila_e_e_cacheado():
    graph.graph_builder.cache_clear()
    app = graph.graph_builder()
    assert app is graph.graph_builder()
    nodes = set(app.get_graph().nodes)
    assert {
        "supervisor", "fora_dominio", "decompositor", "pesquisador",
        "pesquisador_tools", "redator", "planejador", "verificador", "finalizar",
    } <= nodes
    graph.graph_builder.cache_clear()
