"""Fluxos ponta a ponta no grafo LangGraph REAL.

Só as bordas são simuladas: LLMs (`fake_llms`) e acesso ao Neo4j (`fake_retriever` /
`queries`). Roteamento, estado, ToolNode, verificador e finalizador são os de produção.
"""

import pytest
from langchain_core.messages import ToolMessage

from agents.prompts import OUT_OF_SCOPE_MESSAGE
from agents.schemas import ProjectPlan, Review, SupervisorDecision, TaskDecomposition
from agents.utils import format_final_response
from config import TOOL_ERROR
from retrieval import queries
from tests.helpers import ARTICLE_LINK, make_plan, tool_call


def supervisor(intent="qa", complexity="simples", directive="mentoria"):
    return SupervisorDecision(intent=intent, complexity=complexity, research_directive=directive)


def busca(pergunta: str, call_id: str) -> list[dict]:
    return [tool_call("pesquisar_contexto_wit", {"pergunta": pergunta}, call_id)]


def n_tool_messages(result) -> int:
    return sum(isinstance(m, ToolMessage) for m in result["messages"])


class TestFluxoFeliz:
    def test_qa_simples_aprovada_na_primeira(self, fake_llms, fake_retriever, run_graph):
        fake_llms.queue(SupervisorDecision, supervisor())
        fake_llms.tool_call_rounds.append(busca("mentoria", "c1"))
        fake_llms.writer_outputs.append(f"Mentoria é apoio. {ARTICLE_LINK}")
        fake_llms.queue(Review, Review(verdict="aprovado"))

        result = run_graph("O que é mentoria?")

        assert result["final_response"] == f"Mentoria é apoio. {ARTICLE_LINK}"
        assert result["verdict"] == "aprovado"
        assert result["iteration_count"] == 1
        assert result["warning"] is None
        assert fake_retriever.calls == [("mentoria", False)]
        assert "Corpo sobre mentoria." in fake_llms.prompts["writer"][0]
        assert "Mentoria é apoio." in fake_llms.prompts["Review"][0]
        fake_llms.assert_exhausted()

    def test_fora_de_dominio_nao_pesquisa_nem_redige(self, fake_llms, fake_retriever, run_graph):
        fake_llms.queue(SupervisorDecision, supervisor(intent="fora_dominio"))

        result = run_graph("Me dê uma receita de bolo")

        assert result["final_response"] == OUT_OF_SCOPE_MESSAGE
        assert fake_retriever.calls == []
        assert fake_llms.prompts["writer"] == []
        fake_llms.assert_exhausted()

    def test_sintese_complexa_executa_todos_os_passos(self, fake_llms, fake_retriever, run_graph):
        fake_llms.queue(SupervisorDecision, supervisor("sintese", "complexa", "robótica"))
        fake_llms.queue(
            TaskDecomposition, TaskDecomposition(steps=["robótica meninas", "desafios robótica"])
        )
        fake_llms.tool_call_rounds += [busca("robótica meninas", "c1"), busca("desafios robótica", "c2")]
        fake_llms.writer_outputs.append("Panorama completo")
        fake_llms.queue(Review, Review(verdict="aprovado"))

        result = run_graph("Panorama sobre robótica")

        assert result["plan_steps"] == ["robótica meninas", "desafios robótica"]
        assert [c[0] for c in fake_retriever.calls] == ["robótica meninas", "desafios robótica"]
        assert n_tool_messages(result) == 2
        writer_prompt = fake_llms.prompts["writer"][0]
        assert "Corpo sobre robótica meninas." in writer_prompt
        assert "Corpo sobre desafios robótica." in writer_prompt
        assert "síntese acadêmica" in writer_prompt
        assert result["final_response"] == "Panorama completo"
        fake_llms.assert_exhausted()


class TestCicloDeCorrecao:
    def test_reescrever_reaproveita_evidencias_e_injeta_alerta(
        self, fake_llms, fake_retriever, run_graph
    ):
        fake_llms.queue(SupervisorDecision, supervisor())
        fake_llms.tool_call_rounds.append(busca("mentoria", "c1"))
        fake_llms.writer_outputs += ["rascunho ruim", "rascunho bom"]
        fake_llms.queue(
            Review,
            Review(verdict="reescrever", feedback="URL inventada no segundo parágrafo"),
            Review(verdict="aprovado"),
        )

        result = run_graph("O que é mentoria?")

        assert result["final_response"] == "rascunho bom"
        assert result["iteration_count"] == 2
        assert result["warning"] is None
        assert len(fake_retriever.calls) == 1  # sem nova pesquisa
        assert "ALERTA DE REVISÃO" not in fake_llms.prompts["writer"][0]
        assert "URL inventada no segundo parágrafo" in fake_llms.prompts["writer"][1]
        fake_llms.assert_exhausted()

    def test_pesquisar_mais_faz_nova_rodada_so_com_o_que_falta(
        self, fake_llms, fake_retriever, run_graph
    ):
        fake_llms.queue(SupervisorDecision, supervisor())
        fake_llms.tool_call_rounds += [busca("mentoria", "c1"), busca("busca extra", "c2")]
        fake_llms.writer_outputs += ["incompleto", "completo"]
        fake_llms.queue(
            Review,
            Review(verdict="pesquisar_mais", feedback="faltam dados", missing_steps=["busca extra"]),
            Review(verdict="aprovado"),
        )

        result = run_graph("Pergunta")

        assert [c[0] for c in fake_retriever.calls] == ["mentoria", "busca extra"]
        assert "Passo atual (1/1): busca extra" in fake_llms.prompts["researcher"][1]
        # a segunda redação enxerga as evidências das duas rodadas
        segundo_prompt = fake_llms.prompts["writer"][1]
        assert "Corpo sobre mentoria." in segundo_prompt
        assert "Corpo sobre busca extra." in segundo_prompt
        assert result["final_response"] == "completo"
        fake_llms.assert_exhausted()

    def test_reclassificar_volta_ao_supervisor_com_o_feedback(
        self, fake_llms, fake_retriever, run_graph
    ):
        fake_llms.queue(
            SupervisorDecision,
            supervisor("qa", "simples", "mentoria"),
            supervisor("sintese", "simples", "panorama"),
        )
        fake_llms.tool_call_rounds += [busca("mentoria", "c1"), busca("panorama", "c2")]
        fake_llms.writer_outputs += ["resposta curta", "síntese longa"]
        fake_llms.queue(
            Review,
            Review(verdict="reclassificar", feedback="Escopo errado: pedia panorama"),
            Review(verdict="aprovado"),
        )

        result = run_graph("Fale sobre mentoria no WIT")

        assert result["intent"] == "sintese"
        assert result["final_response"] == "síntese longa"
        segundo_supervisor = fake_llms.prompts["SupervisorDecision"][1]
        assert "Escopo errado: pedia panorama" in segundo_supervisor
        assert "redator" in fake_llms.prompts["writer"][0]
        assert "síntese acadêmica" in fake_llms.prompts["writer"][1]
        fake_llms.assert_exhausted()

    def test_limite_de_iteracoes_entrega_melhor_rascunho_com_aviso(
        self, fake_llms, fake_retriever, run_graph, monkeypatch
    ):
        monkeypatch.setattr("agents.graph.MAX_ITERATIONS", 2)
        fake_llms.queue(SupervisorDecision, supervisor())
        fake_llms.tool_call_rounds.append(busca("mentoria", "c1"))
        fake_llms.writer_outputs += ["v1", "v2"]
        fake_llms.queue(
            Review,
            Review(verdict="reescrever", feedback="f1"),
            Review(verdict="reescrever", feedback="f2"),
        )

        result = run_graph("Pergunta")

        assert result["final_response"] == "v2"
        assert result["iteration_count"] == 2
        assert "Limite de iterações atingido" in result["warning"]
        fake_llms.assert_exhausted()


class TestPlanoDeProjeto:
    def test_porta_de_qualidade_rebaixa_plano_raso_aprovado_pelo_llm(
        self, fake_llms, fake_retriever, run_graph, monkeypatch
    ):
        monkeypatch.setattr(
            queries,
            "find_similar_projects_by_keyword",
            lambda kws, limit=10: [
                {"titulo": "Projeto P", "url": "https://wit.example/p",
                 "temas_em_comum": ["mentoria"], "aderencia": 2}
            ],
        )
        raso = make_plan(n_steps=1)
        bom = make_plan()

        fake_llms.queue(SupervisorDecision, supervisor("plano", "simples", "curso para meninas"))
        fake_llms.queue(TaskDecomposition, TaskDecomposition(steps=["projetos semelhantes", "metodologias"]))
        fake_llms.tool_call_rounds += [
            [tool_call("buscar_projetos_semelhantes", {"palavras_chave": ["mentoria"]}, "p1")],
            busca("metodologias", "p2"),
        ]
        fake_llms.queue(ProjectPlan, raso, bom)
        # o LLM "aprova" ambas; a regra em código derruba a primeira
        fake_llms.queue(Review, Review(verdict="aprovado"), Review(verdict="aprovado"))

        result = run_graph("Crie um plano de curso de programação para meninas")

        assert result["intent"] == "plano"
        assert result["iteration_count"] == 2
        assert result["verdict"] == "aprovado"
        assert result["final_response"] == bom.model_dump()
        assert "Metodologia com menos de 4 etapas." in fake_llms.prompts["ProjectPlan"][1]
        assert "[FONTE: Projeto P | https://wit.example/p]" in fake_llms.prompts["ProjectPlan"][0]
        assert format_final_response(result["final_response"]).startswith(f"# {bom.title}")
        fake_llms.assert_exhausted()


class TestResiliencia:
    def test_falha_do_banco_vira_evidencia_vazia_e_o_fluxo_continua(
        self, fake_llms, fake_retriever, run_graph
    ):
        fake_retriever.fail = True
        fake_llms.queue(SupervisorDecision, supervisor())
        fake_llms.tool_call_rounds.append(busca("mentoria", "c1"))
        fake_llms.writer_outputs.append("Não há dados suficientes.")
        fake_llms.queue(Review, Review(verdict="aprovado"))

        result = run_graph("Pergunta")

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert [m.content for m in tool_msgs] == [TOOL_ERROR]
        assert "(nenhuma evidência coletada)" in fake_llms.prompts["writer"][0]
        assert result["final_response"] == "Não há dados suficientes."

    def test_llm_sem_tool_call_aciona_busca_generica(self, fake_llms, fake_retriever, run_graph):
        fake_llms.queue(SupervisorDecision, supervisor(directive="mentoria"))
        # nenhuma rodada de tool_calls enfileirada => o pesquisador usa o fallback
        fake_llms.writer_outputs.append("ok")
        fake_llms.queue(Review, Review(verdict="aprovado"))

        run_graph("Pergunta")

        assert fake_retriever.calls == [("mentoria", False)]

    @pytest.mark.parametrize("n_calls", [1, 2])
    def test_multiplas_ferramentas_na_mesma_rodada(self, fake_llms, fake_retriever, run_graph, n_calls):
        fake_llms.queue(SupervisorDecision, supervisor())
        fake_llms.tool_call_rounds.append(
            [tool_call("pesquisar_contexto_wit", {"pergunta": f"t{i}"}, f"c{i}") for i in range(n_calls)]
        )
        fake_llms.writer_outputs.append("ok")
        fake_llms.queue(Review, Review(verdict="aprovado"))

        result = run_graph("Pergunta")

        assert n_tool_messages(result) == n_calls
        assert len(fake_retriever.calls) == n_calls
