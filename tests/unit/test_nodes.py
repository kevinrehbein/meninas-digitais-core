"""Nós do grafo (supervisor, decompositor, pesquisador, redator, planejador, finalizador)."""

from langchain_core.messages import AIMessage

from agents.decomposer import decompositor_node
from agents.finalizer import finalizar_node, fora_dominio_node
from agents.planner import planejador_node
from agents.prompts import OUT_OF_SCOPE_MESSAGE
from agents.researcher import pesquisador_node
from agents.schemas import ProjectPlan, SupervisorDecision, TaskDecomposition
from agents.supervisor import supervisor_node
from agents.writer import redator_node
from config import MAX_PLAN_STEPS, MAX_TOOL_CALLS, WRITER_TEMPERATURE
from tests.helpers import make_plan, make_state, tool_call, tool_round


class TestSupervisor:
    def test_simples_ja_define_o_unico_passo(self, fake_llms):
        fake_llms.queue(
            SupervisorDecision,
            SupervisorDecision(intent="qa", complexity="simples", research_directive="buscar X"),
        )
        out = supervisor_node(make_state("O que é X?"))
        assert out["intent"] == "qa"
        assert out["complexity"] == "simples"
        assert out["research_directive"] == "buscar X"
        assert out["plan_steps"] == ["buscar X"]
        assert out["step_index"] == 0

    def test_complexa_nao_define_passos(self, fake_llms):
        fake_llms.queue(
            SupervisorDecision,
            SupervisorDecision(intent="sintese", complexity="complexa", research_directive="d"),
        )
        out = supervisor_node(make_state())
        assert out["complexity"] == "complexa"
        assert "plan_steps" not in out

    def test_fora_de_dominio_nao_define_passos(self, fake_llms):
        fake_llms.queue(
            SupervisorDecision,
            SupervisorDecision(intent="fora_dominio", complexity="simples", research_directive="d"),
        )
        out = supervisor_node(make_state("receita de bolo"))
        assert out["intent"] == "fora_dominio"
        assert "plan_steps" not in out

    def test_plano_e_sempre_complexo(self, fake_llms):
        fake_llms.queue(
            SupervisorDecision,
            SupervisorDecision(intent="plano", complexity="simples", research_directive="d"),
        )
        out = supervisor_node(make_state())
        assert out["intent"] == "plano"
        assert out["complexity"] == "complexa"

    def test_primeira_iteracao_no_prompt(self, fake_llms):
        fake_llms.queue(
            SupervisorDecision, SupervisorDecision(intent="qa", research_directive="d")
        )
        supervisor_node(make_state("Pergunta original"))
        prompt = fake_llms.prompts["SupervisorDecision"][0]
        assert "ainda não classificada" in prompt
        assert "Nenhuma (primeira tentativa)" in prompt
        assert "Pergunta original" in prompt

    def test_reclassificacao_usa_feedback_e_intencao_anterior(self, fake_llms):
        fake_llms.queue(
            SupervisorDecision, SupervisorDecision(intent="sintese", research_directive="d")
        )
        supervisor_node(make_state(iteration_count=1, intent="qa", feedback="Escopo errado"))
        prompt = fake_llms.prompts["SupervisorDecision"][0]
        assert "Escopo errado" in prompt
        assert "Iteração atual: 1" in prompt


class TestDecompositor:
    def test_remove_duplicatas_espacos_e_limita(self, fake_llms):
        steps = ["  a  ", "a", "b", "", "   ", "c", "d", "e"]
        fake_llms.queue(TaskDecomposition, TaskDecomposition(steps=steps))
        out = decompositor_node(make_state(research_directive="dir", intent="sintese"))
        assert out["plan_steps"] == ["a", "b", "c"][:MAX_PLAN_STEPS]
        assert out["step_index"] == 0

    def test_passos_em_branco_caem_na_diretiva(self, fake_llms):
        fake_llms.queue(TaskDecomposition, TaskDecomposition(steps=["  ", ""]))
        out = decompositor_node(make_state(research_directive="minha diretiva"))
        assert out["plan_steps"] == ["minha diretiva"]

    def test_sem_diretiva_usa_a_pergunta_do_usuario(self, fake_llms):
        fake_llms.queue(TaskDecomposition, TaskDecomposition(steps=[" "]))
        out = decompositor_node(make_state("Pergunta do usuário"))
        assert out["plan_steps"] == ["Pergunta do usuário"]

    def test_prompt_inclui_feedback_e_limite(self, fake_llms):
        fake_llms.queue(TaskDecomposition, TaskDecomposition(steps=["a"]))
        decompositor_node(make_state(research_directive="d", feedback="faltou X"))
        prompt = fake_llms.prompts["TaskDecomposition"][0]
        assert "faltou X" in prompt
        assert f"NO MÁXIMO {MAX_PLAN_STEPS}" in prompt

    def test_sem_feedback_diz_nenhuma(self, fake_llms):
        fake_llms.queue(TaskDecomposition, TaskDecomposition(steps=["a"]))
        decompositor_node(make_state(research_directive="d"))
        assert "Nenhuma" in fake_llms.prompts["TaskDecomposition"][0]


class TestPesquisador:
    def test_executa_o_passo_atual_e_avanca_o_indice(self, fake_llms):
        calls = [tool_call("pesquisar_contexto_wit", {"pergunta": "b"}, "c1")]
        fake_llms.tool_call_rounds.append(calls)
        state = make_state(plan_steps=["a", "b"], step_index=1, intent="qa")

        out = pesquisador_node(state)

        assert out["step_index"] == 2
        (msg,) = out["messages"]
        assert isinstance(msg, AIMessage)
        assert msg.tool_calls[0]["name"] == "pesquisar_contexto_wit"
        assert "Passo atual (2/2): b" in fake_llms.prompts["researcher"][0]

    def test_limita_chamadas_paralelas(self, fake_llms):
        calls = [
            tool_call("pesquisar_contexto_wit", {"pergunta": str(i)}, f"c{i}")
            for i in range(MAX_TOOL_CALLS + 2)
        ]
        fake_llms.tool_call_rounds.append(calls)
        out = pesquisador_node(make_state(plan_steps=["a"], step_index=0))
        assert len(out["messages"][0].tool_calls) == MAX_TOOL_CALLS

    def test_fallback_quando_llm_nao_chama_ferramenta(self, fake_llms):
        out = pesquisador_node(make_state(plan_steps=["mentoria"], step_index=0))
        (call,) = out["messages"][0].tool_calls
        assert call["name"] == "pesquisar_contexto_wit"
        assert call["args"] == {"pergunta": "mentoria"}
        assert call["id"].startswith("call_")

    def test_ids_do_fallback_sao_unicos(self, fake_llms):
        state = make_state(plan_steps=["a"], step_index=0)
        id1 = pesquisador_node(state)["messages"][0].tool_calls[0]["id"]
        id2 = pesquisador_node(state)["messages"][0].tool_calls[0]["id"]
        assert id1 != id2

    def test_ferramentas_expostas_ao_llm(self, fake_llms):
        pesquisador_node(make_state(plan_steps=["a"], step_index=0))
        assert fake_llms.bound_tools[0] == [
            "pesquisar_contexto_wit",
            "buscar_projetos_semelhantes",
            "artigos_do_autor",
            "autores_do_artigo",
        ]


class TestRedator:
    def _state(self, **extra):
        msgs = tool_round("c1", "pesquisar_contexto_wit", {"pergunta": "x"}, "EVIDENCIA-1")
        state = make_state("Qual a pergunta?", **extra)
        state["messages"] = state["messages"] + msgs
        return state

    def test_qa_usa_instrucao_curta(self, fake_llms):
        fake_llms.writer_outputs.append("Resposta")
        out = redator_node(self._state(intent="qa"))
        assert out == {"draft": "Resposta"}
        prompt = fake_llms.prompts["writer"][0]
        assert "redator" in prompt
        assert "EVIDENCIA-1" in prompt
        assert "Qual a pergunta?" in prompt

    def test_sintese_usa_instrucao_detalhada(self, fake_llms):
        fake_llms.writer_outputs.append("Texto")
        redator_node(self._state(intent="sintese"))
        assert "síntese acadêmica" in fake_llms.prompts["writer"][0]

    def test_sem_intencao_assume_qa(self, fake_llms):
        fake_llms.writer_outputs.append("Texto")
        redator_node(self._state(intent=None))
        assert "redator" in fake_llms.prompts["writer"][0]

    def test_feedback_gera_alerta_de_revisao(self, fake_llms):
        fake_llms.writer_outputs.append("Texto")
        redator_node(self._state(intent="qa", feedback="Corrija a URL"))
        prompt = fake_llms.prompts["writer"][0]
        assert "ALERTA DE REVISÃO" in prompt and "Corrija a URL" in prompt

    def test_usa_temperatura_configurada(self, fake_llms):
        fake_llms.writer_outputs.append("Texto")
        redator_node(self._state(intent="qa"))
        assert fake_llms.writer_temperatures == [WRITER_TEMPERATURE]

    def test_resposta_em_blocos_vira_string(self, fake_llms):
        fake_llms.writer_outputs.append([{"type": "text", "text": "Olá "}, "mundo"])
        assert redator_node(self._state(intent="qa"))["draft"] == "Olá mundo"


class TestPlanejador:
    def test_devolve_plano_como_dict(self, fake_llms):
        plan = make_plan()
        fake_llms.queue(ProjectPlan, plan)
        state = make_state("Crie um plano")
        state["messages"] = state["messages"] + tool_round("c1", "t", {}, "EVIDENCIA-PLANO")

        out = planejador_node(state)

        assert out == {"draft": plan.model_dump()}
        prompt = fake_llms.prompts["ProjectPlan"][0]
        assert "EVIDENCIA-PLANO" in prompt
        assert "Crie um plano" in prompt

    def test_feedback_vira_alerta(self, fake_llms):
        fake_llms.queue(ProjectPlan, make_plan())
        planejador_node(make_state(feedback="Etapas sem detalhe"))
        prompt = fake_llms.prompts["ProjectPlan"][0]
        assert "ALERTA DE REVISÃO" in prompt and "Etapas sem detalhe" in prompt
        assert "(plano)" in prompt


class TestFinalizador:
    def test_aprovado_sem_aviso(self):
        out = finalizar_node({"draft": "texto", "verdict": "aprovado"})
        assert out == {"final_response": "texto"}

    def test_nao_aprovado_gera_aviso(self):
        out = finalizar_node({"draft": "texto", "verdict": "reescrever"})
        assert out["final_response"] == "texto"
        assert "Limite de iterações atingido" in out["warning"]

    def test_rascunho_dict_e_preservado(self):
        plan = make_plan().model_dump()
        assert finalizar_node({"draft": plan, "verdict": "aprovado"})["final_response"] == plan

    def test_fora_dominio(self):
        assert fora_dominio_node({}) == {"final_response": OUT_OF_SCOPE_MESSAGE}
