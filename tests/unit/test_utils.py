import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents import utils
from agents.utils import (
    build_revision_alert,
    format_final_response,
    get_research_evidence,
    get_user_question,
    message_text,
    render_project_plan,
)
from config import TOOL_ERROR
from tests.helpers import make_plan_dict, make_state, tool_round


class TestMessageText:
    def test_conteudo_string(self):
        assert message_text(AIMessage(content="olá")) == "olá"

    def test_conteudo_em_blocos_estilo_gemini(self):
        msg = AIMessage(content=["a", {"type": "text", "text": "b"}, {"type": "image"}])
        assert message_text(msg) == "ab"


class TestGetUserQuestion:
    def test_devolve_ultima_mensagem_humana_ignorando_ferramentas(self):
        state = {
            "messages": [
                HumanMessage(content="primeira"),
                AIMessage(content="resposta"),
                HumanMessage(content="segunda"),
                ToolMessage(content="resultado", tool_call_id="1", name="t"),
            ]
        }
        assert get_user_question(state) == "segunda"

    def test_sem_mensagens(self):
        assert get_user_question({}) == ""
        assert get_user_question({"messages": [AIMessage(content="x")]}) == ""


class TestGetResearchEvidence:
    def test_sem_evidencias(self):
        state = make_state("q")
        assert get_research_evidence(state) == "(nenhuma evidência coletada)"

    def test_inclui_origem_e_texto(self):
        msgs = [HumanMessage(content="q")] + tool_round(
            "c1", "pesquisar_contexto_wit", {"pergunta": "mentoria"}, "corpo A"
        )
        out = get_research_evidence({"messages": msgs})
        assert "--- Origem: pesquisar_contexto_wit(" in out
        assert '"pergunta": "mentoria"' in out
        assert "corpo A" in out

    def test_descarta_erros_de_ferramenta(self):
        msgs = [HumanMessage(content="q")]
        msgs += tool_round("c1", "t", {}, TOOL_ERROR)
        msgs += tool_round("c2", "t", {}, "ok")
        out = get_research_evidence({"messages": msgs})
        assert TOOL_ERROR not in out
        assert "ok" in out

    def test_so_erros_equivale_a_sem_evidencia(self):
        msgs = [HumanMessage(content="q")] + tool_round("c1", "t", {}, TOOL_ERROR)
        assert get_research_evidence({"messages": msgs}) == "(nenhuma evidência coletada)"

    def test_resultados_repetidos_aparecem_uma_vez(self):
        msgs = [HumanMessage(content="q")]
        msgs += tool_round("c1", "t", {"a": 1}, "corpo repetido")
        msgs += tool_round("c2", "t", {"a": 2}, "corpo repetido")
        out = get_research_evidence({"messages": msgs})
        assert out.count("corpo repetido") == 1

    def test_considera_apenas_a_pergunta_atual(self):
        msgs = [HumanMessage(content="q1")] + tool_round("c1", "t", {}, "evidencia antiga")
        msgs += [HumanMessage(content="q2")] + tool_round("c2", "t", {}, "evidencia nova")
        out = get_research_evidence({"messages": msgs})
        assert "evidencia nova" in out
        assert "evidencia antiga" not in out

    def test_orcamento_dividido_igualmente_e_truncado(self, monkeypatch):
        monkeypatch.setattr(utils, "EVIDENCE_CHAR_LIMIT", 20)
        msgs = [HumanMessage(content="q")]
        msgs += tool_round("c1", "t", {}, "x" * 100)
        msgs += tool_round("c2", "t", {}, "y" * 100)
        out = get_research_evidence({"messages": msgs})
        assert out.count("[...truncado...]") == 2
        assert "x" * 10 in out and "x" * 11 not in out
        assert "y" * 10 in out and "y" * 11 not in out

    def test_bloco_curto_nao_e_truncado(self, monkeypatch):
        monkeypatch.setattr(utils, "EVIDENCE_CHAR_LIMIT", 1000)
        msgs = [HumanMessage(content="q")] + tool_round("c1", "t", {}, "curto")
        assert "[...truncado...]" not in get_research_evidence({"messages": msgs})


class TestBuildRevisionAlert:
    @pytest.mark.parametrize("feedback", [None, ""])
    def test_sem_feedback_nao_gera_alerta(self, feedback):
        assert build_revision_alert(feedback, "texto") == ""

    def test_alerta_com_feedback_e_artefato(self):
        out = build_revision_alert("URL errada", "plano")
        assert "ALERTA DE REVISÃO" in out
        assert "URL errada" in out
        assert "(plano)" in out


class TestRenderProjectPlan:
    def test_renderiza_secoes_e_etapas(self):
        plan = make_plan_dict()
        out = render_project_plan(plan)
        assert out.startswith(f"# {plan['title']}")
        assert "**Público-alvo:** Meninas do ensino médio" in out
        assert "## Justificativa" in out
        assert "## Metodologia" in out
        assert "1. **Etapa 1**" in out
        for titulo in ("Desafios", "Indicadores de sucesso", "Resultados esperados", "Referências"):
            assert f"## {titulo}" in out

    def test_fonte_presente_e_proposta_do_planejador(self):
        out = render_project_plan(make_plan_dict())
        assert "_(fonte: [Artigo A](https://wit.example/artigo-a))_" in out  # etapa 1
        assert "_(proposta do planejador)_" in out  # etapa 2 (fonte=None)

    def test_lista_vazia_vira_nenhuma(self):
        plan = make_plan_dict()
        plan["challenges"] = []
        assert "## Desafios\n- (nenhuma)" in render_project_plan(plan)


class TestFormatFinalResponse:
    @pytest.mark.parametrize("vazio", [None, "", {}])
    def test_vazio(self, vazio):
        assert format_final_response(vazio) == "Não foi possível gerar uma resposta."

    def test_string(self):
        assert format_final_response("texto") == "texto"

    def test_dict_usa_render(self):
        plan = make_plan_dict()
        assert format_final_response(plan) == render_project_plan(plan)
