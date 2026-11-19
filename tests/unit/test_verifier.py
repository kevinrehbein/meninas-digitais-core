
from agents.schemas import Review
from agents.verifier import quality_issues, verificador_node
from config import MAX_PLAN_STEPS
from tests.helpers import make_plan_dict, make_state


class TestQualityIssues:
    def test_rascunho_texto_nao_tem_problemas(self):
        assert quality_issues("qualquer texto") == []

    def test_plano_bom(self):
        assert quality_issues(make_plan_dict()) == []

    def test_poucas_etapas(self):
        assert "Metodologia com menos de 4 etapas." in quality_issues(make_plan_dict(n_steps=3))

    def test_etapas_rasas(self):
        issues = quality_issues(make_plan_dict(desc_len=10))
        assert any("Etapas sem detalhe" in i and "Etapa 1" in i for i in issues)

    def test_justificativa_curta(self):
        assert "Justificativa curta demais." in quality_issues(make_plan_dict(justification_len=10))

    def test_acumula_problemas(self):
        issues = quality_issues(make_plan_dict(n_steps=1, desc_len=5, justification_len=5))
        assert len(issues) == 3

    def test_limites_exatos(self):
        plan = make_plan_dict(desc_len=80)
        plan["justification"] = "j" * 300
        assert quality_issues(plan) == []


def _state(draft, **extra):
    return make_state("Pergunta?", draft=draft, **extra)


class TestVerificadorNode:
    def test_rascunho_vazio_pede_reescrita_sem_chamar_llm(self, fake_llms):
        out = verificador_node(_state(None))
        assert out["verdict"] == "reescrever"
        assert out["iteration_count"] == 1
        assert "vazio" in out["feedback"]
        assert fake_llms.prompts["Review"] == []

    def test_aprovado(self, fake_llms):
        fake_llms.queue(Review, Review(verdict="aprovado"))
        out = verificador_node(_state("Resposta boa"))
        assert out == {"iteration_count": 1, "verdict": "aprovado", "feedback": None}

    def test_contador_de_iteracoes_incrementa(self, fake_llms):
        fake_llms.queue(Review, Review(verdict="aprovado"))
        assert verificador_node(_state("x", iteration_count=2))["iteration_count"] == 3

    def test_plano_aprovado_pelo_llm_mas_raso_e_rebaixado(self, fake_llms):
        fake_llms.queue(Review, Review(verdict="aprovado"))
        out = verificador_node(_state(make_plan_dict(n_steps=2)))
        assert out["verdict"] == "reescrever"
        assert "Metodologia com menos de 4 etapas." in out["feedback"]

    def test_plano_bom_aprovado(self, fake_llms):
        fake_llms.queue(Review, Review(verdict="aprovado"))
        assert verificador_node(_state(make_plan_dict()))["verdict"] == "aprovado"

    def test_reescrever_propaga_feedback(self, fake_llms):
        fake_llms.queue(Review, Review(verdict="reescrever", feedback="URL inventada"))
        out = verificador_node(_state("texto"))
        assert out["verdict"] == "reescrever"
        assert out["feedback"] == "URL inventada"
        assert "plan_steps" not in out

    def test_reprovado_sem_feedback_recebe_mensagem_padrao(self, fake_llms):
        fake_llms.queue(Review, Review(verdict="reescrever"))
        assert "Reprovado sem detalhes" in verificador_node(_state("texto"))["feedback"]

    def test_pesquisar_mais_substitui_passos(self, fake_llms):
        fake_llms.queue(
            Review,
            Review(verdict="pesquisar_mais", feedback="faltam dados", missing_steps=["busca X"]),
        )
        out = verificador_node(_state("texto", plan_steps=["antigo"], step_index=1))
        assert out["verdict"] == "pesquisar_mais"
        assert out["plan_steps"] == ["busca X"]
        assert out["step_index"] == 0

    def test_pesquisar_mais_sem_buscas_vira_reescrever(self, fake_llms):
        fake_llms.queue(
            Review, Review(verdict="pesquisar_mais", feedback="f", missing_steps=["  ", ""])
        )
        out = verificador_node(_state("texto"))
        assert out["verdict"] == "reescrever"
        assert "plan_steps" not in out

    def test_missing_steps_limpos_e_limitados(self, fake_llms):
        steps = [f"  busca {i}  " for i in range(MAX_PLAN_STEPS + 3)]
        fake_llms.queue(Review, Review(verdict="pesquisar_mais", feedback="f", missing_steps=steps))
        out = verificador_node(_state("texto"))
        assert len(out["plan_steps"]) == MAX_PLAN_STEPS
        assert out["plan_steps"][0] == "busca 0"

    def test_prompt_contem_passos_pergunta_e_rascunho(self, fake_llms):
        fake_llms.queue(Review, Review(verdict="aprovado"))
        verificador_node(_state("RASCUNHO-XYZ", plan_steps=["passo a", "passo b"]))
        prompt = fake_llms.prompts["Review"][0]
        assert "1. passo a" in prompt and "2. passo b" in prompt
        assert "RASCUNHO-XYZ" in prompt
        assert "Pergunta?" in prompt

    def test_prompt_sem_passos(self, fake_llms):
        fake_llms.queue(Review, Review(verdict="aprovado"))
        verificador_node(_state("x"))
        assert "(nenhum)" in fake_llms.prompts["Review"][0]

    def test_rascunho_dict_e_serializado_como_json(self, fake_llms):
        fake_llms.queue(Review, Review(verdict="aprovado"))
        verificador_node(_state(make_plan_dict()))
        assert '"methodology"' in fake_llms.prompts["Review"][0]
