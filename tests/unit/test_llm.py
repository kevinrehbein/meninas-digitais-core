import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from agents import llm as llm_module


@pytest.fixture(autouse=True)
def _limpa_cache():
    llm_module._available_models.cache_clear()
    yield
    llm_module._available_models.cache_clear()


class Schema(BaseModel):
    x: int = 0


def runnable(nome: str, falha: bool = False) -> RunnableLambda:
    def _run(_):
        if falha:
            raise RuntimeError(nome)
        return nome

    return RunnableLambda(_run)


class FakeModel:
    def __init__(self, nome: str, falha=False, erro_bind: Exception | None = None):
        self.nome, self.falha, self.erro_bind = nome, falha, erro_bind
        self.schema = None
        self.tools = None

    def _bind(self):
        if self.erro_bind:
            raise self.erro_bind
        return runnable(self.nome, self.falha)

    def with_structured_output(self, schema):
        self.schema = schema
        return self._bind()

    def bind_tools(self, tools):
        self.tools = tools
        return self._bind()


class TestAvailableModels:
    @pytest.fixture
    def cadeia(self, monkeypatch):
        monkeypatch.setattr(
            llm_module,
            "LLM_CHAIN",
            [("google_genai", "m1"), ("groq", "m2"), ("ollama", "m3")],
        )
        monkeypatch.setattr(
            llm_module,
            "PROVIDER_API_KEYS",
            {"google_genai": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"},
        )
        monkeypatch.setenv("GOOGLE_API_KEY", "k")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def test_ignora_provedor_sem_chave_e_mantem_sem_mapeamento(self, monkeypatch, cadeia):
        criados = []

        def fake_init(**kwargs):
            criados.append(kwargs)
            return kwargs["model"]

        monkeypatch.setattr(llm_module, "init_chat_model", fake_init)

        models = llm_module._available_models(0.0)

        assert models == ("m1", "m3")  # groq/m2 ignorado; ollama não exige chave
        assert all(c["temperature"] == 0.0 and c["max_retries"] == 0 for c in criados)

    def test_falha_de_init_nao_derruba(self, monkeypatch, cadeia):
        def fake_init(**kwargs):
            if kwargs["model"] == "m1":
                raise ValueError("indisponível")
            return kwargs["model"]

        monkeypatch.setattr(llm_module, "init_chat_model", fake_init)
        assert llm_module._available_models(0.0) == ("m3",)

    def test_nenhum_modelo_levanta_erro(self, monkeypatch, cadeia):
        def fake_init(**kwargs):
            raise ValueError("x")

        monkeypatch.setattr(llm_module, "init_chat_model", fake_init)
        with pytest.raises(RuntimeError, match="Nenhum LLM disponível"):
            llm_module._available_models(0.0)

    def test_resultado_e_cacheado_por_temperatura(self, monkeypatch, cadeia):
        chamadas = []
        monkeypatch.setattr(
            llm_module, "init_chat_model", lambda **kw: chamadas.append(kw) or kw["model"]
        )
        llm_module._available_models(0.0)
        llm_module._available_models(0.0)
        n_primeira = len(chamadas)
        llm_module._available_models(0.3)
        assert n_primeira == 2
        assert len(chamadas) == 4


class TestCompose:
    def _usa_modelos(self, monkeypatch, *models):
        recebidas = []
        monkeypatch.setattr(
            llm_module, "_available_models", lambda t: recebidas.append(t) or tuple(models)
        )
        return recebidas

    def test_get_llm_usa_primeiro_modelo(self, monkeypatch):
        self._usa_modelos(monkeypatch, runnable("a"), runnable("b"))
        assert llm_module.get_llm().invoke("x") == "a"

    def test_fallback_quando_o_primeiro_falha(self, monkeypatch):
        self._usa_modelos(monkeypatch, runnable("a", falha=True), runnable("b"))
        assert llm_module.get_llm().invoke("x") == "b"

    def test_temperatura_e_repassada(self, monkeypatch):
        recebidas = self._usa_modelos(monkeypatch, runnable("a"))
        llm_module.get_llm(temperature=0.3)
        assert recebidas == [0.3]

    def test_structured_pede_o_schema_ao_modelo(self, monkeypatch):
        m = FakeModel("a")
        self._usa_modelos(monkeypatch, m)
        assert llm_module.get_structured_llm(Schema).invoke("x") == "a"
        assert m.schema is Schema

    def test_tools_sao_repassadas_como_lista(self, monkeypatch):
        m = FakeModel("a")
        self._usa_modelos(monkeypatch, m)
        sentinela = object()
        llm_module.get_llm_with_tools((sentinela,))
        assert m.tools == [sentinela]

    @pytest.mark.parametrize("erro", [NotImplementedError("sem suporte"), ValueError("inválido")])
    def test_modelo_que_nao_suporta_a_configuracao_e_pulado(self, monkeypatch, erro):
        self._usa_modelos(monkeypatch, FakeModel("a", erro_bind=erro), FakeModel("b"))
        assert llm_module.get_structured_llm(Schema).invoke("x") == "b"

    def test_nenhum_modelo_suporta(self, monkeypatch):
        self._usa_modelos(monkeypatch, FakeModel("a", erro_bind=NotImplementedError()))
        with pytest.raises(RuntimeError, match="Nenhum modelo suporta"):
            llm_module.get_structured_llm(Schema)
