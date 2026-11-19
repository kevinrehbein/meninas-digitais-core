import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

import config
from retrieval import connection

REQUIRED = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_DATABASE", "NEO4J_PASSWORD", "EMBEDDING_SPACE"]


class TestValidateConfig:
    @pytest.fixture
    def tudo_definido(self, monkeypatch):
        for name in REQUIRED:
            monkeypatch.setattr(config, name, "valor")

    def test_ok(self, tudo_definido):
        config.validate_config()

    @pytest.mark.parametrize("faltando", REQUIRED)
    def test_variavel_ausente_e_reportada(self, tudo_definido, monkeypatch, faltando):
        monkeypatch.setattr(config, faltando, None)
        with pytest.raises(RuntimeError, match=faltando):
            config.validate_config()

    def test_lista_todas_as_ausentes(self, monkeypatch):
        for name in REQUIRED:
            monkeypatch.setattr(config, name, "")
        with pytest.raises(RuntimeError) as exc:
            config.validate_config()
        assert all(name in str(exc.value) for name in REQUIRED)


class TestConsistenciaDaConfiguracao:
    def test_todo_provedor_da_cadeia_tem_variavel_de_chave(self):
        providers = {p for p, _ in config.LLM_CHAIN}
        assert providers <= set(config.PROVIDER_API_KEYS)

    def test_limites_positivos(self):
        assert config.MAX_ITERATIONS >= 1
        assert config.MAX_PLAN_STEPS >= 1
        assert config.MAX_TOOL_CALLS >= 1
        assert config.EVIDENCE_CHAR_LIMIT > 0

    def test_recursion_limit_comporta_todas_as_iteracoes(self):
        # cada ciclo passa por ~5 nós; o limite deve cobrir o pior caso com folga
        assert config.GRAPH_RECURSION_LIMIT >= config.MAX_ITERATIONS * 10


class TestGetGraph:
    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch):
        monkeypatch.setattr(connection, "_graph", None)

    def test_singleton_com_refresh_schema_desligado(self, monkeypatch):
        ctor = MagicMock(return_value=object())
        monkeypatch.setattr(connection, "Neo4jGraph", ctor)

        g1, g2 = connection.get_graph(), connection.get_graph()

        assert g1 is g2
        ctor.assert_called_once()
        kwargs = ctor.call_args.kwargs
        assert kwargs["refresh_schema"] is False
        assert {"url", "username", "password"} <= kwargs.keys()

    def test_thread_safe(self, monkeypatch):
        criados = []

        def lento(**kwargs):
            time.sleep(0.05)
            obj = object()
            criados.append(obj)
            return obj

        monkeypatch.setattr(connection, "Neo4jGraph", lento)
        barreira = threading.Barrier(8)

        def worker():
            barreira.wait()
            return connection.get_graph()

        with ThreadPoolExecutor(8) as pool:
            resultados = [f.result() for f in [pool.submit(worker) for _ in range(8)]]

        assert len(criados) == 1
        assert all(r is criados[0] for r in resultados)
