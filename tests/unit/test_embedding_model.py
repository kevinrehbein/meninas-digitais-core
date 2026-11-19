import json

import pytest

from retrieval import embedding_model as em


class FakeClient:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[tuple[str, str | None]] = []

    def predict(self, payload, api_name=None):
        self.calls.append((payload, api_name))
        resposta = self.responses.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    registro: list[float] = []
    monkeypatch.setattr(em.time, "sleep", registro.append)
    return registro


def instala_client(monkeypatch, responses) -> tuple[FakeClient, list]:
    client = FakeClient(responses)
    criacoes: list = []

    def fabrica(space_id, **kwargs):
        criacoes.append((space_id, kwargs))
        return client

    monkeypatch.setattr(em, "Client", fabrica)
    return client, criacoes


def test_embed_query(monkeypatch, sleeps):
    client, criacoes = instala_client(monkeypatch, [[[0.1, 0.2]]])
    vec = em.SpaceEmbeddings("org/space").embed_query("olá")

    assert vec == [0.1, 0.2]
    payload, api_name = client.calls[0]
    assert json.loads(payload) == ["olá"]
    assert api_name == "/embed"
    assert criacoes[0][0] == "org/space"


def test_cliente_e_reutilizado_entre_chamadas(monkeypatch, sleeps):
    _, criacoes = instala_client(monkeypatch, [[[1.0]], [[2.0]]])
    emb = em.SpaceEmbeddings("s")
    emb.embed_query("a")
    emb.embed_query("b")
    assert len(criacoes) == 1


def test_embed_documents_em_lotes_preserva_ordem(monkeypatch, sleeps):
    client, _ = instala_client(monkeypatch, [[[1], [2]], [[3], [4]], [[5]]])
    emb = em.SpaceEmbeddings("s", batch_size=2)

    out = emb.embed_documents(["a", "b", "c", "d", "e"])

    assert out == [[1], [2], [3], [4], [5]]
    assert [json.loads(p) for p, _ in client.calls] == [["a", "b"], ["c", "d"], ["e"]]


def test_embed_documents_vazio(monkeypatch, sleeps):
    client, _ = instala_client(monkeypatch, [])
    assert em.SpaceEmbeddings("s").embed_documents([]) == []
    assert client.calls == []


def test_tenta_novamente_com_espera_e_recria_o_cliente(monkeypatch, sleeps):
    _, criacoes = instala_client(monkeypatch, [RuntimeError("dormindo"), [[9.0]]])
    emb = em.SpaceEmbeddings("s", retries=3, wait=0.5)

    assert emb.embed_query("x") == [9.0]
    assert sleeps == [0.5]
    assert len(criacoes) == 2  # cliente descartado após a falha


def test_esgota_tentativas(monkeypatch, sleeps):
    instala_client(monkeypatch, [RuntimeError("falha")] * 3)
    emb = em.SpaceEmbeddings("s", retries=3, wait=1.0)

    with pytest.raises(RuntimeError, match="Space de embeddings indisponível"):
        emb.embed_query("x")
    assert sleeps == [1.0, 1.0, 1.0]


class TestGetEmbeddings:
    def test_sem_variavel_de_ambiente(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_SPACE", raising=False)
        with pytest.raises(RuntimeError, match="EMBEDDING_SPACE"):
            em.get_embeddings()

    def test_com_variavel(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_SPACE", "org/space")
        emb = em.get_embeddings()
        assert isinstance(emb, em.SpaceEmbeddings)
        assert emb.space_id == "org/space"
