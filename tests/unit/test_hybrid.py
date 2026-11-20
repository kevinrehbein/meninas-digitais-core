import pytest

from config import TOP_K
from retrieval.hybrid import HybridRetriever
from tests.helpers import ScriptedGraph


class FakeEmbeddings:
    def __init__(self, space_id=None) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


@pytest.fixture
def retriever(monkeypatch) -> HybridRetriever:
    graph = ScriptedGraph()
    monkeypatch.setattr("retrieval.hybrid.SpaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr("retrieval.hybrid.get_graph", lambda: graph)
    monkeypatch.setattr(HybridRetriever, "_instance", None)
    return HybridRetriever()


def rel(eo="e1", ed="e2", origem="Mentoria", destino="Meninas", tipo="APOIA", **extra):
    base = {
        "origem": origem, "rel": tipo, "destino": destino,
        "desc_origem": "", "desc_destino": "",
        "eid_origem": eo, "eid_destino": ed, "pontos": 0.5,
    }
    base.update(extra)
    return base


class TestSingleton:
    def test_get_instance_cria_uma_unica_vez(self, monkeypatch):
        built = []
        monkeypatch.setattr("retrieval.hybrid.SpaceEmbeddings", FakeEmbeddings)
        monkeypatch.setattr(
            "retrieval.hybrid.get_graph", lambda: built.append(1) or ScriptedGraph()
        )
        monkeypatch.setattr(HybridRetriever, "_instance", None)

        a = HybridRetriever.get_instance()
        b = HybridRetriever.get_instance()

        assert a is b
        assert len(built) == 1


class TestFiltrarUmPorArtigo:
    def test_mantem_o_primeiro_chunk_de_cada_artigo(self):
        chunks = [
            {"id": "1", "titulo": "A", "texto": "a1"},
            {"id": "2", "titulo": "A", "texto": "a2"},
            {"id": "3", "titulo": "B", "texto": "b1"},
        ]
        out = HybridRetriever._filtrar_um_por_artigo(chunks)
        assert [c["texto"] for c in out] == ["a1", "b1"]

    def test_sem_titulo_usa_o_id(self):
        chunks = [
            {"id": "1", "titulo": None, "texto": "x"},
            {"id": "2", "titulo": None, "texto": "y"},
            {"id": "1", "titulo": None, "texto": "repetido"},
        ]
        assert [c["texto"] for c in HybridRetriever._filtrar_um_por_artigo(chunks)] == ["x", "y"]


class TestFormatacao:
    def test_formatar_chunk(self):
        c = {"titulo": "Art", "url": "https://a", "texto": "corpo"}
        assert HybridRetriever.formatar_chunk(c) == "FONTE: [Art](https://a)\ncorpo"

    def test_formatar_chunk_sem_titulo_e_url(self):
        c = {"titulo": None, "url": None, "texto": ""}
        assert HybridRetriever.formatar_chunk(c) == "FONTE: [Sem título]()"

    def test_formatar_chunks_vazio(self):
        assert HybridRetriever._formatar_chunks([]) == ""

    def test_formatar_chunks_com_cabecalho(self):
        out = HybridRetriever._formatar_chunks([{"titulo": "A", "url": "u", "texto": "t"}])
        assert out.startswith("=== TRECHOS DE ARTIGOS (CHUNKS) ===\nFONTE: [A](u)")

    def test_relacoes_vazias(self):
        assert HybridRetriever._formatar_relacoes([], []) == ""

    def test_relacoes_com_entidades_triplas_e_fonte(self):
        nos = [
            {"eid": "e1", "nome": "Mentoria", "descricao": "Desc M"},
            {"eid": "e9", "nome": "Solta", "descricao": ""},  # sem relação: não aparece
        ]
        relacoes = [rel(desc_origem="Desc M", desc_destino="")]
        out = HybridRetriever._formatar_relacoes(nos, relacoes, {("e1", "e2"): [("Artigo X", "https://x")]})

        assert "=== ENTIDADES ===" in out
        assert "- Mentoria: Desc M" in out
        assert "- Meninas\n" in out  # sem descrição, sem ":"
        assert "Solta" not in out
        assert "(Mentoria) -[APOIA]-> (Meninas) | FONTE: [Artigo X](https://x)" in out

    def test_varias_fontes_na_mesma_tripla(self):
        out = HybridRetriever._formatar_relacoes([], [rel()], {("e1", "e2"): [("A", "u1"), ("B", "u2")]})
        assert "FONTE: [A](u1) FONTE: [B](u2)" in out

    def test_triplas_identicas_sao_deduplicadas(self):
        out = HybridRetriever._formatar_relacoes([], [rel(), rel()])
        assert out.count("(Mentoria) -[APOIA]-> (Meninas)") == 1

    def test_tripla_sem_fonte_nao_tem_marcador(self):
        out = HybridRetriever._formatar_relacoes([], [rel()])
        assert "FONTE: [" not in out

    def test_formatar_fonte(self):
        assert HybridRetriever._formatar_fonte("Art", "https://a") == "FONTE: [Art](https://a)"

    def test_formatar_fonte_com_campos_ausentes(self):
        assert HybridRetriever._formatar_fonte(None, None) == "FONTE: [Sem título]()"


class TestFontesDasRelacoes:
    def test_sem_relacoes_nao_consulta(self, retriever):
        assert retriever._fontes_das_relacoes([]) == {}
        assert retriever.graph.calls == []

    def test_artigos_que_mencionam_as_duas_pontas(self, retriever):
        retriever.graph.routes["UNWIND $pares"] = [
            {
                "eo": "e1", "ed": "e2",
                "fontes": [
                    {"titulo": "Art A", "url": "https://a"},
                    {"titulo": "Art B", "url": "https://b"},
                ],
            }
        ]
        out = retriever._fontes_das_relacoes([rel()])
        assert out == {("e1", "e2"): [("Art A", "https://a"), ("Art B", "https://b")]}
        assert retriever.graph.called("UNWIND $eids") == []  # sem fallback

    def test_fallback_usa_principal_artigo_de_cada_ponta(self, retriever):
        retriever.graph.routes["UNWIND $pares"] = [{"eo": "e1", "ed": "e2", "fontes": []}]
        retriever.graph.routes["UNWIND $eids"] = [
            {"eid": "e1", "fontes": [{"titulo": "Art A", "url": "https://a"}]},
            {"eid": "e2", "fontes": [{"titulo": "Art B", "url": "https://b"}]},
        ]
        out = retriever._fontes_das_relacoes([rel()])
        assert out == {("e1", "e2"): [("Art A", "https://a"), ("Art B", "https://b")]}


    def test_fallback_remove_fontes_repetidas(self, retriever):
        retriever.graph.routes["UNWIND $pares"] = [{"eo": "e1", "ed": "e2", "fontes": []}]
        retriever.graph.routes["UNWIND $eids"] = [
            {"eid": "e1", "fontes": [{"titulo": "Art A", "url": "https://a"}]},
            {"eid": "e2", "fontes": [{"titulo": "Art A", "url": "https://a"}]},
        ]
        assert retriever._fontes_das_relacoes([rel()])[("e1", "e2")] == [("Art A", "https://a")]


class TestBlocoGrafo:
    def test_sem_entidades_devolve_vazio_e_nao_busca_relacoes(self, retriever):
        retriever.graph.routes["entity_embedding"] = []
        assert retriever.bloco_grafo([0.1]) == ""
        assert retriever.graph.called("UNWIND $seeds") == []

    def test_com_entidades(self, retriever):
        retriever.graph.routes["entity_embedding"] = [
            {"eid": "e1", "nome": "Mentoria", "descricao": "d", "score": 0.9}
        ]
        retriever.graph.routes["UNWIND $seeds"] = [rel()]
        retriever.graph.routes["UNWIND $pares"] = [
            {"eo": "e1", "ed": "e2", "fontes": [{"titulo": "Art A", "url": "https://a"}]}
        ]

        out = retriever.bloco_grafo([0.1])

        assert "(Mentoria) -[APOIA]-> (Meninas) | FONTE: [Art A](https://a)" in out
        seeds_params = retriever.graph.called("UNWIND $seeds")[0]
        assert seeds_params["seeds"][0]["eid"] == "e1"


class TestRecuperarContexto:
    @pytest.fixture
    def com_dados(self, retriever):
        retriever.graph.routes.update(
            {
                "chunk_embedding": [
                    {"id": "c1", "texto": "texto 1", "titulo": "Art A", "url": "https://a", "score": 0.9},
                    {"id": "c2", "texto": "texto 2", "titulo": "Art A", "url": "https://a", "score": 0.8},
                ],
                "entity_embedding": [{"eid": "e1", "nome": "Mentoria", "descricao": "d", "score": 0.9}],
                "UNWIND $seeds": [rel()],
                "UNWIND $pares": [{"eo": "e1", "ed": "e2", "fontes": [{"titulo": "Art A", "url": "https://a"}]}],
            }
        )
        return retriever

    def test_junta_chunks_e_grafo(self, com_dados):
        out = com_dados.recuperar_contexto("mentoria")
        assert "=== TRECHOS DE ARTIGOS (CHUNKS) ===" in out
        assert "texto 1" in out and "texto 2" in out
        assert "=== ENTIDADES ===" in out
        assert "=== RELAÇÕES (TRIPLAS) ===" in out

    def test_embeda_a_pergunta_e_usa_top_k(self, com_dados):
        com_dados.recuperar_contexto("mentoria")
        assert com_dados.embedding_model.queries == ["mentoria"]
        params = com_dados.graph.called("chunk_embedding")[0]
        assert params["k"] == TOP_K and params["vec"] == [0.1, 0.2, 0.3]

    def test_um_por_artigo(self, com_dados):
        out = com_dados.recuperar_contexto("mentoria", um_por_artigo=True)
        assert "texto 1" in out and "texto 2" not in out

    def test_sem_entidades_so_devolve_chunks(self, com_dados):
        com_dados.graph.routes["entity_embedding"] = []
        out = com_dados.recuperar_contexto("x")
        assert "=== ENTIDADES ===" not in out
        assert "texto 1" in out

    def test_grafo_vem_antes_dos_chunks(self, com_dados):
        out = com_dados.recuperar_contexto("mentoria")
        assert out.index("=== ENTIDADES ===") < out.index("=== TRECHOS DE ARTIGOS (CHUNKS) ===")

    def test_sem_grafo_comeca_pelos_chunks(self, com_dados):
        com_dados.graph.routes["entity_embedding"] = []
        out = com_dados.recuperar_contexto("x")
        assert out.startswith("\n\n=== TRECHOS DE ARTIGOS (CHUNKS) ===")