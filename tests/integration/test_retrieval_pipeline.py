"""Ferramentas -> HybridRetriever / queries -> Neo4j (simulado por ScriptedGraph).

Valida o encadeamento real: parâmetros Cypher, hiperparâmetros do config e o formato
`[FONTE: título | url]` que redator e verificador esperam.
"""

import pytest

from agents.tools import (
    artigos_do_autor,
    autores_do_artigo,
    buscar_projetos_semelhantes,
    pesquisar_contexto_wit,
)
from config import MAX_RELACOES, N_ENTIDADES, SCORE_MIN, TETO_GRAU, TOOL_ERROR, TOP_K
from retrieval import queries
from retrieval.hybrid import HybridRetriever
from tests.helpers import ScriptedGraph


class FakeEmbeddings:
    def __init__(self, space_id=None):
        pass

    def embed_query(self, text):
        return [0.5, 0.5]


@pytest.fixture
def graphrag(monkeypatch) -> ScriptedGraph:
    graph = ScriptedGraph(
        {
            "chunk_embedding": [
                {"id": "c1", "texto": "Oficinas de robótica para meninas.", "titulo": "Robótica na escola",
                 "url": "https://wit.example/robotica", "score": 0.92},
                {"id": "c2", "texto": "Resultados da oficina.", "titulo": "Robótica na escola",
                 "url": "https://wit.example/robotica", "score": 0.85},
                {"id": "c3", "texto": "Mentoria entre pares.", "titulo": "Mentoria em TI",
                 "url": "https://wit.example/mentoria", "score": 0.80},
            ],
            "entity_embedding": [
                {"eid": "e1", "nome": "Robótica", "descricao": "Área da computação", "score": 0.9}
            ],
            "UNWIND $seeds": [
                {"origem": "Robótica", "rel": "APLICADA_EM", "destino": "Ensino médio",
                 "desc_origem": "Área da computação", "desc_destino": "",
                 "eid_origem": "e1", "eid_destino": "e2", "pontos": 0.7}
            ],
            "UNWIND $pares": [
                {"eo": "e1", "ed": "e2",
                "fontes": [{"titulo": "Robótica na escola", "url": "https://wit.example/robotica"}]}
            ],
        }
    )
    monkeypatch.setattr("retrieval.hybrid.SpaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr("retrieval.hybrid.get_graph", lambda: graph)
    monkeypatch.setattr(HybridRetriever, "_instance", None)
    return graph


class TestPesquisarContextoWitComGrafo:
    def test_contexto_completo_com_chunks_entidades_e_triplas(self, graphrag):
        out = pesquisar_contexto_wit.invoke({"pergunta": "robótica para meninas"})

        assert "FONTE: [Robótica na escola](https://wit.example/robotica)" in out
        assert "Oficinas de robótica para meninas." in out
        assert "- Robótica: Área da computação" in out
        assert (
            "(Robótica) -[APLICADA_EM]-> (Ensino médio) "
            "| FONTE: [Robótica na escola](https://wit.example/robotica)"
        ) in out

    def test_hiperparametros_do_config_chegam_ao_cypher(self, graphrag):
        pesquisar_contexto_wit.invoke({"pergunta": "x"})

        assert graphrag.called("chunk_embedding")[0] == {"k": TOP_K, "vec": [0.5, 0.5]}
        ent = graphrag.called("entity_embedding")[0]
        assert (ent["n"], ent["min"], ent["teto"]) == (N_ENTIDADES, SCORE_MIN, TETO_GRAU)
        assert graphrag.called("UNWIND $seeds")[0]["max"] == MAX_RELACOES

    def test_listar_artigos_devolve_um_trecho_por_artigo(self, graphrag):
        out = pesquisar_contexto_wit.invoke({"pergunta": "x", "listar_artigos": True})

        # os chunks agora vêm depois do grafo
        chunks = out.split("=== TRECHOS DE ARTIGOS (CHUNKS) ===")[1]

        assert chunks.count("FONTE: [Robótica na escola](https://wit.example/robotica)") == 1
        assert "Resultados da oficina." not in chunks
        assert "Mentoria entre pares." in chunks

    def test_instancia_do_retriever_e_reaproveitada(self, graphrag):
        pesquisar_contexto_wit.invoke({"pergunta": "a"})
        primeira = HybridRetriever._instance
        pesquisar_contexto_wit.invoke({"pergunta": "b"})
        assert primeira is not None
        assert HybridRetriever._instance is primeira
        assert len(graphrag.called("chunk_embedding")) == 2

    def test_erro_do_neo4j_vira_tool_error(self, graphrag):
        graphrag.routes["chunk_embedding"] = ConnectionError("Neo4j indisponível")
        assert pesquisar_contexto_wit.invoke({"pergunta": "x"}) == TOOL_ERROR


class TestFerramentasDeConsultaEstruturada:
    @pytest.fixture
    def graph(self, monkeypatch) -> ScriptedGraph:
        g = ScriptedGraph()
        monkeypatch.setattr(queries, "get_graph", lambda: g)
        return g

    def test_buscar_projetos_semelhantes(self, graph):
        graph.routes["count(DISTINCT e) AS aderencia"] = [
            {"titulo": "Projeto P", "url": "https://wit.example/p",
             "temas_em_comum": ["mentoria", "meninas"], "aderencia": 2}
        ]
        out = buscar_projetos_semelhantes.invoke({"palavras_chave": ["Mentoria", "Meninas na Computação"]})

        assert "FONTE: [Projeto P](https://wit.example/p)" in out
        assert "Temas em comum: mentoria, meninas (aderência: 2)" in out
        params = graph.called("count(DISTINCT e) AS aderencia")[0]
        assert params["kws"] == ["mentoria", "meninas na computacao"]

    def test_artigos_do_autor(self, graph):
        graph.routes["collect(DISTINCT {titulo"] = [
            {"autor": "Laura Jurgina",
             "artigos": [{"titulo": "Artigo 1", "url": "https://1"}, {"titulo": "Artigo 2", "url": "https://2"}]}
        ]
        out = artigos_do_autor.invoke({"nome": "Laura Jurgina"})

        assert out.count("FONTE: [") == 2
        assert "Autor(a): Laura Jurgina" in out
        assert graph.called("collect(DISTINCT {titulo")[0]["tokens"] == ["laura", "jurgina"]

    def test_artigos_do_autor_nao_encontrado(self, graph):
        graph.routes["collect(DISTINCT {titulo"] = []
        assert "Nenhum autor encontrado" in artigos_do_autor.invoke({"nome": "Ninguém"})

    def test_autores_do_artigo(self, graph):
        graph.routes["collect(au.nome) AS autores"] = [
            {"titulo": "Robótica na escola", "url": "https://r", "autores": ["Ana", "Bia"]}
        ]
        out = autores_do_artigo.invoke({"titulo": "Robótica na escola"})
        assert out == "FONTE: [Robótica na escola](https://r)\nAutores: Ana, Bia"

    def test_erro_de_consulta_vira_tool_error(self, graph):
        graph.routes["collect(au.nome) AS autores"] = RuntimeError("timeout")
        assert autores_do_artigo.invoke({"titulo": "x"}) == TOOL_ERROR
