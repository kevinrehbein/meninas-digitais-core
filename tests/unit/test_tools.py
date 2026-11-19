import logging

from agents import tools
from agents.tools import (
    RESEARCHER_TOOLS,
    _fontes,
    _safe_tool,
    artigos_do_autor,
    autores_do_artigo,
    buscar_projetos_semelhantes,
    pesquisar_contexto_wit,
)
from config import TOOL_ERROR


class TestFontes:
    def test_formata_bloco_por_item(self):
        out = _fontes([("Título A", "https://a", "corpo A"), ("Título B", "https://b", "corpo B")])
        assert out == (
            "[FONTE: Título A | https://a]\ncorpo A\n\n[FONTE: Título B | https://b]\ncorpo B"
        )

    def test_titulo_e_url_ausentes(self):
        assert _fontes([(None, None, "corpo")]) == "[FONTE: Sem título | ]\ncorpo"

    def test_corpo_vazio_nao_deixa_quebra_no_fim(self):
        assert _fontes([("T", "u", "")]) == "[FONTE: T | u]"


class TestSafeTool:
    def test_devolve_resultado_normal(self):
        @_safe_tool
        def ok(x):
            return x * 2

        assert ok(2) == 4

    def test_excecao_vira_tool_error_e_e_logada(self, caplog):
        @_safe_tool
        def quebra():
            raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR, logger="agents.tools"):
            assert quebra() == TOOL_ERROR
        assert "Falha na ferramenta quebra" in caplog.text

    def test_preserva_metadados(self):
        @_safe_tool
        def minha_funcao():
            """Doc."""

        assert minha_funcao.__name__ == "minha_funcao"
        assert minha_funcao.__doc__ == "Doc."


class TestPesquisarContextoWit:
    def test_delega_ao_retriever(self, fake_retriever):
        out = pesquisar_contexto_wit.invoke({"pergunta": "mentoria"})
        assert "Corpo sobre mentoria." in out
        assert fake_retriever.calls == [("mentoria", False)]

    def test_listar_artigos_vira_um_por_artigo(self, fake_retriever):
        pesquisar_contexto_wit.invoke({"pergunta": "x", "listar_artigos": True})
        assert fake_retriever.calls == [("x", True)]

    def test_falha_do_retriever_vira_tool_error(self, fake_retriever):
        fake_retriever.fail = True
        assert pesquisar_contexto_wit.invoke({"pergunta": "x"}) == TOOL_ERROR


class TestBuscarProjetosSemelhantes:
    def test_sem_resultados(self, monkeypatch):
        monkeypatch.setattr(tools.queries, "find_similar_projects_by_keyword", lambda kws: [])
        out = buscar_projetos_semelhantes.invoke({"palavras_chave": ["x"]})
        assert "Nenhum projeto estruturalmente semelhante" in out

    def test_formata_fontes_com_temas_e_aderencia(self, monkeypatch):
        rows = [
            {"titulo": "Projeto P", "url": "https://p", "temas_em_comum": ["a", "b"], "aderencia": 2}
        ]
        monkeypatch.setattr(tools.queries, "find_similar_projects_by_keyword", lambda kws: rows)
        out = buscar_projetos_semelhantes.invoke({"palavras_chave": ["a"]})
        assert out == "[FONTE: Projeto P | https://p]\nTemas em comum: a, b (aderência: 2)"

    def test_erro_de_banco(self, monkeypatch):
        def falha(kws):
            raise ConnectionError("sem conexão")

        monkeypatch.setattr(tools.queries, "find_similar_projects_by_keyword", falha)
        assert buscar_projetos_semelhantes.invoke({"palavras_chave": ["a"]}) == TOOL_ERROR


class TestArtigosDoAutor:
    def test_nao_encontrado_sugere_sobrenome(self, monkeypatch):
        monkeypatch.setattr(tools.queries, "find_author_works", lambda nome: [])
        out = artigos_do_autor.invoke({"nome": "Fulana"})
        assert "Nenhum autor encontrado com o nome 'Fulana'" in out
        assert "sobrenome" in out

    def test_lista_um_bloco_por_artigo(self, monkeypatch):
        rows = [
            {
                "autor": "Laura Jurgina",
                "artigos": [
                    {"titulo": "Artigo 1", "url": "https://1"},
                    {"titulo": "Artigo 2", "url": "https://2"},
                ],
            }
        ]
        monkeypatch.setattr(tools.queries, "find_author_works", lambda nome: rows)
        out = artigos_do_autor.invoke({"nome": "Laura"})
        assert out.count("[FONTE:") == 2
        assert "Autor(a): Laura Jurgina" in out
        assert "[FONTE: Artigo 2 | https://2]" in out


class TestAutoresDoArtigo:
    def test_nao_encontrado(self, monkeypatch):
        monkeypatch.setattr(tools.queries, "find_articles_by_title", lambda t: [])
        out = autores_do_artigo.invoke({"titulo": "Inexistente"})
        assert "Nenhum artigo encontrado para 'Inexistente'" in out

    def test_lista_autores(self, monkeypatch):
        rows = [{"titulo": "Artigo 1", "url": "https://1", "autores": ["Ana", "Bia"]}]
        monkeypatch.setattr(tools.queries, "find_articles_by_title", lambda t: rows)
        out = autores_do_artigo.invoke({"titulo": "Artigo"})
        assert out == "[FONTE: Artigo 1 | https://1]\nAutores: Ana, Bia"


def test_conjunto_de_ferramentas_do_pesquisador():
    assert {t.name for t in RESEARCHER_TOOLS} == {
        "pesquisar_contexto_wit",
        "buscar_projetos_semelhantes",
        "artigos_do_autor",
        "autores_do_artigo",
    }
