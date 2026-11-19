import pytest

from retrieval import queries
from tests.helpers import FakeGraph


@pytest.fixture
def graph(monkeypatch) -> FakeGraph:
    fake = FakeGraph(rows=[{"linha": 1}])
    monkeypatch.setattr(queries, "get_graph", lambda: fake)
    return fake


class TestNormalizar:
    @pytest.mark.parametrize(
        ("entrada", "esperado"),
        [
            ("  Meninas   na Computação! ", "meninas na computacao"),
            ("Educação\xa0Básica", "educacao basica"),
            ("C++/Python", "c python"),
            ("MENTORIA", "mentoria"),
            ("", ""),
            ("!!!", ""),
        ],
    )
    def test_normalizacao(self, entrada, esperado):
        assert queries.normalizar(entrada) == esperado


class TestTokens:
    def test_minusculas_e_split(self):
        assert queries._tokens("Laura  Jurgina") == ["laura", "jurgina"]

    def test_nbsp(self):
        assert queries._tokens("Laura\xa0Jurgina") == ["laura", "jurgina"]

    def test_limita_a_seis_tokens(self):
        assert queries._tokens("a b c d e f g h") == list("abcdef")

    def test_vazio(self):
        assert queries._tokens("   ") == []


class TestFindSimilarProjectsByKeyword:
    @pytest.mark.parametrize("kws", [[], [""], ["  ", "!!!"]])
    def test_sem_palavras_validas_nao_consulta_o_banco(self, graph, kws):
        assert queries.find_similar_projects_by_keyword(kws) == []
        assert graph.calls == []

    def test_normaliza_palavras_e_repassa_limite(self, graph):
        out = queries.find_similar_projects_by_keyword(["Mentoria", " Meninas na Computação!", ""])
        assert out == [{"linha": 1}]
        (_, params), = graph.calls
        assert params == {"kws": ["mentoria", "meninas na computacao"], "limit": 10}

    def test_limite_customizado(self, graph):
        queries.find_similar_projects_by_keyword(["x"], limit=3)
        assert graph.calls[0][1]["limit"] == 3


class TestFindAuthorWorks:
    def test_nome_vazio(self, graph):
        assert queries.find_author_works("   ") == []
        assert graph.calls == []

    def test_repassa_tokens_e_limite(self, graph):
        assert queries.find_author_works("Laura Jurgina") == [{"linha": 1}]
        assert graph.calls[0][1] == {"tokens": ["laura", "jurgina"], "limit": 20}


class TestFindArticlesByTitle:
    def test_titulo_vazio(self, graph):
        assert queries.find_articles_by_title("") == []
        assert graph.calls == []

    def test_repassa_tokens_e_limite(self, graph):
        queries.find_articles_by_title("Meninas Digitais", limit=5)
        assert graph.calls[0][1] == {"tokens": ["meninas", "digitais"], "limit": 5}
