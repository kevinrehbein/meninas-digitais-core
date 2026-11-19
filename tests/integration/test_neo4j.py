
import uuid
import pytest
from retrieval.connection import get_graph


@pytest.fixture
def graph():
    return get_graph()


def test_conexao_real_neo4j(graph):
    resultado = graph.query("RETURN 1 AS valor")

    assert resultado == [{"valor": 1}]


def test_criar_consultar_e_remover_no(graph):
    identificador = str(uuid.uuid4())

    try:
        criado = graph.query(
            """
            CREATE (n:MeninasDigitaisTest {id: $id})
            RETURN n.id AS id
            """,
            params={"id": identificador},
        )

        assert criado == [{"id": identificador}]

        consultado = graph.query(
            """
            MATCH (n:MeninasDigitaisTest {id: $id})
            RETURN n.id AS id
            """,
            params={"id": identificador},
        )

        assert consultado == [{"id": identificador}]

    finally:
        graph.query(
            """
            MATCH (n:MeninasDigitaisTest {id: $id})
            DETACH DELETE n
            """,
            params={"id": identificador},
        )
