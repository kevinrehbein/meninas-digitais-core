
"""Testes de integração do retrieval contra Neo4j e Hugging Face reais.

Estes testes são somente de leitura: não criam, alteram ou removem dados no Neo4j.
"""

import pytest

from config import (
    EMBEDDING_SPACE,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)
from retrieval.embedding_model import SpaceEmbeddings
from retrieval.hybrid import HybridRetriever


@pytest.fixture
def retriever() -> HybridRetriever:
    """Cria o retriever com conexões reais."""
    required = {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME": NEO4J_USERNAME,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "NEO4J_DATABASE": NEO4J_DATABASE,
        "EMBEDDING_SPACE": EMBEDDING_SPACE,
    }

    missing = [name for name, value in required.items() if not value]

    if missing:
        pytest.skip(
            f"Configuração ausente para teste live: {', '.join(missing)}"
        )

    instance = HybridRetriever()

    # Evita as 8 tentativas de 20 segundos do padrão.
    instance.embedding_model = SpaceEmbeddings(
        EMBEDDING_SPACE,
        retries=1,
        wait=0,
    )

    return instance


def test_recuperar_contexto_com_neo4j_e_embeddings_reais(retriever):
    """Valida embedding, busca vetorial, expansão do grafo e formatação."""

    contexto = retriever.recuperar_contexto(
        "meninas na computação e inclusão digital"
    )

    assert isinstance(contexto, str)
    assert contexto.strip(), "O retrieval não retornou nenhum contexto."

    tem_chunks = "=== TRECHOS DE ARTIGOS (CHUNKS) ===" in contexto
    tem_grafo = "=== ENTIDADES ===" in contexto

    assert tem_chunks or tem_grafo, (
        "O contexto não contém nem trechos de artigos nem entidades do grafo."
    )

    if tem_chunks:
        trecho_chunks = contexto.split(
            "=== TRECHOS DE ARTIGOS (CHUNKS) ===",
            maxsplit=1,
        )[1]

        assert "[FONTE:" in trecho_chunks
        assert trecho_chunks.strip()

    if tem_grafo:
        assert "=== RELAÇÕES (TRIPLAS) ===" in contexto
