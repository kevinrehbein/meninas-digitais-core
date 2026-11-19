import math
import os

import pytest

from retrieval.embedding_model import DIM, SpaceEmbeddings


@pytest.fixture
def embeddings():
    space_id = os.getenv("EMBEDDING_SPACE")

    if not space_id:
        pytest.skip("EMBEDDING_SPACE não configurado")

    return SpaceEmbeddings(
        space_id=space_id,
        retries=1,
        wait=0,
    )


def validar_vetor(vetor):
    assert len(vetor) == DIM
    assert all(math.isfinite(float(valor)) for valor in vetor)
    assert any(abs(float(valor)) > 0 for valor in vetor)


def test_embed_query_real(embeddings):
    vetor = embeddings.embed_query(
        "O Programa Meninas Digitais incentiva a participação "
        "de mulheres na computação."
    )

    validar_vetor(vetor)


def test_embed_documents_real(embeddings):
    textos = [
        "O WIT é um evento sobre mulheres na tecnologia.",
        "Mentoria auxilia estudantes no desenvolvimento acadêmico.",
    ]

    vetores = embeddings.embed_documents(textos)

    assert len(vetores) == len(textos)

    for vetor in vetores:
        validar_vetor(vetor)
