
from retrieval.hybrid import HybridRetriever


def test_filtrar_um_por_artigo():
    chunks = [
        {"id": "1", "titulo": "Artigo A", "score": 0.95},
        {"id": "2", "titulo": "Artigo A", "score": 0.85},
        {"id": "3", "titulo": "Artigo B", "score": 0.80},
    ]

    result = HybridRetriever._filtrar_um_por_artigo(chunks)

    assert len(result) == 2
    assert result[0]["id"] == "1"
    assert result[1]["id"] == "3"


def test_filtrar_um_por_artigo_sem_titulo():
    chunks = [
        {"id": "1", "titulo": None},
        {"id": "2", "titulo": None},
    ]

    result = HybridRetriever._filtrar_um_por_artigo(chunks)

    assert len(result) == 2


def test_filtrar_um_por_artigo_lista_vazia():
    assert HybridRetriever._filtrar_um_por_artigo([]) == []


def test_formatar_chunk():
    chunk = {
        "titulo": "Artigo sobre computação",
        "url": "https://exemplo.com/artigo",
        "texto": "Conteúdo do artigo.",
    }

    result = HybridRetriever.formatar_chunk(chunk)

    assert "FONTE: [Artigo sobre computação" in result
    assert "https://exemplo.com/artigo" in result
    assert "Conteúdo do artigo." in result


def test_formatar_chunk_sem_metadados():
    chunk = {"texto": "Conteúdo sem metadados."}

    result = HybridRetriever.formatar_chunk(chunk)

    assert "FONTE: [Sem título]()" in result
    assert "Conteúdo sem metadados." in result


def test_formatar_chunks():
    chunks = [
        {
            "titulo": "Artigo A",
            "url": "https://exemplo.com/a",
            "texto": "Conteúdo A.",
        },
        {
            "titulo": "Artigo B",
            "url": "https://exemplo.com/b",
            "texto": "Conteúdo B.",
        },
    ]

    result = HybridRetriever._formatar_chunks(chunks)

    assert "=== TRECHOS DE ARTIGOS (CHUNKS) ===" in result
    assert "Conteúdo A." in result
    assert "Conteúdo B." in result


def test_formatar_chunks_vazio():
    assert HybridRetriever._formatar_chunks([]) == ""


def test_formatar_relacoes_vazio():
    assert HybridRetriever._formatar_relacoes([], []) == ""


def test_formatar_relacoes_com_entidades_e_fontes():
    nos = [
        {
            "eid": "1",
            "nome": "Python",
            "descricao": "Linguagem de programação.",
        },
        {
            "eid": "2",
            "nome": "Meninas Digitais",
            "descricao": "Programa de incentivo à participação feminina.",
        },
    ]

    relacoes = [
        {
            "eid_origem": "1",
            "eid_destino": "2",
            "origem": "Python",
            "destino": "Meninas Digitais",
            "rel": "UTILIZADO_POR",
            "desc_origem": "Linguagem de programação.",
            "desc_destino": "Programa de incentivo à participação feminina.",
        }
    ]

    # depois
    fontes = {
        ("1", "2"): [("Artigo de referência", "https://wit.example/referencia")],
    }

    result = HybridRetriever._formatar_relacoes(
        nos,
        relacoes,
        fontes,
    )

    assert "=== ENTIDADES ===" in result
    assert "=== RELAÇÕES (TRIPLAS) ===" in result
    assert "(Python) -[UTILIZADO_POR]-> (Meninas Digitais)" in result
    assert "FONTE: [Artigo de referência](https://wit.example/referencia)" in result

def test_formatar_relacoes_fonte_sem_url():
    relacoes = [
        {
            "eid_origem": "1", "eid_destino": "2",
            "origem": "A", "destino": "B", "rel": "REL",
            "desc_origem": "", "desc_destino": "",
        }
    ]
    result = HybridRetriever._formatar_relacoes(
        [], relacoes, {("1", "2"): [("Artigo X", None)]}
    )
    assert "FONTE: [Artigo X]()" in result