import functools
import logging
from contextvars import ContextVar
from typing import Any

from langchain_core.tools import BaseTool, tool
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from config import TOOL_ERROR
from retrieval import queries
from retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

# Pergunta original da execução atual. É definida em MeninasDigitaisApp.ask()
# e lida pela instrumentação de cada ferramenta chamada pelo ToolNode.
USER_QUERY: ContextVar[str] = ContextVar("trulens_user_query", default="")

RETRIEVAL = SpanAttributes.SpanType.RETRIEVAL
QUERY = SpanAttributes.RETRIEVAL.QUERY_TEXT
CONTEXTS = SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS

def _retrieval_attributes(ret: Any, exception: Exception | None, *args, **kwargs) -> dict:
    """Anexa a pergunta original e o retorno da ferramenta ao span de retrieval.

    Não usa os argumentos da ferramenta como QUERY: eles podem ser reformulações
    ou decomposições produzidas pelo agente, e não a pergunta feita pelo usuário.
    """
    contexts = []
    if exception is None and ret and ret != TOOL_ERROR:
        contexts = [str(ret)]

    return {
        QUERY: USER_QUERY.get(),
        CONTEXTS: contexts,
    }

def _fontes(itens: list[tuple[str | None, str | None, str]]) -> str:
    """Bloco [FONTE: título | url] por item (formato que o writer/verifier esperam)."""
    return "\n\n".join(
        f"FONTE: [{titulo or 'Sem título'}]({url or ''})\n{corpo}".rstrip()
        for titulo, url, corpo in itens
    )

def _safe_tool(func):
    """Falha de infraestrutura (Neo4j, embeddings) vira texto, não derruba o grafo."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception("Falha na ferramenta %s", func.__name__)
            return TOOL_ERROR

    return wrapper


@tool
@instrument(span_type=RETRIEVAL, attributes=_retrieval_attributes)
@_safe_tool
def pesquisar_contexto_wit(pergunta: str, listar_artigos: bool = False) -> str:
    """
    Busca informações cruciais na base de dados do evento Women in Information Technology (WIT).
    Usa GraphRAG para cruzar similaridade de artigos com relações de entidades no Neo4j.
    Sempre use esta ferramenta para coletar evidências antes de responder ou planejar.

    Args:
        pergunta: A dúvida ou tópico a ser pesquisado (ex: "projetos de mentoria para meninas").
        listar_artigos: True para perguntas de listagem ("quais artigos...?"),
            devolvendo no máximo um trecho por artigo.
    """
    return HybridRetriever.get_instance().recuperar_contexto(
        pergunta, um_por_artigo=listar_artigos
    )


@tool
@instrument(span_type=RETRIEVAL, attributes=_retrieval_attributes)
@_safe_tool
def buscar_projetos_semelhantes(palavras_chave: list[str]) -> str:
    """
    Encontra artigos e projetos no WIT que focam especificamente em um conjunto de palavras-chave.
    Útil para verificar lacunas na literatura ou levantar projetos que já fizeram algo similar.

    Args:
        palavras_chave: Lista de strings com os temas de interesse (ex: ["Mentoria", "Meninas na Computação"]).
    """
    results = queries.find_similar_projects_by_keyword(palavras_chave)
    if not results:
        return "Nenhum projeto estruturalmente semelhante encontrado com essas palavras-chave."
    return _fontes([
        (r["titulo"], r["url"],
        f"Temas em comum: {', '.join(r['temas_em_comum'])} (aderência: {r['aderencia']})")
        for r in results
    ])


@tool
@instrument(span_type=RETRIEVAL, attributes=_retrieval_attributes)
@_safe_tool
def artigos_do_autor(nome: str) -> str:
    """
    Lista os artigos do WIT escritos por uma pessoa (autora ou autor), com link.
    Aceita nome parcial e ignora acentos e maiúsculas (ex: "Laura Jurgina").
    Não tolera erros de digitação: em caso de falha, tente o sobrenome isolado.

    Args:
        nome: Nome ou parte do nome da pessoa.
    """
    results = queries.find_author_works(nome)
    if not results:
        return (
            f"Nenhum autor encontrado com o nome '{nome}'. "
            "Tente outra variação (apenas o sobrenome, por exemplo)."
        )

    return _fontes([
        (a["titulo"], a["url"], f"Autor(a): {r['autor']}")
        for r in results for a in r["artigos"]
    ])



@tool
@instrument(span_type=RETRIEVAL, attributes=_retrieval_attributes)
@_safe_tool
def autores_do_artigo(titulo: str) -> str:
    """
    Encontra um artigo do WIT pelo título (ou palavras distintivas dele)
    e retorna seus autores e link.

    Args:
        titulo: Título completo ou palavras-chave do título.
    """
    results = queries.find_articles_by_title(titulo)
    if not results:
        return (
            f"Nenhum artigo encontrado para '{titulo}'. "
            "Tente menos palavras, só as mais distintivas."
        )
    return _fontes([
        (r["titulo"], r["url"], f"Autores: {', '.join(r['autores'])}")
        for r in results
    ])


RESEARCHER_TOOLS: list[BaseTool] = [
    pesquisar_contexto_wit,
    buscar_projetos_semelhantes,
    artigos_do_autor,
    autores_do_artigo,
]
