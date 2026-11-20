import logging
import threading

from config import (
    EMBEDDING_SPACE,
    MAX_RELACOES,
    N_ENTIDADES,
    RELACOES_POR_ENTIDADE,
    SCORE_MIN,
    TETO_GRAU,
    TOP_K,
)

from .connection import get_graph
from .embedding_model import SpaceEmbeddings

logger = logging.getLogger(__name__)

Fonte = tuple[str | None, str | None]  # (titulo, url)


class HybridRetriever:
    """GraphRAG: busca vetorial em chunks/entidades + expansão por relações do grafo."""

    _instance: "HybridRetriever | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.embedding_model = SpaceEmbeddings(EMBEDDING_SPACE)  # type: ignore
        self.graph = get_graph()

    @classmethod
    def get_instance(cls) -> "HybridRetriever":
        """Singleton thread-safe: evita recarregar o BGE-M3 a cada chamada."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _chunks_entrada(self, vec: list[float]) -> list[dict]:
        return self.graph.query(
            """
            MATCH (d:Document)
                SEARCH d IN (VECTOR INDEX chunk_embedding FOR $vec LIMIT $k)
                SCORE AS score
            OPTIONAL MATCH (d)-[:PARTE_DE]->(a:Artigo)
            RETURN d.id AS id, d.text AS texto, a.titulo AS titulo, a.url_artigo AS url, score
            ORDER BY score DESC
            """,
            {"k": TOP_K, "vec": vec},
        )

    def _entidades_entrada(self, vec: list[float]) -> list[dict]:
        return self.graph.query(
            """
            MATCH (e:__Entity__)
                SEARCH e IN (VECTOR INDEX entity_embedding FOR $vec LIMIT $n)
                SCORE AS score
            WHERE score >= $min
                AND e.grau <= $teto
                AND EXISTS { (e)-[]-(:__Entity__) }
            RETURN elementId(e) AS eid, e.id AS nome,
                coalesce(e.descricao, '') AS descricao, score
            """,
            {"n": N_ENTIDADES, "vec": vec, "min": SCORE_MIN, "teto": TETO_GRAU},
        )

    def _relacoes_do_grafo(self, seeds: list[dict], vec: list[float]) -> list[dict]:
        return self.graph.query(
            """
            UNWIND $seeds AS s
            MATCH (e:__Entity__) WHERE elementId(e) = s.eid
            MATCH (e)-[r]-(v:__Entity__)
            WHERE v.grau <= $teto AND v.embedding IS NOT NULL
            WITH e, r, s.score * vector.similarity.cosine(v.embedding, $vec) AS pontos
            ORDER BY pontos DESC
            WITH e, collect({r: r, pontos: pontos})[..$por_entidade] AS top
            UNWIND top AS t
            WITH t.r AS r, max(t.pontos) AS pontos
            ORDER BY pontos DESC
            LIMIT $max
            RETURN startNode(r).id AS origem, type(r) AS rel, endNode(r).id AS destino,
                coalesce(startNode(r).descricao, '') AS desc_origem,
                coalesce(endNode(r).descricao, '') AS desc_destino,
                elementId(startNode(r)) AS eid_origem,
                elementId(endNode(r)) AS eid_destino, pontos
            """,
            {
                "seeds": seeds,
                "vec": vec,
                "teto": TETO_GRAU,
                "por_entidade": RELACOES_POR_ENTIDADE,
                "max": MAX_RELACOES,
            },
        )

    def _fontes_fallback(self, eids: set[str]) -> dict[str, list[Fonte]]:
        """Artigo que mais menciona cada entidade (usado só como fallback)."""
        rows = self.graph.query(
            """
            UNWIND $eids AS eid
            MATCH (x:__Entity__) WHERE elementId(x) = eid
            MATCH (d:Document)-[:MENTIONS]->(x)
            MATCH (d)-[:PARTE_DE]->(a:Artigo)
            WITH eid, a.titulo AS titulo, a.url_artigo as url, count(*) AS n
            ORDER BY n DESC
            RETURN eid, collect({titulo: titulo, url: url})[..2] AS fontes
            """,
            {"eids": list(eids)},
        )
        return {r["eid"]: [(f["titulo"], f["url"]) for f in r["fontes"]] for r in rows}

    def _fontes_das_relacoes(
        self, relacoes: list[dict]
    ) -> dict[tuple[str, str], list[Fonte]]:
        """Fonte de cada tripla (eid_origem, eid_destino) -> títulos de artigos.

        1) Artigos com chunk que menciona AMBAS as entidades (comprovam a tripla).
        2) Fallback: principal artigo de cada ponta.
        """
        if not relacoes:
            return {}
        pares = list({(r["eid_origem"], r["eid_destino"]) for r in relacoes})
        rows = self.graph.query(
            """
            UNWIND $pares AS par
            MATCH (o:__Entity__) WHERE elementId(o) = par[0]
            MATCH (d:__Entity__) WHERE elementId(d) = par[1]
            CALL (o, d) {
                MATCH (o)<-[:MENTIONS]-(c:Document)-[:MENTIONS]->(d)
                MATCH (c)-[:PARTE_DE]->(a:Artigo)
                WITH a.titulo AS titulo, a.url_artigo AS url, count(*) AS n
                ORDER BY n DESC
                RETURN collect({titulo: titulo, url: url})[..2] AS fontes
            }
            RETURN par[0] AS eo, par[1] AS ed, fontes
            """,
            {"pares": [list(p) for p in pares]},
        )
        fontes: dict[tuple[str, str], list[Fonte]] = {
            (r["eo"], r["ed"]): [(f["titulo"], f["url"]) for f in r["fontes"]]
            for r in rows
        }

        sem_comum = [p for p in pares if not fontes.get(p)]
        if sem_comum:
            por_ent = self._fontes_fallback({e for p in sem_comum for e in p})
            for eo, ed in sem_comum:
                fontes[(eo, ed)] = list(
                    dict.fromkeys(por_ent.get(eo, []) + por_ent.get(ed, []))
                )
        return fontes

    @staticmethod
    def _filtrar_um_por_artigo(chunks: list[dict]) -> list[dict]:
        vistos: set[str] = set()
        saida = []
        for c in chunks:  # já chegam ordenados por score
            chave = c.get("titulo") or c["id"]
            if chave not in vistos:
                vistos.add(chave)
                saida.append(c)
        return saida

    @staticmethod
    def _formatar_fonte(titulo: str | None, url: str | None) -> str:
        titulo = (titulo or "Sem título").replace("[", "(").replace("]", ")")
        if not url:
            return f"FONTE: [{titulo}]()"
        # parênteses na URL quebram o link Markdown
        url = url.strip().replace("(", "%28").replace(")", "%29").replace(" ", "%20")
        return f"FONTE: [{titulo}]({url})"

    @staticmethod
    def _formatar_relacoes(
        nos_entrada: list[dict],
        relacoes: list[dict],
        fontes: dict[tuple[str, str], list[Fonte]] | None = None,
    ) -> str:
        if not relacoes:
            return ""
        fontes = fontes or {}

        # Entidades únicas: sementes primeiro (ordem do índice), depois vizinhas.
        eids_com_relacao = {r["eid_origem"] for r in relacoes} | {
            r["eid_destino"] for r in relacoes
        }
        entidades = {
            e["eid"]: (e["nome"], e.get("descricao", ""))
            for e in nos_entrada
            if e["eid"] in eids_com_relacao
        }

        triplas: list[str] = []
        for r in relacoes:
            eo, ed = r["eid_origem"], r["eid_destino"]
            entidades.setdefault(eo, (r["origem"], r["desc_origem"]))
            entidades.setdefault(ed, (r["destino"], r["desc_destino"]))

            linha = f"({r['origem']}) -[{r['rel']}]-> ({r['destino']})"
            fontes_tripla = fontes.get((eo, ed), [])
            if fontes_tripla:
                linha += " | " + " ".join(
                    HybridRetriever._formatar_fonte(t, u) for t, u in fontes_tripla
                )
            triplas.append(linha)

        linhas_ent = [
            f"- {nome}: {desc}" if desc else f"- {nome}"
            for nome, desc in entidades.values()
        ]
        return (
            "=== ENTIDADES ===\n"
            + "\n".join(linhas_ent)
            + "\n\n=== RELAÇÕES (TRIPLAS) ===\n"
            + "\n".join(dict.fromkeys(triplas))  # remove triplas idênticas
            + "\n\n"
        )

    @staticmethod
    def formatar_chunk(c: dict) -> str:
        return (
            f"{HybridRetriever._formatar_fonte(c.get('titulo'), c.get('url'))}\n"
            f"{c['texto']}"
        ).rstrip()

    @staticmethod
    def _formatar_chunks(chunks: list[dict]) -> str:
        if not chunks:
            return ""
        corpo = "\n\n".join(HybridRetriever.formatar_chunk(c) for c in chunks)
        return f"=== TRECHOS DE ARTIGOS (CHUNKS) ===\n{corpo}"

    def bloco_grafo(self, vec: list[float]) -> str:
        """Entidades + triplas com fonte. Vazio se não houver entidade/relação."""
        nos = self._entidades_entrada(vec)
        if not nos:
            return ""

        relacoes = self._relacoes_do_grafo(nos, vec)
        return self._formatar_relacoes(
            nos, relacoes, self._fontes_das_relacoes(relacoes)
        )

    def recuperar_contexto(self, pergunta: str, um_por_artigo: bool = False) -> str:
        """Pipeline GraphRAG: retorna o contexto formatado para o LLM."""
        vec = self.embedding_model.embed_query(pergunta)

        chunks = self._chunks_entrada(vec)
        if um_por_artigo:
            chunks = self._filtrar_um_por_artigo(chunks)

        return self.bloco_grafo(vec) + "\n\n" + self._formatar_chunks(chunks)